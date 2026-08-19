#!/usr/bin/env python3
"""
Cybertron 预训练任务监控脚本 (三层告警架构 + 激活值监控)

告警级别:
  P0 (熔断器): nan, 卡死, loss_scale 崩溃, grad_norm 连续异常, loss 发散,
               激活值 absmax 接近 bf16 饱和
  P1 (统计检测): grad_norm spike, loss EMA 漂移, Δ(mtp-lm) 异常, 慢迭代,
                 SCF 异常, RMS 漂移
  P2 (信息): 轻微异常, 长尾模式

阈值基于任务 196333 的 72k 步训练数据校准 (brainstorm consensus)。
激活值 SCF (Standardized Crest Factor) 阈值基于极值统计理论。

数据源:
  - Pod 日志解析: loss, grad_norm, iter_time 等核心指标
  - TensorBoard event 文件: per-layer 激活值统计 (attn/mlp rms/max)
    通过本地 TB 目录或从 Cybertron 文件系统下载

自动检测 rank-0 所在 worker（从 TensorBoard event 文件名推断）。

Usage:
  # 单次检查
  python3 scripts/monitor_training.py --job-id 196333

  # 定期轮询 (每 5 分钟)
  python3 scripts/monitor_training.py --job-id 196333 --interval 300

  # 飞书 webhook 报警
  python3 scripts/monitor_training.py --job-id 196333 --interval 300 --webhook URL

  # 激活值监控 (默认通过 TB scalar API 拉取哨兵层)
  python3 scripts/monitor_training.py --job-id 196333 --num-layers 60

  # 指定本地 TB 目录进行全量激活值监控 (需要 tbparse)
  python3 scripts/monitor_training.py --job-id 196333 --tb-dir /path/to/tensorboard/

  # 跳过激活值监控
  python3 scripts/monitor_training.py --job-id 196333 --no-activation

  # Grafana 系统监控 (GPU%, Temp, Mem 等，自动从 Prometheus 拉取)
  python3 scripts/monitor_training.py --job-id 196333 --grafana-window 7200

  # 跳过 Grafana 系统监控
  python3 scripts/monitor_training.py --job-id 196333 --no-grafana

Dependencies:
  pip install tbparse   # 可选，仅 --tb-dir 模式需要
"""

import argparse
import json
import os
import re
import ssl
import sys
import time
import calendar
import concurrent.futures
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

try:
    from tbparse import SummaryReader
    HAS_TBPARSE = True
except ImportError:
    HAS_TBPARSE = False

BASE_URL = "https://cybertron.modelbest.co/api/job"
LOGS_URL = "https://cybertron.modelbest.co/api/job/logs"
SSL_CTX = ssl.create_default_context()

# Megatron-LM training log iteration line pattern:
# " [datetime] iteration   12345/ 1068115 | consumed samples: ... | ... lm loss: 1.766E+00 | ..."
ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s*\|"
    r".*?elapsed time per iteration \(ms\):\s*([\d.]+)\s*\|"
    r".*?learning rate:\s*([\d.Ee+-]+)\s*\|"
    r".*?global batch size:\s*(\d+)\s*\|"
    r".*?lm loss:\s*([\d.Ee+-]+)\s*\|"
)
MTP_RE = re.compile(r"mtp_\d+ loss:\s*([\d.Ee+-]+)")
GRAD_RE = re.compile(r"grad norm:\s*([\d.Ee+-]+)")
LOSS_SCALE_RE = re.compile(r"loss scale:\s*([\d.Ee+-]+)")
SKIP_RE = re.compile(r"number of skipped iterations:\s*(\d+)")
NAN_RE = re.compile(r"number of nan iterations:\s*(\d+)")

# Calibrated on job 196333 (72k steps, batch=800, lr=4.439e-4, stable training)
THRESHOLDS = {
    "grad_norm_p2": 5.0,
    "grad_norm_p1": 8.0,
    "grad_norm_freq_thresh": 2.0,
    "grad_norm_freq_count": 3,      # ≥3 times above freq_thresh in 100 steps → P1
    "grad_norm_consec_thresh": 1.0,
    "grad_norm_consec_count": 3,    # 3 consecutive steps above consec_thresh → P0
    "loss_ema_p2": 0.05,
    "loss_ema_p1": 0.10,
    "loss_ema_p0": 0.15,
    "loss_diverge": 3.0,
    "delta_p1_high": 0.30,
    "delta_p1_low": 0.10,
    "delta_p0_high": 0.40,
    "delta_p0_low": 0.0,
    "delta_drift_p2": 0.05,
    "iter_p1_single_ms": 10000,
    "iter_p1_consec_ms": 2000,
    "iter_p1_consec_count": 3,
    "iter_p2_slow_ratio": 0.05,
    "loss_scale_p0": 0.0625,
    "ema_window": 500,
}

ACTIVATION_THRESHOLDS = {
    "absmax_p0": 30000.0,
    "rms_drift_p1": 0.30,
    "rms_drift_p2": 0.15,
}

# ---------------------------------------------------------------------------
# Grafana system metrics config
# ---------------------------------------------------------------------------

GRAFANA_CONFIG = {
    "jingsuan_train": {
        "base_url": "https://g.mb1.bj3.paratera.com",
        "datasource_id": 1,
    },
    "paratera_train": {
        "base_url": "https://g.hs1.paratera.com",
        "datasource_id": 1,
    },
}

GRAFANA_SERIES = [
    ("GPU%",          'avg(DCGM_FI_DEV_GPU_UTIL{exported_namespace="$ns", exported_pod=~"$pod"})'),
    ("Tensor Core%",  'avg(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{exported_namespace="$ns", exported_pod=~"$pod"})*100'),
    ("GMem%",         '(sum(DCGM_FI_DEV_FB_USED{exported_namespace="$ns", exported_pod=~"$pod"})/(sum(DCGM_FI_DEV_FB_USED{exported_namespace="$ns", exported_pod=~"$pod"})+sum(DCGM_FI_DEV_FB_FREE{exported_namespace="$ns", exported_pod=~"$pod"})))*100'),
    ("GPU Temp(°C)",  'avg(DCGM_FI_DEV_GPU_TEMP{exported_namespace="$ns", exported_pod=~"$pod"})'),
    ("Power(W/card)", 'avg(DCGM_FI_DEV_POWER_USAGE{exported_namespace="$ns", exported_pod=~"$pod"})'),
    ("CPU%",          '(sum(rate(container_cpu_usage_seconds_total{namespace="$ns", pod=~"$pod"}[5m])) / sum(kube_pod_container_resource_limits{namespace="$ns", pod=~"$pod", resource="cpu"}))*100'),
    ("Mem%",          '(sum(container_memory_working_set_bytes{container!="",namespace="$ns", pod=~"$pod"}) / sum(kube_pod_container_resource_limits{namespace="$ns", pod=~"$pod", resource="memory"}))*100'),
]

# Per-metric anomaly detection config.
# drop/spike: percentage deviation from trimmed mean that triggers alert.
# cv_max: max acceptable coefficient of variation (std/mean).
# climb: if True, alert when this metric monotonically climbs while peers are stable.
GRAFANA_ANOMALY = {
    "GPU%":          {"drop": 15, "cv_max": 0.05, "level": "P1"},
    "Tensor Core%":  {"drop": 20, "cv_max": 0.15, "level": "P2"},
    "GMem%":         {"drop": 10, "cv_max": 0.03, "climb": True, "level": "P1"},
    "GPU Temp(°C)":  {"spike": 15, "climb": True, "level": "P1"},
    "Power(W/card)": {"drop": 20, "cv_max": 0.15, "level": "P2"},
    "CPU%":          {"spike": 30, "level": "P2"},
    "Mem%":          {"climb": True, "level": "P1"},
}



@dataclass
class Alert:
    level: str  # P0 (circuit breaker), P1 (statistical), P2 (informational)
    metric: str
    message: str
    value: float = 0.0
    threshold: float = 0.0

    @property
    def severity(self) -> int:
        return {"P0": 0, "P1": 1, "P2": 2}.get(self.level, 9)


@dataclass
class MonitorState:
    last_step: int = -1
    last_check_time: float = 0.0
    stall_count: int = 0
    rank0_worker: Optional[str] = None
    alert_cooldown: dict = field(default_factory=dict)
    loss_ema_baseline: Optional[float] = None
    # Cache last successful fetch for fallback on transient API failures
    cached_entries: Optional[list] = None
    cached_at: float = 0.0
    last_activation_fetch: float = 0.0


def resolve_token(args) -> str:
    if args.token:
        return args.token
    for path in [".env", os.path.expanduser("~/.cybertron.env")]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("CYBERTRON_TOKEN="):
                        return line.split("=", 1)[1].strip()
    print("ERROR: 未找到 CYBERTRON_TOKEN。", file=sys.stderr)
    print("请用 --token 参数传入，或写入 ~/.cybertron.env", file=sys.stderr)
    sys.exit(1)


def api_get(url: str, token: str, timeout: int = 30, retries: int = 1):
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    last_err = None
    for attempt in range(1 + retries):
        try:
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2)
    raise last_err


def get_job_info(job_id: int, token: str) -> dict:
    data = api_get(f"{BASE_URL}?id={job_id}", token)
    return data.get("data", {}).get("job", {})


# ---------------------------------------------------------------------------
# Rank-0 worker detection
# ---------------------------------------------------------------------------

def _get_project_info(job: dict) -> tuple[str, str, int]:
    """Returns (cluster, project_name, project_id)."""
    cluster = job.get("cluster", "")
    proj = job.get("edges", {}).get("project", {})
    return cluster, proj.get("name", ""), proj.get("id", 0)


def _list_fs_dir(cluster: str, path: str, fs_id: int, token: str) -> list[dict]:
    url = f"https://cybertron.modelbest.co/{cluster}/file/{path}"
    try:
        data = api_get(
            url + "?" + urllib.parse.urlencode({"id": fs_id, "page": 1, "limit": 50}),
            token,
        )
        return data.get("data", {}).get("files", [])
    except Exception as e:
        print(f"  [warn] 文件列表获取失败 ({path}): {e}", file=sys.stderr)
        return []


def detect_rank0_worker(job: dict, token: str) -> Optional[str]:
    """从 TensorBoard event 文件名推断 rank-0 所在的 worker pod。

    event 文件名格式: events.out.tfevents.{timestamp}.{hostname}.{pid}.0
    hostname 即 rank-0 所在 pod。
    """
    cluster, pname, pid = _get_project_info(job)
    job_id = job.get("id")
    fs_id = job.get("filesystem_id")
    if not all([cluster, pname, pid, job_id, fs_id]):
        return None

    tb_dir = f"{cluster}/training/projects/{pid}-{pname}/{job_id}/logs/tensorboard"
    files = _list_fs_dir(cluster, tb_dir, fs_id, token)

    for f in files:
        name = f.get("name", "")
        if name.startswith("events.out.tfevents."):
            parts = name.split(".")
            if len(parts) >= 5:
                hostname = parts[4]
                return hostname
    return None


def _pod_role_index_from_hostname(hostname: str) -> tuple[str, int]:
    """Extract pod role and index from hostname like 'pytorchjob-minicpm5-196333-worker-18'."""
    m = re.search(r"-(master|worker)-(\d+)$", hostname)
    if m:
        return m.group(1), int(m.group(2))
    return "master", 0


# ---------------------------------------------------------------------------
# Pod log fetching
# ---------------------------------------------------------------------------

def fetch_pod_logs(job: dict, token: str, pod_role: str = "master",
                   pod_index: int = 0):
    """Fetch realtime pod logs via Cybertron API."""
    job_id = job.get("id")
    training_type = job.get("training_type", "pytorchjob")
    pname = job.get("edges", {}).get("project", {}).get("name", "")
    pod = f"{training_type}-{pname}-{job_id}-{pod_role}-{pod_index}"

    try:
        data = api_get(f"{LOGS_URL}?" + urllib.parse.urlencode({"id": job_id, "pod": pod}), token)
        result = data.get("data")
        if not result:
            print(f"  [warn] Pod 日志为空 (pod={pod})", file=sys.stderr)
        return result
    except Exception as e:
        print(f"  [warn] Pod 日志获取失败 (pod={pod}): {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_iteration_lines(log_text) -> list[dict]:
    """Parse training iteration lines from Megatron-LM log output.

    log_text can be a string or a list of strings (lines).

    Returns list of dicts sorted by step, each containing:
      step, total_iters, iter_time_ms, lr, batch_size, lm_loss,
      mtp_loss, grad_norm, loss_scale, skipped, nan_iters
    """
    results = []
    lines = log_text if isinstance(log_text, list) else log_text.split("\n")
    for line in lines:
        m = ITER_RE.search(line)
        if not m:
            continue

        entry = {
            "step": int(m.group(1)),
            "total_iters": int(m.group(2)),
            "iter_time_ms": float(m.group(3)),
            "lr": float(m.group(4)),
            "batch_size": int(m.group(5)),
            "lm_loss": float(m.group(6)),
        }

        mtp = MTP_RE.search(line)
        entry["mtp_loss"] = float(mtp.group(1)) if mtp else None

        grad = GRAD_RE.search(line)
        entry["grad_norm"] = float(grad.group(1)) if grad else None

        ls = LOSS_SCALE_RE.search(line)
        entry["loss_scale"] = float(ls.group(1)) if ls else None

        skip = SKIP_RE.search(line)
        entry["skipped"] = int(skip.group(1)) if skip else 0

        nan = NAN_RE.search(line)
        entry["nan_iters"] = int(nan.group(1)) if nan else 0

        results.append(entry)

    results.sort(key=lambda x: x["step"])
    return results


def fetch_metrics_from_logs(job: dict, token: str,
                            state: MonitorState) -> Optional[list[dict]]:
    """Fetch and parse training metrics from pod logs.

    Auto-detects rank-0 worker on first call, caches in state.
    """
    job_id = job.get("id")

    if not state.rank0_worker:
        hostname = detect_rank0_worker(job, token)
        if hostname:
            state.rank0_worker = hostname
            print(f"  [info] rank-0 检测: {hostname}", file=sys.stderr)

    if state.rank0_worker:
        role, index = _pod_role_index_from_hostname(state.rank0_worker)
    else:
        role, index = "master", 0

    log_text = fetch_pod_logs(job, token, role, index)
    if not log_text:
        return None

    entries = parse_iteration_lines(log_text)
    return entries if entries else None


# ---------------------------------------------------------------------------
# TensorBoard API (fallback)
# ---------------------------------------------------------------------------

def get_tb_prefix(job: dict) -> Optional[str]:
    tb = job.get("edges", {}).get("tensorboard", {})
    if tb and tb.get("status") == "Running":
        return tb.get("prefix")
    return None


def tb_get_scalar(tb_base: str, tag: str, token: str,
                  max_points: int = 100, run: str = ".",
                  quiet: bool = False) -> list:
    encoded_tag = urllib.parse.quote(tag, safe="")
    url = (f"https://cybertron.modelbest.co{tb_base}"
           f"/data/plugin/scalars/scalars?tag={encoded_tag}"
           f"&run={urllib.parse.quote(run, safe='')}")
    try:
        data = api_get(url, token)
        if not data:
            return []
        return data[-max_points:] if len(data) > max_points else data
    except Exception as e:
        if not quiet:
            print(f"  [warn] TensorBoard 标量获取失败 ({tag}): {e}",
                  file=sys.stderr)
        return []


def fetch_metrics_from_tb(job: dict, token: str) -> Optional[dict]:
    """Fetch core metrics via TensorBoard API. Returns metrics dict or None."""
    tb_prefix = get_tb_prefix(job)
    if not tb_prefix:
        return None

    metrics = {}
    tag_map = {
        "lm loss": "lm_loss", "mtp_1 loss": "mtp_loss",
        "grad-norm": "grad_norm", "learning-rate": "lr",
        "loss-scale": "loss_scale", "batch-size": "batch_size",
    }
    for tb_tag, key in tag_map.items():
        points = tb_get_scalar(tb_prefix, tb_tag, token, max_points=200)
        if points:
            metrics[key] = {"step": int(points[-1][1]), "value": points[-1][2],
                            "points": [p[2] for p in points]}
    return metrics if metrics else None


# ---------------------------------------------------------------------------
# Per-dataset task loss — lm_loss/<dataset> tags via TensorBoard
# ---------------------------------------------------------------------------

def _list_task_loss_tags(tb_base: str, token: str) -> tuple[list[str], str]:
    """List per-dataset task loss tags (lm_loss/<name>) from TensorBoard.

    Returns (tags, run_name). Only collects from the first run that has
    matching tags to avoid cross-run 404 errors.
    """
    url = (f"https://cybertron.modelbest.co{tb_base}"
           f"/data/plugin/scalars/tags")
    try:
        data = api_get(url, token)
        if not data:
            return [], "."
        for rn, run_tags in data.items():
            if isinstance(run_tags, dict):
                tag_names = list(run_tags.keys())
            elif isinstance(run_tags, list):
                tag_names = run_tags
            else:
                continue
            matched = [t for t in tag_names
                       if t.startswith("lm_loss/") and "vs samples" not in t]
            if matched:
                return matched, rn
        return [], "."
    except Exception as e:
        print(f"  [warn] TB task loss tag 列表获取失败: {e}",
              file=sys.stderr)
        return [], "."


def fetch_task_losses(job: dict, token: str,
                      max_points: int = 100) -> Optional[dict]:
    """Fetch per-dataset task losses from TensorBoard (concurrent).

    Returns dict: tag -> list of (wall_time, step, value), or None.
    """
    tb_prefix = get_tb_prefix(job)
    if not tb_prefix:
        return None
    tags, run_name = _list_task_loss_tags(tb_prefix, token)
    if not tags:
        return None

    print(f"  [info] 发现 {len(tags)} 个 task loss tag，并发拉取中...",
          file=sys.stderr)

    def _fetch_one(tag: str) -> tuple[str, list]:
        points = tb_get_scalar(tb_prefix, tag, token,
                               max_points=max_points, run=run_name,
                               quiet=True)
        return tag, points if points else []

    result: dict[str, list] = {}
    n_fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tags}
        for f in concurrent.futures.as_completed(futures):
            tag, points = f.result()
            if points and len(points) >= 10:
                result[tag] = points
            elif not points:
                n_fail += 1

    if n_fail:
        print(f"  [info] task loss: {len(result)} 成功, {n_fail} 失败/为空",
              file=sys.stderr)
    return result if result else None


def analyze_task_loss_trends(task_data: dict,
                             top_n: int = 5) -> Optional[dict]:
    """Compare recent half vs earlier half mean for each dataset.

    Returns summary with top risers (loss going up) and fallers (loss
    going down), sorted by percentage change.
    """
    trends = []
    for tag, points in task_data.items():
        values = [p[2] for p in points]
        mid = len(values) // 2
        earlier = values[:mid]
        recent = values[mid:]
        mean_earlier = sum(earlier) / len(earlier)
        mean_recent = sum(recent) / len(recent)
        if mean_earlier == 0:
            continue

        abs_change = mean_recent - mean_earlier
        pct_change = abs_change / mean_earlier * 100
        dataset_name = tag.split("/", 1)[1] if "/" in tag else tag

        step_from = int(points[0][1])
        step_to = int(points[-1][1])

        trends.append({
            "dataset": dataset_name,
            "tag": tag,
            "mean_recent": mean_recent,
            "mean_earlier": mean_earlier,
            "abs_change": abs_change,
            "pct_change": pct_change,
            "latest_val": values[-1],
            "step_from": step_from,
            "step_to": step_to,
        })

    if not trends:
        return None

    risers = sorted([t for t in trends if t["abs_change"] > 0],
                    key=lambda x: x["pct_change"], reverse=True)[:top_n]
    fallers = sorted([t for t in trends if t["abs_change"] < 0],
                     key=lambda x: x["pct_change"])[:top_n]

    step_from = min(t["step_from"] for t in trends)
    step_to = max(t["step_to"] for t in trends)

    return {
        "total_datasets": len(trends),
        "risers": risers,
        "fallers": fallers,
        "step_range": (step_from, step_to),
    }


# ---------------------------------------------------------------------------
# Activation stats — TB scalar API (primary) + tfevents local (fallback)
# ---------------------------------------------------------------------------

ACTIVATION_TAG_RE = re.compile(
    r"^(?:attention_output_(?:rms|max)|mlp_output_(?:rms|max))/layer_(\d+)$"
)

_ACTIVATION_TAG_PARTS = [
    ("attention_output_rms", "attn_rms"),
    ("attention_output_max", "attn_max"),
    ("mlp_output_rms", "mlp_rms"),
    ("mlp_output_max", "mlp_max"),
]


def _fetch_tb_activation_tags(tb_base: str, token: str) -> tuple[list[str], dict]:
    """List activation-related scalar tags from TensorBoard API.

    Returns (activation_tags, run_tag_map) where run_tag_map maps
    run_name -> list of matching activation tags.
    """
    url = (f"https://cybertron.modelbest.co{tb_base}"
           f"/data/plugin/scalars/tags")
    try:
        data = api_get(url, token)
        if not data:
            print(f"  [info] TB tags 端点返回空", file=sys.stderr)
            return [], {}
        run_tag_map: dict[str, list[str]] = {}
        all_act_tags: list[str] = []
        for run_name, run_tags in data.items():
            if isinstance(run_tags, dict):
                tag_names = list(run_tags.keys())
            elif isinstance(run_tags, list):
                tag_names = run_tags
            else:
                continue
            n_total = len(tag_names)
            act = [t for t in tag_names if ACTIVATION_TAG_RE.match(t)]
            if act:
                run_tag_map[run_name] = act
                all_act_tags.extend(act)
            print(f"  [info] TB run '{run_name}': {n_total} tags, "
                  f"{len(act)} 激活值相关", file=sys.stderr)
        return all_act_tags, run_tag_map
    except Exception as e:
        print(f"  [warn] TB tag 列表获取失败: {e}", file=sys.stderr)
        return [], {}


def _pick_sentinel_layers(activation_tags: list[str],
                          num_layers: int) -> list[int]:
    """Pick sentinel layers (first, ~1/4, ~1/2, ~3/4, last)."""
    if activation_tags:
        layers = sorted({int(ACTIVATION_TAG_RE.match(t).group(1))
                         for t in activation_tags
                         if ACTIVATION_TAG_RE.match(t)})
        if layers:
            n = len(layers)
            if n <= 5:
                return layers
            indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
            return [layers[i] for i in indices]

    if num_layers <= 5:
        return list(range(num_layers))
    return sorted(set([0, num_layers // 4, num_layers // 2,
                       3 * num_layers // 4, num_layers - 1]))


def fetch_activation_from_tb_api(tb_base: str, token: str,
                                 sentinel_layers: list[int],
                                 run: str = ".") -> dict:
    """Fetch activation stats for sentinel layers via TB scalar API.

    Returns layer_data in same format as parse_activation_stats:
    {layer_idx: {"attn_rms": [(step, value), ...], ...}}
    """
    layer_data: dict[int, dict] = {}
    for layer_idx in sentinel_layers:
        layer_info: dict[str, list] = {}
        for tb_prefix, key in _ACTIVATION_TAG_PARTS:
            tag = f"{tb_prefix}/layer_{layer_idx}"
            points = tb_get_scalar(tb_base, tag, token,
                                   max_points=200, run=run)
            if points:
                layer_info[key] = [(int(p[1]), p[2]) for p in points]
        if layer_info:
            for key in ("attn_rms", "attn_max", "mlp_rms", "mlp_max"):
                layer_info.setdefault(key, [])
            layer_data[layer_idx] = layer_info

    if layer_data:
        n = len(layer_data)
        total = sum(len(v) for ld in layer_data.values() for v in ld.values())
        print(f"  [info] 激活值数据 (TB API, run='{run}'): {n} 哨兵层, "
              f"{total} 个数据点", file=sys.stderr)
    return layer_data


def _download_tb_events(job: dict, token: str, cache_dir: Path,
                        max_age: int = 1800) -> Optional[Path]:
    """Download TensorBoard event files from Cybertron to local cache.

    Uses the same /api/job/ endpoint family as pod log fetching.
    Falls back to /{cluster}/file/ endpoint if needed.

    Returns path to local cache directory containing tfevents files, or None.
    Skips download if cache is fresh (< max_age seconds old).
    """
    marker = cache_dir / ".last_download"
    if marker.exists():
        age = time.time() - marker.stat().st_mtime
        if age < max_age:
            tfevents = list(cache_dir.glob("events.out.tfevents.*"))
            if tfevents:
                return cache_dir

    cluster, pname, pid = _get_project_info(job)
    job_id = job.get("id")
    fs_id = job.get("filesystem_id")
    if not all([cluster, pname, pid, job_id, fs_id]):
        return None

    tb_remote = (f"{cluster}/training/projects/{pid}-{pname}"
                 f"/{job_id}/logs/tensorboard")
    files = _list_fs_dir(cluster, tb_remote, fs_id, token)
    if not files:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = False
    for f in files:
        fname = f.get("name", "")
        if not fname.startswith("events.out.tfevents."):
            continue
        local_path = cache_dir / fname
        remote_size = f.get("size", -1)
        if local_path.exists() and remote_size > 0:
            if local_path.stat().st_size >= remote_size:
                downloaded = True
                continue

        file_path = f"{tb_remote}/{fname}"
        url = (f"https://cybertron.modelbest.co/file/{file_path}"
               f"?{urllib.parse.urlencode({'id': fs_id, 'download': 1})}")
        try:
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {token}",
            })
            with urllib.request.urlopen(req, context=SSL_CTX,
                                        timeout=300) as resp:
                with open(local_path, "wb") as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            actual = local_path.stat().st_size
            if actual < 1024:
                print(f"  [warn] 下载异常: {fname} 仅 {actual} 字节 "
                      f"(预期 {remote_size})", file=sys.stderr)
                local_path.unlink(missing_ok=True)
                continue
            downloaded = True
            size_mb = actual / 1e6
            print(f"  [info] 已下载 {fname} ({size_mb:.1f}MB)",
                  file=sys.stderr)
        except Exception as e:
            print(f"  [warn] TB event 下载失败 ({fname}): {e}",
                  file=sys.stderr)
            local_path.unlink(missing_ok=True)

    if downloaded:
        marker.touch()
        return cache_dir
    return None


_LOCAL_TAG_MAP = {
    "attention_output_rms": "attn_rms",
    "attention_output_max": "attn_max",
    "mlp_output_rms": "mlp_rms",
    "mlp_output_max": "mlp_max",
}

_ACTIVATION_KEYS = ("attn_rms", "attn_max", "mlp_rms", "mlp_max")


def parse_activation_stats(tb_path: Path) -> dict:
    """Parse per-layer activation stats from TensorBoard event files.

    Returns:
        {layer_idx(int): {
            "attn_rms": [(step, value), ...],
            "attn_max": [(step, value), ...],
            "mlp_rms":  [(step, value), ...],
            "mlp_max":  [(step, value), ...],
        }}
    """
    if not HAS_TBPARSE:
        print("  [warn] tbparse 未安装，跳过激活值监控。pip install tbparse",
              file=sys.stderr)
        return {}

    try:
        reader = SummaryReader(str(tb_path), pivot=False)
        df = reader.scalars
    except Exception as e:
        print(f"  [warn] TB event 解析失败: {e}", file=sys.stderr)
        return {}

    if df is None or df.empty:
        return {}

    mask = df["tag"].str.match(ACTIVATION_TAG_RE.pattern)
    act_df = df[mask]

    layer_data: dict[int, dict] = {}
    for _, row in act_df.iterrows():
        m = ACTIVATION_TAG_RE.match(row["tag"])
        if not m:
            continue
        layer_idx = int(m.group(1))
        tag_prefix = row["tag"].rsplit("/layer_", 1)[0]
        key = _LOCAL_TAG_MAP.get(tag_prefix)
        if key is None:
            continue
        if layer_idx not in layer_data:
            layer_data[layer_idx] = {k: [] for k in _ACTIVATION_KEYS}
        layer_data[layer_idx][key].append(
            (int(row["step"]), float(row["value"])))

    for layer_idx in layer_data:
        for key in layer_data[layer_idx]:
            layer_data[layer_idx][key].sort()

    if layer_data:
        n_layers = len(layer_data)
        total_points = sum(
            len(v) for ld in layer_data.values() for v in ld.values())
        print(f"  [info] 激活值数据 (本地): {n_layers} 层, "
              f"{total_points} 个数据点", file=sys.stderr)
    return layer_data


# ---------------------------------------------------------------------------
# Grafana / Prometheus queries + anomaly detection
# ---------------------------------------------------------------------------

def _grafana_query_range(base_url, ds_id, expr, start_ts, end_ts, step=60):
    """Query Prometheus via Grafana proxy. Returns [(ts, value), ...] or None."""
    url = (f"{base_url}/api/datasources/proxy/{ds_id}/api/v1/query_range?"
           + urllib.parse.urlencode({
               "query": expr, "start": str(start_ts),
               "end": str(end_ts), "step": str(step),
           }))
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") != "success":
                return None
            results = data.get("data", {}).get("result", [])
            if not results:
                return None
            return [(float(ts), float(val)) for ts, val in results[0]["values"]
                    if val != "NaN"]
    except Exception:
        return None


def _trimmed_stats(values, trim_pct=20):
    """Compute mean/std, skipping first trim_pct% (startup) and sorting to drop extremes."""
    if not values:
        return 0, 0
    n = len(values)
    skip_start = max(1, n * trim_pct // 100)
    trimmed = values[skip_start:]
    if not trimmed:
        trimmed = values
    # Also drop top/bottom 5% by value to remove transient spikes
    s = sorted(trimmed)
    edge = max(1, len(s) // 20)
    if len(s) > edge * 2 + 2:
        s = s[edge:-edge]
    mean = sum(s) / len(s)
    var = sum((v - mean) ** 2 for v in s) / len(s)
    return mean, var ** 0.5


def _detect_trend(values):
    """Detect monotonic trend by comparing quarter means.

    Skips the first quarter to avoid startup artifacts (0→steady state).
    Compares Q2 vs Q3 vs Q4 for monotonicity, reports change relative to Q2.
    Returns (direction, relative_change).
    """
    n = len(values)
    if n < 16:
        return "stable", 0.0
    q = n // 4
    # Use Q2-Q4 only (skip Q1 = startup phase)
    qmeans = [sum(values[i * q:(i + 1) * q]) / q for i in range(1, 4)]
    if qmeans[0] == 0:
        return "stable", 0.0
    change = (qmeans[2] - qmeans[0]) / abs(qmeans[0])
    mono_up = all(qmeans[i] < qmeans[i + 1] for i in range(2))
    mono_down = all(qmeans[i] > qmeans[i + 1] for i in range(2))
    if mono_up and change > 0.05:
        return "up", change
    if mono_down and change < -0.05:
        return "down", change
    return "stable", change


def fetch_grafana_series(job, lookback_sec=3600, step=60):
    """Fetch Grafana time series for a job. Returns {metric: [(ts, val), ...]}."""
    cluster = job.get("cluster", "")
    cfg = GRAFANA_CONFIG.get(cluster)
    if not cfg:
        return None

    ns = job.get("namespace", "training")
    pname = job.get("edges", {}).get("project", {}).get("name", "")
    job_id = job.get("id")
    ttype = job.get("training_type", "pytorchjob")
    pod_pat = f"{ttype}-{pname}-{job_id}-.*"

    end_ts = int(time.time())
    start_raw = job.get("start_time", "")
    if start_raw:
        m = re.match(
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?'
            r'([+-]\d{2}:\d{2}|Z)?', str(start_raw))
        if m:
            dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
            tz = m.group(2)
            if tz and tz != "Z":
                sign = 1 if tz[0] == '+' else -1
                dt -= timedelta(hours=sign * int(tz[1:3]),
                                minutes=sign * int(tz[4:6]))
            job_start = calendar.timegm(dt.timetuple())
        else:
            job_start = end_ts - lookback_sec
    else:
        job_start = end_ts - lookback_sec

    start_ts = max(job_start, end_ts - lookback_sec)

    series = {}
    for name, expr_tpl in GRAFANA_SERIES:
        expr = expr_tpl.replace("$ns", ns).replace("$pod", pod_pat)
        data = _grafana_query_range(
            cfg["base_url"], cfg["datasource_id"], expr, start_ts, end_ts, step)
        if data and len(data) >= 3:
            series[name] = data
    return series if series else None


def analyze_grafana(series):
    """Anomaly detection on Grafana system metrics.

    Detects: sudden drop, sudden spike, instability (high CV),
    and monotonic climb while peers are stable.
    Returns (summary_dict, alerts).
    """
    alerts = []
    summary = {}
    stability_scores = {}

    for name, points in series.items():
        vals = [v for _, v in points]
        cfg = GRAFANA_ANOMALY.get(name, {})
        level = cfg.get("level", "P2")
        mean, std = _trimmed_stats(vals)
        cv = std / mean if mean > 0 else 0
        direction, trend_change = _detect_trend(vals)

        summary[name] = {
            "mean": mean, "std": std, "cv": cv,
            "min": min(vals), "max": max(vals),
            "trend": direction, "trend_pct": trend_change,
            "points": len(vals),
        }
        stability_scores[name] = cv

        if mean == 0:
            continue

        # Sudden drop: any point below mean * (1 - drop_pct/100)
        # Skip first 25% to avoid startup artifacts (0 → steady state)
        skip = max(2, len(vals) // 4)
        stable_vals = vals[skip:]
        if not stable_vals:
            continue

        drop_pct = cfg.get("drop")
        if drop_pct is not None:
            floor = mean * (1 - drop_pct / 100)
            worst = min(stable_vals)
            if worst < floor:
                alerts.append(Alert(level, f"grafana_{name}",
                    f"{name} 突降: 最低 {worst:.1f}, 均值 {mean:.1f} "
                    f"(偏离 {(mean - worst) / mean * 100:.0f}%)",
                    value=worst, threshold=floor))

        # Sudden spike: any point above mean * (1 + spike_pct/100)
        spike_pct = cfg.get("spike")
        if spike_pct is not None:
            ceil = mean * (1 + spike_pct / 100)
            worst = max(stable_vals)
            if worst > ceil:
                alerts.append(Alert(level, f"grafana_{name}",
                    f"{name} 突升: 最高 {worst:.1f}, 均值 {mean:.1f} "
                    f"(偏离 +{(worst - mean) / mean * 100:.0f}%)",
                    value=worst, threshold=ceil))

        # Instability: coefficient of variation too high
        cv_max = cfg.get("cv_max")
        if cv_max is not None and cv > cv_max:
            alerts.append(Alert(level, f"grafana_{name}",
                f"{name} 不稳定: CV={cv:.3f} > {cv_max} "
                f"(std={std:.1f}, mean={mean:.1f})",
                value=cv, threshold=cv_max))

    # Cross-metric: one metric climbing while peers are stable
    climbers = []
    stable_count = 0
    for name, s in summary.items():
        cfg = GRAFANA_ANOMALY.get(name, {})
        if not cfg.get("climb"):
            if s["cv"] < 0.10:
                stable_count += 1
            continue
        if s["trend"] == "up" and abs(s["trend_pct"]) > 0.05:
            climbers.append((name, s["trend_pct"]))

    if climbers and stable_count >= 3:
        for name, pct in climbers:
            level = GRAFANA_ANOMALY.get(name, {}).get("level", "P1")
            alerts.append(Alert(level, f"grafana_{name}",
                f"{name} 持续爬升 (+{pct:.1%}), 而其他 {stable_count} "
                f"项指标稳定",
                value=pct, threshold=0.05))

    alerts.sort(key=lambda a: a.severity)
    return summary, alerts


_ACTIVATION_UNAVAILABLE = "_unavailable"
_ACTIVATION_SKIPPED = "_skipped"


def fetch_activation_stats(job: Optional[dict], token: str,
                           state: MonitorState,
                           tb_dir: Optional[str] = None,
                           activation_interval: int = 1800,
                           num_layers: int = 60) -> dict:
    """Fetch activation stats.  Priority: --tb-dir (local) > TB scalar API.

    Returns layer_data dict on success, or a status-only dict
    (keys prefixed with '_') on failure/skip so the caller can report it.
    """
    elapsed = time.time() - state.last_activation_fetch
    if elapsed < activation_interval and state.last_activation_fetch > 0:
        remaining = int(activation_interval - elapsed)
        return {_ACTIVATION_SKIPPED: True,
                "_status": "skipped",
                "_reason": f"冷却中 ({remaining}s 后刷新)"}

    # Priority 1: local --tb-dir (needs tbparse)
    if tb_dir:
        if not HAS_TBPARSE:
            return {_ACTIVATION_UNAVAILABLE: True,
                    "_status": "unavailable",
                    "_reason": "--tb-dir 需要 tbparse 库"}
        tb_path = Path(tb_dir)
        if tb_path.exists():
            result = parse_activation_stats(tb_path)
            if result:
                state.last_activation_fetch = time.time()
                return result
            return {_ACTIVATION_UNAVAILABLE: True,
                    "_status": "unavailable",
                    "_reason": f"TB 目录存在但无激活值 tag: {tb_dir}"}
        print(f"  [warn] TB 目录不存在: {tb_dir}", file=sys.stderr)
        return {_ACTIVATION_UNAVAILABLE: True,
                "_status": "unavailable",
                "_reason": f"TB 目录不存在: {tb_dir}"}

    if not job:
        return {_ACTIVATION_SKIPPED: True,
                "_status": "skipped", "_reason": "无 job 信息"}

    # Priority 2: TB scalar API (no tbparse needed)
    tb_base = get_tb_prefix(job)
    if tb_base:
        activation_tags, run_tag_map = _fetch_tb_activation_tags(
            tb_base, token)
        if activation_tags:
            sentinel = _pick_sentinel_layers(activation_tags, num_layers)
            act_run = next(iter(run_tag_map)) if run_tag_map else "."
            layer_data = fetch_activation_from_tb_api(
                tb_base, token, sentinel, run=act_run)
            if layer_data:
                state.last_activation_fetch = time.time()
                return layer_data
            print("  [warn] TB API 拿到 tag 但获取数据失败",
                  file=sys.stderr)

    # Priority 3: download tfevents + local parse (needs tbparse)
    if HAS_TBPARSE:
        job_id = job.get("id", "unknown")
        cache_dir = (Path.home() / ".cache" / "monitor_training"
                     / str(job_id) / "tb")
        local_tb = _download_tb_events(
            job, token, cache_dir, max_age=activation_interval)
        if local_tb:
            result = parse_activation_stats(local_tb)
            if result:
                state.last_activation_fetch = time.time()
                return result
            print("  [info] tfevents 已下载但未找到激活值 tag",
                  file=sys.stderr)

    reason = ("TB API 无激活值 tag"
              if tb_base else "TensorBoard 服务未运行")
    if not HAS_TBPARSE:
        reason += "; tfevents 回退需要 tbparse (pip install tbparse)"
    return {_ACTIVATION_UNAVAILABLE: True,
            "_status": "unavailable", "_reason": reason}


# ---------------------------------------------------------------------------
# Anomaly detection — three-tier architecture
# ---------------------------------------------------------------------------

def check_job_status(job: dict) -> list[Alert]:
    alerts = []
    status = job.get("status", "Unknown")
    if status in ("Failed", "Killed"):
        reason = job.get("killed_reason", "未知原因")
        alerts.append(Alert("P0", "job_status", f"任务已{status}: {reason}"))
    elif status == "Queued":
        alerts.append(Alert("P2", "job_status", "任务仍在排队中"))
    elif status != "Running":
        alerts.append(Alert("P1", "job_status", f"任务状态异常: {status}"))
    return alerts


def _max_consecutive_above(values: list, threshold: float) -> int:
    best = cur = 0
    for v in values:
        if v is not None and v > threshold:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _check_loss_ema(losses: list[float], current_loss: float,
                    cfg: dict, state: MonitorState, alerts: list[Alert]):
    """Loss EMA drift detection + single-step diverge."""
    if current_loss > cfg["loss_diverge"]:
        alerts.append(Alert("P0", "loss_trend",
            f"Loss 异常高: {current_loss:.4f} > {cfg['loss_diverge']}",
            value=current_loss, threshold=cfg["loss_diverge"]))
        return

    w = min(cfg["ema_window"], len(losses) // 3)
    if w < 50:
        return

    recent_mean = sum(losses[-w:]) / w
    baseline_mean = (
        sum(losses[-w * 2:-w]) / w if len(losses) >= w * 2
        else sum(losses[:w]) / w
    )

    if state.loss_ema_baseline is None:
        state.loss_ema_baseline = baseline_mean

    drift = recent_mean - baseline_mean
    if drift > cfg["loss_ema_p0"]:
        alerts.append(Alert("P0", "loss_trend",
            f"Loss EMA 严重上升: 近{w}步={recent_mean:.4f}, "
            f"基线={baseline_mean:.4f}, 漂移={drift:+.4f}",
            value=drift, threshold=cfg["loss_ema_p0"]))
    elif drift > cfg["loss_ema_p1"]:
        alerts.append(Alert("P1", "loss_trend",
            f"Loss EMA 上升: 近{w}步={recent_mean:.4f}, "
            f"基线={baseline_mean:.4f}, 漂移={drift:+.4f}",
            value=drift, threshold=cfg["loss_ema_p1"]))
    elif drift > cfg["loss_ema_p2"]:
        alerts.append(Alert("P2", "loss_trend",
            f"Loss EMA 轻微上升: 近{w}步={recent_mean:.4f}, "
            f"基线={baseline_mean:.4f}, 漂移={drift:+.4f}",
            value=drift, threshold=cfg["loss_ema_p2"]))


def _check_iter_time(entries: list[dict], latest: dict,
                     cfg: dict, alerts: list[Alert]):
    """Iter time anomaly detection: single extreme, consecutive slow, long tail."""
    iter_time = latest.get("iter_time_ms", 0)

    if iter_time > cfg["iter_p1_single_ms"]:
        alerts.append(Alert("P1", "iter_time",
            f"迭代极慢: {iter_time:.0f}ms > "
            f"{cfg['iter_p1_single_ms'] / 1000:.0f}s",
            value=iter_time, threshold=cfg["iter_p1_single_ms"]))
        return

    recent_times = [e.get("iter_time_ms", 0) for e in entries[-10:]]
    consec_slow = _max_consecutive_above(recent_times, cfg["iter_p1_consec_ms"])
    if consec_slow >= cfg["iter_p1_consec_count"]:
        tail = [f"{t:.0f}" for t in recent_times[-5:]]
        alerts.append(Alert("P1", "iter_time",
            f"连续 {consec_slow} 步 > "
            f"{cfg['iter_p1_consec_ms'] / 1000:.0f}s "
            f"(最近: {tail})",
            value=consec_slow, threshold=cfg["iter_p1_consec_count"]))
        return

    iter_times = [e["iter_time_ms"] for e in entries if e.get("iter_time_ms")]
    n = min(1000, len(iter_times))
    if n >= 100:
        window = iter_times[-n:]
        slow = sum(1 for t in window if t > cfg["iter_p1_consec_ms"])
        ratio = slow / n
        if ratio > cfg["iter_p2_slow_ratio"]:
            alerts.append(Alert("P2", "iter_time",
                f"慢迭代比例偏高: 近{n}步中 {slow} 次 > "
                f"{cfg['iter_p1_consec_ms'] / 1000:.0f}s ({ratio:.1%})",
                value=ratio, threshold=cfg["iter_p2_slow_ratio"]))


def _check_delta(entries: list[dict], delta: float,
                 cfg: dict, alerts: list[Alert]):
    """Δ(mtp-lm) bounds and trend drift detection."""
    if delta < cfg["delta_p0_low"] or delta > cfg["delta_p0_high"]:
        alerts.append(Alert("P0", "delta_mtp_lm",
            f"Δ(mtp-lm) = {delta:.4f}, 超出安全范围 "
            f"[{cfg['delta_p0_low']}, {cfg['delta_p0_high']}]"))
        return

    if delta > cfg["delta_p1_high"] or delta < cfg["delta_p1_low"]:
        alerts.append(Alert("P1", "delta_mtp_lm",
            f"Δ(mtp-lm) = {delta:.4f}, 偏离正常 "
            f"[{cfg['delta_p1_low']}, {cfg['delta_p1_high']}]"))
        return

    deltas = [e["mtp_loss"] - e["lm_loss"]
              for e in entries if e.get("mtp_loss") is not None]
    if len(deltas) < 200:
        return
    n = min(cfg["ema_window"], len(deltas) // 3)
    if n < 50:
        return
    recent_d = sum(deltas[-n:]) / n
    earlier_d = sum(deltas[:n]) / n
    drift = abs(recent_d - earlier_d)
    if drift > cfg["delta_drift_p2"]:
        alerts.append(Alert("P2", "delta_mtp_lm",
            f"Δ(mtp-lm) 趋势漂移: {earlier_d:.4f} → "
            f"{recent_d:.4f} (|变化|={drift:.4f})"))


def _check_grad_norm(entries: list[dict], latest: dict,
                     cfg: dict, alerts: list[Alert]) -> bool:
    """Grad norm: consecutive P0, spike/frequency P1, mild P2. Returns True if P0 fired."""
    recent_grads = [e.get("grad_norm") for e in entries[-20:]
                    if e.get("grad_norm") is not None]
    consec = _max_consecutive_above(recent_grads, cfg["grad_norm_consec_thresh"])
    if consec >= cfg["grad_norm_consec_count"]:
        alerts.append(Alert("P0", "grad_norm",
            f"grad_norm 连续 {consec} 步 > {cfg['grad_norm_consec_thresh']}",
            value=consec, threshold=cfg["grad_norm_consec_count"]))
        return True

    gn = latest.get("grad_norm")
    if gn is None:
        return False

    if gn > cfg["grad_norm_p1"]:
        alerts.append(Alert("P1", "grad_norm",
            f"grad_norm 极端 spike: {gn:.3f} > {cfg['grad_norm_p1']}",
            value=gn, threshold=cfg["grad_norm_p1"]))
        return False

    last_100 = [e.get("grad_norm") for e in entries[-100:]
                if e.get("grad_norm") is not None]
    freq = sum(1 for g in last_100 if g > cfg["grad_norm_freq_thresh"])
    if freq >= cfg["grad_norm_freq_count"]:
        alerts.append(Alert("P1", "grad_norm",
            f"grad_norm 高频异常: 近{len(last_100)}步中 "
            f"{freq} 次 > {cfg['grad_norm_freq_thresh']}",
            value=freq, threshold=cfg["grad_norm_freq_count"]))
        return False

    if gn > cfg["grad_norm_p2"]:
        alerts.append(Alert("P2", "grad_norm",
            f"grad_norm spike: {gn:.3f} > {cfg['grad_norm_p2']}",
            value=gn, threshold=cfg["grad_norm_p2"]))

    return False


def analyze_metrics(entries: list[dict], state: MonitorState,
                    thresholds: Optional[dict] = None) -> tuple[dict, list[Alert]]:
    """Three-tier anomaly detection (brainstorm consensus architecture).

    P0: Circuit breakers — nan, stall, loss_scale collapse, consecutive grad_norm
    P1: Statistical detection — grad_norm spike, loss EMA drift, Δ anomaly, slow iters
    P2: Informational — mild anomalies, long tail patterns
    """
    alerts: list[Alert] = []
    cfg = {**THRESHOLDS, **(thresholds or {})}
    latest = entries[-1]
    current_step = latest["step"]

    # ── Build summary metrics for report ──────────────────────────
    metrics = {
        "lm loss": {"step": current_step, "value": latest["lm_loss"]},
        "grad-norm": {"step": current_step, "value": latest.get("grad_norm", 0)},
        "learning-rate": {"step": current_step, "value": latest.get("lr", 0)},
        "loss-scale": {"step": current_step, "value": latest.get("loss_scale", 0)},
        "iter-time-ms": {"step": current_step, "value": latest.get("iter_time_ms", 0)},
    }
    if latest.get("mtp_loss") is not None:
        metrics["mtp_1 loss"] = {"step": current_step, "value": latest["mtp_loss"]}
        delta = latest["mtp_loss"] - latest["lm_loss"]
        metrics["delta(mtp-lm)"] = {"step": current_step, "value": delta}
    else:
        delta = None

    # ── P0: Circuit breakers ──────────────────────────────────────

    if latest.get("nan_iters", 0) > 0:
        alerts.append(Alert("P0", "nan_iters",
            f"NaN iterations: {latest['nan_iters']}", value=latest["nan_iters"]))

    if state.last_step >= 0 and current_step == state.last_step:
        state.stall_count += 1
        if state.stall_count >= 2:
            elapsed = time.time() - state.last_check_time
            alerts.append(Alert("P0", "training_stall",
                f"训练卡死 {state.stall_count} 个周期 (~{elapsed / 60:.0f}分钟), "
                f"step 停在 {current_step}"))
    else:
        state.stall_count = 0

    ls = latest.get("loss_scale")
    if ls is not None and 0 < ls < cfg["loss_scale_p0"]:
        alerts.append(Alert("P0", "loss_scale",
            f"Loss scale 崩溃: {ls:.4g} < {cfg['loss_scale_p0']}",
            value=ls, threshold=cfg["loss_scale_p0"]))

    _check_grad_norm(entries, latest, cfg, alerts)

    losses = [e["lm_loss"] for e in entries]
    _check_loss_ema(losses, latest["lm_loss"], cfg, state, alerts)

    if delta is not None:
        _check_delta(entries, delta, cfg, alerts)

    # ── P1: Statistical detection ─────────────────────────────────

    if latest.get("skipped", 0) > 0:
        alerts.append(Alert("P1", "skipped_iters",
            f"跳过迭代: {latest['skipped']}", value=latest["skipped"]))

    _check_iter_time(entries, latest, cfg, alerts)

    loss_scales = [e.get("loss_scale") for e in entries[-10:]
                   if e.get("loss_scale") is not None]
    if len(loss_scales) >= 2 and loss_scales[-2] > 0:
        if (loss_scales[-1] < loss_scales[-2] * 0.5
                and loss_scales[-1] >= cfg["loss_scale_p0"]):
            alerts.append(Alert("P1", "loss_scale",
                f"Loss scale 骤降: {loss_scales[-2]:.0f} → {loss_scales[-1]:.0f}"))

    # ── Summary stats (附加到 metrics 供 report 显示) ─────────────

    all_grads = [e.get("grad_norm") for e in entries
                 if e.get("grad_norm") is not None]
    if all_grads:
        metrics["grad_stats"] = {
            "mean": sum(all_grads) / len(all_grads),
            "max": max(all_grads),
            "above_1": sum(1 for g in all_grads if g > 1.0),
            "count": len(all_grads),
        }

    iter_times = [e["iter_time_ms"] for e in entries if e.get("iter_time_ms")]
    if iter_times:
        st = sorted(iter_times)
        metrics["iter_stats"] = {
            "p50": st[len(st) // 2],
            "p99": st[min(int(len(st) * 0.99), len(st) - 1)],
            "slow_count": sum(1 for t in iter_times if t > 2000),
            "count": len(iter_times),
        }

    state.last_step = current_step
    state.last_check_time = time.time()
    alerts.sort(key=lambda a: a.severity)
    return metrics, alerts


# ---------------------------------------------------------------------------
# Activation stats analysis (SCF / drift / absmax)
# ---------------------------------------------------------------------------

def analyze_activation(layer_data: dict,
                       thresholds: Optional[dict] = None
                       ) -> tuple[dict, list[Alert]]:
    """Analyze per-layer activation stats for anomalies.

    Available signals (from states_logger):
      - attn_rms / attn_max: attention output RMS & absmax
      - mlp_rms  / mlp_max:  MLP output RMS & absmax

    Checks:
      - absmax > 30000  → P0 hard limit (bf16 saturation risk)
      - absmax > 10000  → P2 (elevated, observe)
      - RMS window drift > 30%  → P1 (trend shift)
      - RMS window drift > 15%  → P2 (observe)
    """
    alerts: list[Alert] = []
    cfg = {**ACTIVATION_THRESHOLDS, **(thresholds or {})}

    if not layer_data:
        return {}, alerts

    summary_rms: list[float] = []
    summary_max: list[float] = []
    worst_max = (-1, 0.0, "")
    per_layer = {}

    for layer_idx in sorted(layer_data.keys()):
        stats = layer_data[layer_idx]

        latest = {}
        for key in _ACTIVATION_KEYS:
            vals = stats.get(key, [])
            latest[key] = vals[-1][1] if vals else 0.0

        for max_key, label in (("attn_max", "attn"), ("mlp_max", "mlp")):
            absmax = latest[max_key]
            if absmax > cfg["absmax_p0"]:
                alerts.append(Alert("P0", "activation",
                    f"layer {layer_idx} {label}_max={absmax:.1f} "
                    f"接近 bf16 饱和 (>{cfg['absmax_p0']:.0f})",
                    value=absmax, threshold=cfg["absmax_p0"]))
            elif absmax > 10000:
                alerts.append(Alert("P2", "activation",
                    f"layer {layer_idx} {label}_max={absmax:.1f} 偏高",
                    value=absmax, threshold=10000))
            if absmax > worst_max[1]:
                worst_max = (layer_idx, absmax, label)

        for rms_key in ("attn_rms", "mlp_rms"):
            vals = stats.get(rms_key, [])
            w = min(100, len(vals) // 3)
            if w < 10:
                continue
            recent = [v[1] for v in vals[-w:]]
            earlier = ([v[1] for v in vals[-w * 2:-w]]
                       if len(vals) >= w * 2
                       else [v[1] for v in vals[:w]])
            r_mean = sum(recent) / len(recent)
            e_mean = sum(earlier) / len(earlier)
            if e_mean > 0:
                drift = (r_mean - e_mean) / e_mean
                if abs(drift) > cfg["rms_drift_p1"]:
                    alerts.append(Alert("P1", "activation",
                        f"layer {layer_idx} {rms_key} 漂移 {drift:+.1%} "
                        f"({e_mean:.3f}\u2192{r_mean:.3f})",
                        value=abs(drift), threshold=cfg["rms_drift_p1"]))
                elif abs(drift) > cfg["rms_drift_p2"]:
                    alerts.append(Alert("P2", "activation",
                        f"layer {layer_idx} {rms_key} 漂移 {drift:+.1%} "
                        f"({e_mean:.3f}\u2192{r_mean:.3f})",
                        value=abs(drift), threshold=cfg["rms_drift_p2"]))

        for rms_key in ("attn_rms", "mlp_rms"):
            if latest[rms_key] > 0:
                summary_rms.append(latest[rms_key])
        for max_key in ("attn_max", "mlp_max"):
            if latest[max_key] > 0:
                summary_max.append(latest[max_key])
        per_layer[layer_idx] = latest

    activation_metrics = {"layers": per_layer}
    if summary_rms or summary_max:
        latest_step = 0
        for stats in layer_data.values():
            for vals in stats.values():
                if vals:
                    latest_step = max(latest_step, vals[-1][0])
        activation_metrics["summary"] = {
            "step": latest_step,
            "rms_min": min(summary_rms) if summary_rms else 0,
            "rms_max": max(summary_rms) if summary_rms else 0,
            "rms_mean": (sum(summary_rms) / len(summary_rms)
                         if summary_rms else 0),
            "absmax_worst": worst_max[1],
            "absmax_worst_layer": worst_max[0],
            "absmax_worst_label": worst_max[2],
            "num_layers": len(layer_data),
        }

    alerts.sort(key=lambda a: a.severity)
    return activation_metrics, alerts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_eta(current_step: int, total_iters: int,
                iter_p50_ms: float = 0, start_step: int = 0) -> str:
    """Estimate remaining time.

    Primary: use iter_p50_ms (median iteration time) for direct calculation.
    Fallback: use elapsed wall-clock time corrected by start_step (checkpoint resume).
    """
    remaining_steps = total_iters - current_step
    if remaining_steps <= 0 or current_step <= 0:
        return "N/A"

    if iter_p50_ms > 0:
        remaining = remaining_steps * iter_p50_ms / 1000.0
    else:
        return "N/A"

    eta_dt = datetime.now() + timedelta(seconds=remaining)
    return f"{remaining / 86400:.1f} 天 (预计 {eta_dt.strftime('%m-%d %H:%M')})"


def format_report(job: dict, metrics: dict, alerts: list[Alert],
                  total_iters: int, source: str,
                  activation_metrics: Optional[dict] = None,
                  grafana_summary: Optional[dict] = None,
                  task_trends: Optional[dict] = None) -> str:
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_id = job.get("id", "?")
    status = job.get("status", "?")
    cluster = job.get("cluster", "?")

    lm = metrics.get("lm loss", {})
    step = lm.get("step", 0)
    if step == 0:
        for m in metrics.values():
            if isinstance(m, dict) and m.get("step", 0) > 0:
                step = m["step"]
                break
    progress = step / total_iters * 100 if total_iters > 0 else 0
    its = metrics.get("iter_stats")
    iter_p50 = its["p50"] if its else 0
    eta = compute_eta(step, total_iters, iter_p50_ms=iter_p50)

    has_p0 = any(a.level == "P0" for a in alerts)
    has_p1 = any(a.level == "P1" for a in alerts)
    has_p2 = any(a.level == "P2" for a in alerts)
    icon = "🔴" if has_p0 else "🟡" if has_p1 else "🔵" if has_p2 else "🟢"

    lines.append(f"{icon} Job {job_id} 监控报告 [{now}]  (数据源: {source})")
    lines.append("=" * 60)
    lines.append(f"  状态: {status} | 集群: {cluster}")
    lines.append(f"  进度: step {step:,} / {total_iters:,} ({progress:.2f}%)")
    lines.append(f"  预计剩余: {eta}")
    lines.append("")

    # Core metrics
    lines.append("--- 核心指标 ---")
    display = [
        ("lm loss", ".6f"), ("mtp_1 loss", ".6f"), ("delta(mtp-lm)", ".4f"),
        ("grad-norm", ".6f"), ("learning-rate", ".6e"),
        ("loss-scale", ".1f"), ("iter-time-ms", ".1f"),
    ]
    for tag, fmt in display:
        m = metrics.get(tag)
        if m:
            lines.append(f"  {tag:20s}: {m['value']:{fmt}}  (step {m['step']})")

    # Extended stats
    gs = metrics.get("grad_stats")
    if gs:
        lines.append(f"  {'grad-norm 统计':20s}: "
                     f"mean={gs['mean']:.4f}  max={gs['max']:.3f}  "
                     f">1.0: {gs['above_1']}/{gs['count']}")
    if its:
        pct = its["slow_count"] / its["count"] if its["count"] else 0
        lines.append(f"  {'iter-time 统计':20s}: "
                     f"P50={its['p50']:.0f}ms  P99={its['p99']:.0f}ms  "
                     f">2s: {its['slow_count']}/{its['count']} ({pct:.1%})")
    lines.append("")

    # Activation stats
    act_status = activation_metrics.get("_status") if activation_metrics else None
    if activation_metrics and activation_metrics.get("summary"):
        s = activation_metrics["summary"]
        lines.append("--- 激活值统计 (TensorBoard) ---")
        act_layers = sorted(activation_metrics.get("layers", {}).keys())
        if act_layers:
            layer_label = ", ".join(f"L{i}" for i in act_layers)
            data_desc = f"{s['num_layers']} 哨兵层 ({layer_label})"
        else:
            data_desc = f"{s['num_layers']} 层"
        lines.append(f"  数据: {data_desc}, step {s.get('step', '?')}")
        lines.append(f"  {'RMS 范围':20s}: "
                     f"min={s['rms_min']:.3f}  max={s['rms_max']:.3f}  "
                     f"mean={s['rms_mean']:.3f}")
        if s.get("absmax_worst", 0) > 0:
            lines.append(f"  {'最大绝对值':20s}: "
                         f"{s['absmax_worst']:.1f} "
                         f"({s['absmax_worst_label']}, "
                         f"layer {s['absmax_worst_layer']})")
        layers = activation_metrics.get("layers", {})
        if layers:
            top_mlp = sorted(layers.items(),
                             key=lambda x: x[1].get("mlp_rms", 0),
                             reverse=True)[:3]
            parts = [f"L{li}={lv.get('mlp_rms', 0):.3f}"
                     for li, lv in top_mlp]
            lines.append(f"  {'Top MLP RMS':20s}: {', '.join(parts)}")
            top_attn = sorted(layers.items(),
                              key=lambda x: x[1].get("attn_rms", 0),
                              reverse=True)[:3]
            parts = [f"L{li}={lv.get('attn_rms', 0):.3f}"
                     for li, lv in top_attn]
            lines.append(f"  {'Top Attn RMS':20s}: {', '.join(parts)}")
        lines.append("")
    elif act_status == "unavailable":
        reason = activation_metrics.get("_reason", "tfevents 下载失败")
        lines.append("--- 激活值统计 (TensorBoard) ---")
        lines.append(f"  ⏳ 数据暂不可用: {reason}")
        lines.append("")
    elif act_status == "skipped":
        reason = activation_metrics.get("_reason", "")
        lines.append("--- 激活值统计 (TensorBoard) ---")
        lines.append(f"  ⏸ 已跳过: {reason}")
        lines.append("")

    # Grafana system metrics
    if grafana_summary:
        lines.append("--- 系统监控 (Grafana) ---")
        hdr = f"  {'指标':<16} {'均值':>8} {'CV':>8} {'最小':>8} {'最大':>8} {'趋势':>8}"
        lines.append(hdr)
        for name, s in grafana_summary.items():
            trend_str = {"up": "↑爬升", "down": "↓下降"}.get(
                s["trend"], "—")
            if s["trend"] != "stable":
                trend_str += f" {s['trend_pct']:+.0%}"
            lines.append(
                f"  {name:<16} {s['mean']:>8.1f} {s['cv']:>8.3f} "
                f"{s['min']:>8.1f} {s['max']:>8.1f} {trend_str:>8}")
        lines.append("")

    # Per-dataset task loss trends
    if task_trends:
        sr = task_trends["step_range"]
        lines.append(f"--- 分数据集 loss 趋势 (step {sr[0]:,} → {sr[1]:,}) ---")
        lines.append(f"  共 {task_trends['total_datasets']} 个数据集")
        if task_trends["risers"]:
            lines.append("  ↑ 上升 top:")
            for t in task_trends["risers"]:
                lines.append(
                    f"    {t['dataset']:<30s} "
                    f"{t['mean_earlier']:.4f} → {t['mean_recent']:.4f}"
                    f"  ({t['pct_change']:+.2f}%)")
        if task_trends["fallers"]:
            lines.append("  ↓ 下降 top:")
            for t in task_trends["fallers"]:
                lines.append(
                    f"    {t['dataset']:<30s} "
                    f"{t['mean_earlier']:.4f} → {t['mean_recent']:.4f}"
                    f"  ({t['pct_change']:+.2f}%)")
        if not task_trends["risers"] and not task_trends["fallers"]:
            lines.append("  所有数据集 loss 无变化")
        lines.append("")

    # Alerts by tier
    if alerts:
        lines.append("--- 告警 ---")
        tier_icons = {"P0": "🚨", "P1": "⚠️", "P2": "ℹ️"}
        for a in sorted(alerts, key=lambda x: x.severity):
            ic = tier_icons.get(a.level, "❓")
            lines.append(f"  {ic} [{a.level}] {a.metric}: {a.message}")
        lines.append("")
    else:
        lines.append("--- 状态正常，无告警 ---")
        lines.append("")

    return "\n".join(lines)


def send_feishu_webhook(webhook_url: str, content: str, at_all: bool = False):
    if at_all:
        content = '<at user_id="all">所有人</at>\n' + content
    payload = json.dumps({"msg_type": "text", "content": {"text": content}}).encode()
    req = urllib.request.Request(
        webhook_url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=10) as resp:
            resp.read()
    except Exception as e:
        print(f"  飞书 webhook 发送失败: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_check(args, token: str, state: MonitorState) -> bool:
    job = get_job_info(args.job_id, token)
    if not job:
        print(f"ERROR: 未找到 Job {args.job_id}", file=sys.stderr)
        return True

    all_alerts = check_job_status(job)
    metrics = {}
    source = "none"
    entries = None

    # Build threshold overrides from CLI args
    overrides = {}
    if args.grad_norm_max != THRESHOLDS["grad_norm_p1"]:
        overrides["grad_norm_p1"] = args.grad_norm_max
    if args.loss_diverge != THRESHOLDS["loss_diverge"]:
        overrides["loss_diverge"] = args.loss_diverge

    if job.get("status") == "Running":
        max_fetch_retries = 3
        fetch_delay = 5

        for attempt in range(max_fetch_retries):
            if args.source in ("logs", "auto"):
                entries = fetch_metrics_from_logs(job, token, state)
                if entries:
                    state.cached_entries = entries
                    state.cached_at = time.time()
                    source = f"pod-logs ({len(entries)} iters)"
                    metrics, metric_alerts = analyze_metrics(
                        entries, state, overrides or None)
                    all_alerts.extend(metric_alerts)
                    break

            if args.source in ("tb", "auto"):
                tb_metrics = fetch_metrics_from_tb(job, token)
                if tb_metrics:
                    tag_name_map = {
                        "lm_loss": "lm loss", "mtp_loss": "mtp_1 loss",
                        "grad_norm": "grad-norm", "lr": "learning-rate",
                        "loss_scale": "loss-scale", "batch_size": "batch-size",
                    }
                    for key, val in tb_metrics.items():
                        tag_name = tag_name_map.get(key, key)
                        metrics[tag_name] = {"step": val["step"],
                                             "value": val["value"]}
                    if "lm_loss" not in tb_metrics:
                        fetched = ", ".join(tb_metrics.keys())
                        print(f"  [warn] TB 数据不完整 (缺少 lm_loss), "
                              f"仅获取: {fetched}", file=sys.stderr)
                        if attempt < max_fetch_retries - 1:
                            continue
                    source = "tensorboard-api"
                    break

            if attempt < max_fetch_retries - 1:
                print(f"  [info] 数据获取失败，{fetch_delay}秒后重试 "
                      f"({attempt + 1}/{max_fetch_retries})...", file=sys.stderr)
                time.sleep(fetch_delay)

        if not metrics and state.cached_entries:
            age = time.time() - state.cached_at
            entries = state.cached_entries
            source = f"缓存 ({len(entries)} iters, {age / 60:.0f}分钟前)"
            metrics, metric_alerts = analyze_metrics(
                entries, state, overrides or None)
            all_alerts.extend(metric_alerts)
            all_alerts.append(Alert("P2", "stale_data",
                f"使用缓存数据 (API 暂时不可用, 缓存于 {age / 60:.0f} 分钟前)"))

        if not metrics:
            all_alerts.append(Alert("P1", "no_data",
                f"数据获取失败 (已重试 {max_fetch_retries} 次)"))
        elif metrics and not metrics.get("lm loss"):
            missing = [k for k in ("lm loss", "mtp_1 loss", "loss-scale")
                       if k not in metrics]
            all_alerts.append(Alert("P2", "partial_data",
                f"数据不完整, 缺少: {', '.join(missing)}"))

    # Activation stats (TB scalar API or local --tb-dir)
    activation_metrics = {}
    if not args.no_activation:
        tb_dir = args.tb_dir if args.tb_dir else None
        layer_data = fetch_activation_stats(
            job, token, state, tb_dir=tb_dir,
            activation_interval=args.activation_interval,
            num_layers=args.num_layers)
        if layer_data.get("_status"):
            activation_metrics = layer_data
        elif layer_data:
            activation_metrics, act_alerts = analyze_activation(layer_data)
            all_alerts.extend(act_alerts)
        else:
            activation_metrics = {
                "_status": "unavailable",
                "_reason": "无激活值数据 (TB API 无 tag, tfevents 为空或未下载)"
            }

    # Grafana system metrics (GPU%, Temp, Mem, etc.)
    grafana_summary = None
    if not args.no_grafana:
        series = fetch_grafana_series(job, lookback_sec=args.grafana_window)
        if series:
            grafana_summary, grafana_alerts = analyze_grafana(series)
            all_alerts.extend(grafana_alerts)
        else:
            cluster = job.get("cluster", "")
            if cluster not in GRAFANA_CONFIG:
                print(f"  [info] Grafana 未配置 (集群: {cluster})",
                      file=sys.stderr)

    # Per-dataset task loss trends (lm_loss/<dataset>)
    task_trends = None
    if not args.no_task_loss:
        task_data = fetch_task_losses(job, token)
        if task_data:
            task_trends = analyze_task_loss_trends(
                task_data, top_n=args.val_top_n)

    total = args.total_iters
    if not total and entries:
        total = entries[-1].get("total_iters", 0)
    if not total and metrics:
        total = 1068115

    report = format_report(job, metrics, all_alerts, total, source,
                           activation_metrics=activation_metrics,
                           grafana_summary=grafana_summary,
                           task_trends=task_trends)
    print(report)

    has_p0 = any(a.level == "P0" for a in all_alerts)
    if args.webhook and (has_p0 or args.notify_all):
        send_feishu_webhook(args.webhook, report, at_all=has_p0)

    return has_p0


def main():
    parser = argparse.ArgumentParser(
        description="Cybertron 预训练任务监控 (P0 熔断/P1 统计/P2 信息)")
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--interval", type=int, default=0,
                        help="轮询间隔(秒), 0=单次")
    parser.add_argument("--total-iters", type=int, default=0,
                        help="总步数 (0=自动检测)")
    parser.add_argument("--source", choices=["auto", "logs", "tb"], default="auto",
                        help="数据源: auto=日志优先, logs=仅日志, tb=仅TensorBoard")

    parser.add_argument("--grad-norm-max", type=float,
                        default=THRESHOLDS["grad_norm_p1"],
                        help=f"grad_norm P1 阈值 (默认 {THRESHOLDS['grad_norm_p1']})")
    parser.add_argument("--loss-diverge", type=float,
                        default=THRESHOLDS["loss_diverge"],
                        help=f"单步 loss P0 阈值 (默认 {THRESHOLDS['loss_diverge']})")

    # Activation monitoring
    parser.add_argument("--tb-dir", type=str, default="",
                        help="本地 TensorBoard event 目录 "
                             "(不指定则从 Cybertron 下载)")
    parser.add_argument("--activation-interval", type=int, default=1800,
                        help="激活值数据拉取间隔(秒), 默认 1800 (30分钟)")
    parser.add_argument("--no-activation", action="store_true",
                        help="跳过激活值监控")
    parser.add_argument("--num-layers", type=int, default=60,
                        help="模型层数 (用于哨兵层选择, 默认 60)")

    # Grafana system monitoring
    parser.add_argument("--no-grafana", action="store_true",
                        help="跳过 Grafana 系统监控 (GPU%, Temp, Mem 等)")
    parser.add_argument("--grafana-window", type=int, default=3600,
                        help="Grafana 查询回看窗口(秒), 默认 3600 (1小时)")

    # Per-dataset task loss monitoring
    parser.add_argument("--no-task-loss", action="store_true",
                        help="跳过分数据集 task loss 趋势监控")
    parser.add_argument("--val-top-n", type=int, default=5,
                        help="分数据集 loss 趋势展示 top N (默认 5)")

    parser.add_argument("--webhook", type=str, default="")
    parser.add_argument("--notify-all", action="store_true",
                        help="所有级别都发 webhook (默认仅 P0)")

    args = parser.parse_args()
    token = resolve_token(args)
    state = MonitorState()

    if args.interval <= 0:
        has_p0 = run_check(args, token, state)
        sys.exit(1 if has_p0 else 0)

    print(f"开始监控 Job {args.job_id}，间隔 {args.interval} 秒")
    print(f"按 Ctrl+C 停止\n")

    while True:
        try:
            run_check(args, token, state)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            time.sleep(min(args.interval, 60))


if __name__ == "__main__":
    main()
