'''
从已经配置好数据集配比的csv，得到最终longdecay配比

逻辑（参考 /user/xuxiaoyue/skills_sz/process-pretrain-data-weight/scripts/process_weight_longdecay.py
的 `build_data_split_info` / `process_split_info` 实现）：
  1. 读取 csv 文件
  2. 把"需要上采样"=是 的行，对分桶进行重新配比：
       - 32k decay (ld_len==32k) , divider_k=16 , 0~16k 桶=37.5% , 16k~ 桶=62.5%
       - 64k decay (ld_len==64k) , divider_k=32 , 0~32k 桶=50%   , 32k~ 桶=50%
       - 128k decay (ld_len==128k), divider_k=64 , 0~64k 桶=50%   , 64k~ 桶=50%
       - 256k / 512k 同理
     注意：长于 ld_len 的桶**不丢弃**，全部归入"长"组（与 canonical 行为一致）。
  3. 对于不规则结构（无分桶 / 仅 qa / 仅单桶），不做上采样拆分，权重落在原数据集 / 单桶上。
  4. （可选）读取 entry_count_json，把 multi_buckets 中条数 <= min_entry_count 的桶丢弃，
     剩余桶在该数据集的 short/long 配比内重新等分（与 gen_longdecay.py STEP 4 一致）。
  5. （可选）读取 reference_decay_sh：对**非上采样**数据集，从该参考文件读取每个 path 的
     原始权重，归一化得到该数据集内部的"自然配比"，再乘以 csv 设定的数据集级 portion，
     从而把 csv 配比应用到 bucket 粒度上。上采样数据集仍走第 2 步逻辑。
  6. 输出新的 csv 文件，打印需要上采样的行的占比。
'''
import os
import re
import csv
import json
import argparse
from collections import defaultdict

# 默认 entry count 阈值（与 gen_longdecay.py 中 MIN_ENTRY_COUNT 一致）
DEFAULT_MIN_ENTRY_COUNT = 1000
DEFAULT_ENTRY_COUNT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'minicpm5_entry_count.json'
)

# ------ CONFIGS ------
# 与 process_weight_longdecay.py 的 _LONG_SPLIT_CONFIGS 一一对应：
#   long_<ld_len>: divider_k = ld_len // 2 (k)
LONG_SPLIT_CONFIGS = {
    "32k":  {"divider_k": 16,  "short_ratio": 0.375, "long_ratio": 0.625},
    "64k":  {"divider_k": 32,  "short_ratio": 0.5,   "long_ratio": 0.5},
    "128k": {"divider_k": 64,  "short_ratio": 0.5,   "long_ratio": 0.5},
    "256k": {"divider_k": 128, "short_ratio": 0.5,   "long_ratio": 0.5},
    "512k": {"divider_k": 256, "short_ratio": 0.5,   "long_ratio": 0.5},
}

# 长度分桶名称（保持顺序：从短到长），与 process_weight_longdecay.py 中的 base_split_folder_list 一致
BUCKET_FOLDER_LIST = [
    "0k_4k", "4k_8k", "8k_16k", "16k_32k", "32k_64k",
    "64k_128k", "128k_256k", "256k_512k", "512k_gt",
]
OTHER_FOLDER_LIST = ["qa"]

# 数据集物理位置的可能 root（顺序敏感：后者优先）
DEFAULT_MAYBE_ROOT_PATHS = [
    "/user/zhangyixuan/data/minicpm5/sst_cpm5_0117_merge",
    "/user/zhangyixuan/data/minicpm5/sst_cpm5_0321_merge",
]

# csv 字段名
PATH_FIELD     = 'MiniCPM5-128k cybertron 路径（merge-link）'
UPSAMPLE_FIELD = '需要上采样'
TEXT_FIELD     = '文本'
TYPE_FIELD     = '类型'
CLASS_FIELD    = '类别'

# ------ HELPERS --------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_file", type=str, required=True)
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--portion_field", type=int, required=True,
                   help="csv 中 portion 列的索引（从 0 开始）。例：longdecay 加权配比 = 29")
    p.add_argument("--ld_len", choices=list(LONG_SPLIT_CONFIGS.keys()), required=True,
                   help="longdecay 长度（决定 divider_k 与 short/long ratio）")
    p.add_argument("--maybe-root-path", action="append", default=[], metavar="DIR",
                   help="可多次指定：用于解析数据名的 SST merge 根目录（默认使用内置两处 merge 根）")
    p.add_argument("--entry-count-json", type=str, default=DEFAULT_ENTRY_COUNT_JSON,
                   help=f"每数据集每分桶的条数 JSONL 文件，用于丢弃 <= --min-entry-count 的桶。"
                        f"找不到此文件则跳过过滤。默认: {DEFAULT_ENTRY_COUNT_JSON}")
    p.add_argument("--min-entry-count", type=int, default=DEFAULT_MIN_ENTRY_COUNT,
                   help=f"multi_buckets 中条数 <= 此值的桶将被丢弃，仅在 --entry-count-json 可读时生效。"
                        f"传 0 关闭。默认: {DEFAULT_MIN_ENTRY_COUNT}")
    p.add_argument("--reference-decay-sh", type=str, default=None,
                   help="参考的 decay.sh（含 DATA_PATH=\"...\" 块），用作非上采样数据集的"
                        "桶内自然配比来源；该文件提供每条 path 的原始权重，会按数据集名归一化"
                        "后乘以 csv 设定的数据集级 portion。不提供则非上采样数据集仍按 csv path 整体保留。")
    return p.parse_args()


def _bucket_lower_k(bucket_name):
    """'16k_32k' -> 16 ; '512k_gt' -> 512  （与 canonical 同名函数一致）。"""
    return int(bucket_name.split("_")[0].rstrip("k"))


def filter_low_entry_buckets(entries, bucket_counts, min_entry_count):
    """对 [(weight, path), ...] 应用 1k 过滤。

    判定规则：path 末段名若属于 BUCKET_FOLDER_LIST，则查 bucket_counts，
    count <= min_entry_count 时丢弃；非桶名后缀（qa / wrapper / 直接 dataset 路径等）
    一律保留（无法匹配 bucket_counts，无法判定）。

    返回 (kept, dropped)：kept 与原顺序一致；dropped = [(bucket_name, count, weight, path), ...]
    """
    if not bucket_counts or min_entry_count <= 0:
        return list(entries), []
    kept = []
    dropped = []
    for w, p in entries:
        last = os.path.basename(p.rstrip('/'))
        if last in BUCKET_FOLDER_LIST:
            ec = bucket_counts.get(last, 0)
            if ec <= min_entry_count:
                dropped.append((last, ec, w, p))
                continue
        kept.append((w, p))
    return kept, dropped


def load_entry_counts(path):
    """读取 minicpm5_entry_count.json (JSONL)。

    每行格式: ["<dataset_name>", <total_count>, {"<bucket_name>": <count>, ...}]
    返回: {dataset_name: {bucket_name: count}}
    若文件不存在则返回 None（调用方据此跳过过滤）。
    """
    if not path or not os.path.isfile(path):
        return None
    counts = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (isinstance(rec, list) and len(rec) >= 3):
                continue
            ds_name, _total, bkt_map = rec[0], rec[1], rec[2]
            if isinstance(ds_name, str) and isinstance(bkt_map, dict):
                counts[ds_name] = {b: int(c) for b, c in bkt_map.items()}
    return counts


def parse_decay_reference(path):
    """读取 decay.sh 类型文件，提取 DATA_PATH="..." 块内 (weight, path) 二元组。

    返回 [(weight, path), ...]，原样保留 weight==0 的条目（调用方决定是否过滤）。
    """
    with open(path, 'r') as f:
        text = f.read()
    m = re.search(r'DATA_PATH="(.*?)"', text, re.DOTALL)
    if not m:
        raise ValueError(f"在 {path} 中未找到 DATA_PATH=\"...\" 块")
    block = m.group(1)
    entries = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            w = float(parts[0])
        except ValueError:
            continue
        entries.append((w, parts[1].strip()))
    return entries


def group_reference_by_dataset(ref_entries, name_map, maybe_root_paths):
    """把 (weight, path) 列表按"数据集名"分组。

    数据集名识别策略（按优先级）：
      1) 若某个 name_map 中的物理路径是 ref_path 的前缀，则取对应 data_name
      2) 若 ref_path 落在某个已知 merge root 下，则取 merge root 之后的第一段
      3) 否则取 basename；若 basename 看起来是分桶名 / qa，则上溯一层

    返回: {data_name: [(weight, ref_path), ...]}（保持原始顺序）
    """
    phys_to_name = {}
    for name, paths in name_map.items():
        for p in paths:
            phys_to_name[p.rstrip('/')] = name
    sorted_phys = sorted(phys_to_name.keys(), key=len, reverse=True)
    norm_roots = [r.rstrip('/') for r in maybe_root_paths]

    grouped = defaultdict(list)
    path_to_dataname = {}
    for w, p in ref_entries:
        ds = None
        for phys in sorted_phys:
            if p == phys or p.startswith(phys + '/'):
                ds = phys_to_name[phys]
                break
        if ds is None:
            for root in norm_roots:
                if p.startswith(root + '/'):
                    rest = p[len(root) + 1:]
                    ds = rest.split('/', 1)[0]
                    break
        if ds is None:
            base = os.path.basename(p.rstrip('/'))
            if base in BUCKET_FOLDER_LIST or base in OTHER_FOLDER_LIST:
                base = os.path.basename(os.path.dirname(p.rstrip('/')))
            ds = base
        grouped[ds].append((w, p))
        path_to_dataname[p] = ds
    return grouped, path_to_dataname


# --- 与 process_weight_longdecay.py 的 build_data_split_info(dry_run=True) 等价的 layout 探测 ---
def build_data_split_info(orig_path, split_folder_list=BUCKET_FOLDER_LIST,
                          other_split_folder_list=OTHER_FOLDER_LIST):
    """扫描 orig_path 下的目录结构，仅输出 layout（不统计 size / row 数）。

    与 process_weight_longdecay.py 的 `build_data_split_info(dry_run=True)` 行为一致：

      情形 1   配置路径下直接是文件（无分桶/qa） → sub_folder=None, sub_sub_folder=None
      情形 2   存在单层或多层包装子目录（名不在 allowed 中且只有一个孩子） → 连续剥开记入 sub_folder
        2.1  剥完仍为直接文件目录          → sub_sub_folder=None
        2.2  剥完仅有 qa                  → sub_sub_folder=["qa"]
        2.3  剥完为长度分桶目录            → sub_sub_folder=按 split_folder_list 顺序排列的桶名

    返回:
        {
          'root_path':      orig_path 的绝对路径,
          'sub_folder':     List[str] | None,
          'sub_sub_folder': List[str] | None,
        }
    """
    orig_path = os.path.abspath(orig_path)
    allowed = set(split_folder_list) | set(other_split_folder_list)
    sub_folder_chain = []
    path = orig_path

    while True:
        names = [x for x in os.listdir(path) if not x.startswith(".")]
        if len(names) == 1:
            one = names[0]
            one_path = os.path.join(path, one)
            if one not in allowed and os.path.isdir(one_path):
                sub_folder_chain.append(one)
                path = one_path
                continue
        break

    names = [x for x in os.listdir(path) if not x.startswith(".")]
    if not names:
        raise RuntimeError(f"空目录: {path}")

    if any(not os.path.isdir(os.path.join(path, n)) for n in names):
        return {
            'root_path':      orig_path,
            'sub_folder':     sub_folder_chain or None,
            'sub_sub_folder': None,
        }

    for n in names:
        if n not in allowed:
            raise NotImplementedError(
                f"{orig_path} -> {path} 下存在未识别子目录 {n!r}，应为分桶名或 qa"
            )

    present_splits = {n for n in names if n in split_folder_list}
    split_names = [bn for bn in split_folder_list if bn in present_splits]
    qa_names = [n for n in names if n in other_split_folder_list]

    if split_names and qa_names:
        raise NotImplementedError(f"{path} 下同时存在 qa 与长度分桶，暂不支持")
    if qa_names:
        if qa_names != ["qa"]:
            raise NotImplementedError(f"{path} 下仅支持单一 qa 子目录，当前: {qa_names}")
        return {
            'root_path':      orig_path,
            'sub_folder':     sub_folder_chain or None,
            'sub_sub_folder': ["qa"],
        }

    return {
        'root_path':      orig_path,
        'sub_folder':     sub_folder_chain or None,
        'sub_sub_folder': split_names,
    }


def build_root_path_map(maybe_root_paths):
    """root 列表 -> {data_name: [physical_path,...]}。

    与 process_weight_longdecay.py main() 中相同：每个 root 下的一级子目录视为一个 data_name；
    同名时按 maybe_root_paths 出现顺序依次 append（生产中用 [-1] 取后者优先）。
    """
    name_map = {}
    for root in maybe_root_paths:
        if not os.path.isdir(root):
            continue
        for n in sorted(os.listdir(root)):
            sub = os.path.join(root, n)
            if os.path.isdir(sub):
                name_map.setdefault(n, []).append(sub)
    return name_map


def resolve_dataset_dir(csv_path, name_map):
    """根据 csv 中的 path 列得到物理 dataset 目录。

    用 basename 在 name_map 中查找。多个候选时按以下优先级挑选：
      1) **multi_buckets** layout（有标准桶子目录）优先
      2) 任何能成功探测 layout 的（direct_files / qa_only / single_bucket）
      3) 都探测失败，回退到 name_map[data_name][-1]（保留原 SDK [-1] 语义）

    这样可以避免误选到 dedup_link_all 这种含非标准 part 子目录的 root，
    从而让 multi_buckets 数据集被正确识别并走桶级上采样逻辑。

    若 name_map 未命中则把 csv_path 本身视作 direct path 返回。
    """
    data_name = os.path.basename(csv_path.rstrip("/"))
    if data_name in name_map and name_map[data_name]:
        candidates = name_map[data_name]
        if len(candidates) == 1:
            return candidates[0], 'merge'
        any_layout = None
        for cand in candidates:
            try:
                info = build_data_split_info(cand)
            except (NotImplementedError, RuntimeError, OSError):
                continue
            ssf = info.get('sub_sub_folder')
            if ssf and len(ssf) > 1:
                return cand, 'merge'
            if any_layout is None:
                any_layout = cand
        if any_layout is not None:
            return any_layout, 'merge'
        return candidates[-1], 'merge'
    if os.path.isdir(csv_path):
        return csv_path, 'direct'
    raise FileNotFoundError(
        f"{csv_path!r} 既不在 maybe_root_paths 中（data_name={data_name!r}），"
        f"也不是已存在的目录"
    )


# --- 与 process_split_info 中 long_* 分支等价的桶级权重计算 ---
def compute_split_weights(split_info, total_weight, ld_len,
                          bucket_counts=None, min_entry_count=0):
    """根据 layout 与 long_<ld_len> 配置计算各桶（或单一路径）的权重。

    返回 (weight_paths, dropped_buckets, diverted_by_bucket)：
      weight_paths        -- [(weight, path), ...]，**保留桶**对应的权重
      dropped_buckets     -- [(bucket_name, count), ...]，因条数过低被丢弃的桶
      diverted_by_bucket  -- 总是 {}（兼容历史接口；新逻辑下被丢弃桶的份额已在桶内重分配）

    分桶策略（multi_buckets, len(sub_sub_folder) > 1）：

      1) **先**应用 1k 过滤（min_entry_count）拿到 kept 桶集合
      2) n_short / n_long **只统计 kept 桶**：
         16k- 的 kept 桶平分 short_ratio，16k+ 的 kept 桶平分 long_ratio
      3) 若某一侧 kept 数为 0，该侧 ratio 让给另一侧（eff_short=0 / eff_long=1 反之亦然）
      4) 被丢弃桶的份额自动留在该侧的 kept 桶里（不再 divert 到 merge_small/decay）

      - sub_sub_folder is None        : 整体落在 root_path（或 root_path/sub_folder[0]）
      - len(sub_sub_folder) == 1      : 整体落在 root_path/sub_sub_folder[0]

    1k 过滤仅作用于 multi_buckets layout。direct_files / qa_only / single_bucket
    由于无桶名信息无法在 entry_count_json 中匹配，按原逻辑整体保留 total_weight。
    """
    cfg = LONG_SPLIT_CONFIGS[ld_len]
    div_k = cfg["divider_k"]

    root_path = split_info["root_path"]
    sub_folder = split_info["sub_folder"]
    sub_sub_folder = split_info["sub_sub_folder"]

    if sub_folder is not None:
        assert len(sub_folder) == 1, (
            f"len(sub_folder)={len(sub_folder)} 不等于 1（root_path={root_path}, "
            f"sub_folder={sub_folder}）"
        )
        root_path = os.path.join(root_path, sub_folder[0])

    if sub_sub_folder is None:
        return [(total_weight, root_path)], [], {}

    if len(sub_sub_folder) == 1:
        return [(total_weight, os.path.join(root_path, sub_sub_folder[0]))], [], {}

    use_filter = (bucket_counts is not None and min_entry_count > 0)
    kept = []
    dropped = []
    for b in sub_sub_folder:
        if use_filter:
            ec = bucket_counts.get(b, 0)
            if ec <= min_entry_count:
                dropped.append((b, ec))
                continue
        kept.append(b)

    n_short = sum(1 for b in kept if _bucket_lower_k(b) < div_k)
    n_long = len(kept) - n_short

    if n_short == 0 and n_long == 0:
        return [], dropped, {}

    if n_short == 0:
        eff_short, eff_long = 0.0, 1.0
    elif n_long == 0:
        eff_short, eff_long = 1.0, 0.0
    else:
        eff_short = cfg["short_ratio"]
        eff_long = cfg["long_ratio"]

    out = []
    for b in kept:
        if _bucket_lower_k(b) < div_k:
            share = total_weight * eff_short / n_short if n_short > 0 else 0.0
        else:
            share = total_weight * eff_long / n_long if n_long > 0 else 0.0
        out.append((share, os.path.join(root_path, b)))
    return out, dropped, {}


def load_csv(file_path, portion_field_idx):
    """读取 csv 文件，返回 list of dict, 每行包含:
        text     -- 文本（数据集名）
        type     -- 类型（pretrain / sft / ...）
        cls      -- 类别（中文 PDF / 英文论文 / ...）
        path     -- PATH_FIELD 列的值（cybertron merge-link 路径）
        portion  -- portion_field_idx 列的值 (float)
        upsample -- 是否需要上采样 (bool, "是" => True)
    """
    csv_data = []
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        return csv_data

    header = [h.lstrip('\ufeff').strip() for h in rows[0]]
    field_idx = {name: i for i, name in enumerate(header)}

    for h in (PATH_FIELD, UPSAMPLE_FIELD, TEXT_FIELD, TYPE_FIELD, CLASS_FIELD):
        if h not in field_idx:
            raise ValueError(f"csv 缺少必需的列 {h!r}\n实际列名: {header}")

    portion_col_name = header[portion_field_idx] if 0 <= portion_field_idx < len(header) else f"#{portion_field_idx}"
    print(f"  portion_field={portion_field_idx} -> 列名 {portion_col_name!r}")

    def _get(row, idx):
        return row[idx].strip() if idx < len(row) else ''

    for row in rows[1:]:
        if not row or all(not (c or '').strip() for c in row):
            continue
        try:
            portion = float(row[portion_field_idx])
        except (ValueError, IndexError):
            portion = 0.0
        csv_data.append({
            'text':     _get(row, field_idx[TEXT_FIELD]),
            'type':     _get(row, field_idx[TYPE_FIELD]),
            'cls':      _get(row, field_idx[CLASS_FIELD]),
            'path':     _get(row, field_idx[PATH_FIELD]),
            'portion':  portion,
            'upsample': _get(row, field_idx[UPSAMPLE_FIELD]) == '是',
        })
    return csv_data


# ------ MAIN --------
def gen_minicpm5_longdecay(args):
    print(f"读取 csv: {args.csv_file}")
    print(f"longdecay 长度: {args.ld_len}")

    cfg = LONG_SPLIT_CONFIGS[args.ld_len]
    print(f"  divider_k={cfg['divider_k']}k, "
          f"short_ratio={cfg['short_ratio']}, long_ratio={cfg['long_ratio']}")

    maybe_root_paths = list(args.maybe_root_path) if args.maybe_root_path else list(DEFAULT_MAYBE_ROOT_PATHS)
    print(f"  maybe_root_paths: {maybe_root_paths}")
    name_map = build_root_path_map(maybe_root_paths)
    print(f"  在 maybe_root_paths 中发现 {len(name_map)} 个数据集名")

    entry_counts = load_entry_counts(args.entry_count_json)
    if entry_counts is None:
        print(f"  [警告] entry_count_json 不存在或不可读: {args.entry_count_json}")
        print(f"          跳过 1k 过滤（multi_buckets 中所有桶都保留）")
        effective_min = 0
    elif args.min_entry_count <= 0:
        print(f"  --min-entry-count={args.min_entry_count} <= 0，关闭 1k 过滤")
        effective_min = 0
    else:
        print(f"  entry_count_json: {args.entry_count_json} ({len(entry_counts)} 数据集)")
        print(f"  min_entry_count: {args.min_entry_count}（multi_buckets 中条数 <= {args.min_entry_count} 的桶将被丢弃）")
        effective_min = args.min_entry_count

    reference_grouped = None
    ref_entries = []
    ref_path_order = {}
    ref_path_to_dataname = {}
    if args.reference_decay_sh:
        ref_entries = parse_decay_reference(args.reference_decay_sh)
        n_pos = sum(1 for w, _ in ref_entries if w > 0)
        print(f"  reference_decay_sh: {args.reference_decay_sh}")
        print(f"    总条目 {len(ref_entries)} 行，weight>0 {n_pos} 行（保留全部，weight=0 视为占位 path）")
        reference_grouped, ref_path_to_dataname = group_reference_by_dataset(
            ref_entries, name_map, maybe_root_paths)
        print(f"    按 data_name 分组得到 {len(reference_grouped)} 个数据集")
        ref_path_order = {p: i for i, (_, p) in enumerate(ref_entries)}

    datasets = load_csv(args.csv_file, args.portion_field)
    print(f"  共加载 {len(datasets)} 行")

    csv_ds_by_dataname = {}
    for d in datasets:
        if d['type'] == '舍弃':
            continue
        dn = os.path.basename(d['path'].rstrip('/'))
        csv_ds_by_dataname.setdefault(dn, d)

    output_portion_paths = []
    path_to_type = {}
    path_to_cls = {}
    upsample_records = []
    layout_summary = defaultdict(int)
    skipped_zero = 0
    skipped_discarded = []
    skipped_resolve = 0
    skipped_layout = 0
    dropped_low_entry_total = 0
    dropped_low_entry_details = []
    dropped_full_dataset = []
    no_entry_info_datasets = []
    nonups_used_ref = 0
    nonups_fallback = []
    diversions_to_merge_small = defaultdict(float)

    def _emit(weight, path, source_ds):
        """Append to output_portion_paths and record source type/cls for breakdown."""
        output_portion_paths.append((weight, path))
        if source_ds is not None:
            path_to_type.setdefault(path, source_ds['type'])
            path_to_cls.setdefault(path, source_ds['cls'])

    for ds in datasets:
        if ds['type'] == '舍弃':
            skipped_discarded.append((ds['text'], ds['cls'], ds['portion']))
            continue

        is_zero = ds['portion'] <= 0
        if is_zero:
            skipped_zero += 1
        # portion=0 也走非上采样逻辑（reference path or csv path），输出权重 0 占位
        if (not ds['upsample']) or is_zero:
            if reference_grouped is not None:
                data_name = os.path.basename(ds['path'].rstrip('/'))
                ref_for_ds = reference_grouped.get(data_name)
                if ref_for_ds:
                    bcs = (entry_counts.get(data_name)
                           if (effective_min > 0 and entry_counts is not None) else None)
                    has_bucket_path = any(
                        os.path.basename(p.rstrip('/')) in BUCKET_FOLDER_LIST
                        for _, p in ref_for_ds
                    )
                    if effective_min > 0 and has_bucket_path and bcs is None:
                        no_entry_info_datasets.append((ds['text'], data_name))

                    # 非上采样语义：
                    #   ref 提供桶之间的"自然分布比例"，CSV 提供数据集总配比。
                    #   每个桶的最终权重 = ds.portion * (ref_w_bkt / ref_sum)
                    #   ref_sum 为 0 时退化为按桶数等分。
                    ref_sum = sum(w for w, _ in ref_for_ds)
                    n_ref = len(ref_for_ds)
                    if is_zero:
                        scaled = [(0.0, p) for _, p in ref_for_ds]
                    elif ref_sum > 0:
                        scaled = [(ds['portion'] * w / ref_sum, p)
                                  for w, p in ref_for_ds]
                    else:
                        # ref 全为 0：按桶数等分 ds.portion
                        equal = ds['portion'] / n_ref
                        scaled = [(equal, p) for _, p in ref_for_ds]

                    filtered, dropped = filter_low_entry_buckets(
                        scaled, bcs, effective_min,
                    )
                    if dropped:
                        dropped_low_entry_total += len(dropped)
                        for bkt, ec, w, _p in dropped:
                            dropped_low_entry_details.append((ds['text'], data_name, bkt, ec))
                            diversions_to_merge_small[bkt] += w
                    if not filtered:
                        dropped_full_dataset.append((ds['text'], data_name, ds['portion']))
                        continue
                    for rw, rp in filtered:
                        _emit(rw, rp, ds)
                    if not is_zero:
                        nonups_used_ref += 1
                    continue
                # reference 提供了但该数据集在 ref 里找不到：仍输出，按 csv path 占一行
                nonups_fallback.append((ds['text'], data_name, ds['portion']))
            _emit(ds['portion'], ds['path'], ds)
            continue

        try:
            dataset_dir, src = resolve_dataset_dir(ds['path'], name_map)
        except FileNotFoundError as e:
            print(f"  [警告] {ds['text']}: 解析物理路径失败，按未上采样处理 ({e})")
            _emit(ds['portion'], ds['path'], ds)
            skipped_resolve += 1
            continue

        try:
            split_info = build_data_split_info(dataset_dir)
        except (NotImplementedError, RuntimeError, OSError) as e:
            print(f"  [警告] {ds['text']}: 探测 layout 失败，按未上采样处理 ({e})")
            _emit(ds['portion'], ds['path'], ds)
            skipped_layout += 1
            continue

        ssf = split_info['sub_sub_folder']
        if ssf is None:
            layout_label = 'direct_files'
        elif ssf == ['qa']:
            layout_label = 'qa_only'
        elif len(ssf) == 1:
            layout_label = 'single_bucket'
        else:
            layout_label = 'multi_buckets'
        layout_summary[layout_label] += 1

        data_name = os.path.basename(dataset_dir.rstrip("/"))
        bucket_counts = None
        if effective_min > 0 and layout_label == 'multi_buckets':
            if entry_counts is not None and data_name in entry_counts:
                bucket_counts = entry_counts[data_name]
            else:
                no_entry_info_datasets.append((ds['text'], data_name))

        weight_paths, dropped, diverted = compute_split_weights(
            split_info, ds['portion'], args.ld_len,
            bucket_counts=bucket_counts, min_entry_count=effective_min,
        )
        for w, p in weight_paths:
            _emit(w, p, ds)

        if dropped:
            dropped_low_entry_total += len(dropped)
            for bkt, ec in dropped:
                dropped_low_entry_details.append((ds['text'], data_name, bkt, ec))
        for bkt, w in diverted.items():
            diversions_to_merge_small[bkt] += w

        if not weight_paths:
            dropped_full_dataset.append((ds['text'], data_name, ds['portion']))

        n_short = n_long = 0
        if ssf and len(ssf) > 1:
            div_k = cfg['divider_k']
            kept = [b for b in ssf if not any(b == d_b for d_b, _ in dropped)]
            n_short = sum(1 for b in kept if _bucket_lower_k(b) < div_k)
            n_long = len(kept) - n_short

        upsample_records.append({
            'text':    ds['text'],
            'type':    ds['type'],
            'cls':     ds['cls'],
            'portion': ds['portion'],
            'layout':  layout_label,
            'src':     src,
            'n_short': n_short,
            'n_long':  n_long,
            'n_bucket': len(ssf) if ssf else 0,
            'n_dropped': len(dropped),
        })

    appended_from_ref = []
    diverted_total_added = 0.0
    diverted_buckets_consumed = set()
    if ref_entries:
        existing_paths = {p for _, p in output_portion_paths}
        for w, p in ref_entries:
            if p in existing_paths:
                continue
            extra = 0.0
            if '/merge_small/decay/' in p:
                last = os.path.basename(p.rstrip('/'))
                if last in diversions_to_merge_small:
                    extra = diversions_to_merge_small[last]
                    diverted_total_added += extra
                    diverted_buckets_consumed.add(last)
            final_w = w + extra
            _emit(final_w, p, None)
            appended_from_ref.append((final_w, p))
        if appended_from_ref:
            n_zero = sum(1 for w, _ in appended_from_ref if w == 0)
            n_nonzero = len(appended_from_ref) - n_zero
            ref_sum_extra = sum(w for w, _ in appended_from_ref)
            print('\n' + '=' * 72)
            print(f"从 reference 补齐 {len(appended_from_ref)} 个路径"
                  f"（{n_zero} 个 weight=0 占位, {n_nonzero} 个 weight>0），"
                  f"补齐路径累计权重 {ref_sum_extra:.9f}"
                  f"（其中 1k 切下来转移到 merge_small/decay 的 {diverted_total_added:.9f}）")
        leftover = {b: w for b, w in diversions_to_merge_small.items()
                    if b not in diverted_buckets_consumed and w > 0}
        if leftover:
            print(f"  [警告] {len(leftover)} 个 bucket 的转移权重未在 reference 找到对应"
                  f" merge_small/decay 路径，将无处可去:")
            for b, w in leftover.items():
                print(f"    {b:12s}  {w:.9f}")

    total_portion = sum(p for p, _ in output_portion_paths)
    upsample_total = sum(d['portion'] for d in upsample_records)

    if ref_path_order:
        n_in_ref = sum(1 for _, p in output_portion_paths if p in ref_path_order)
        n_total = len(output_portion_paths)
        print('\n' + '=' * 72)
        print(f"按 reference 顺序排序输出: {n_in_ref}/{n_total} 行命中 reference，"
              f"剩 {n_total - n_in_ref} 行追加在末尾")
        indexed = list(enumerate(output_portion_paths))
        indexed.sort(key=lambda kv: (
            (0, ref_path_order[kv[1][1]]) if kv[1][1] in ref_path_order else (1, kv[0])
        ))
        output_portion_paths = [item for _, item in indexed]

    print('\n' + '=' * 72)
    print(f"校验所有输出 path 是否存在 (共 {len(output_portion_paths)} 行)")
    missing = [(w, p) for w, p in output_portion_paths if not os.path.exists(p)]
    if missing:
        print(f"  [错误] 发现 {len(missing)} 个不存在的 path：")
        for w, p in missing:
            print(f"    {w:.9f}  {p}")
        raise FileNotFoundError(
            f"{len(missing)} 个 path 不存在，未写出 {args.output_file}；请检查 csv 中的路径或 maybe_root_paths"
        )
    print(f"  全部 path 存在 ✓")

    is_sh = args.output_file.endswith('.sh')
    with open(args.output_file, 'w') as f:
        if is_sh:
            f.write('DATA_PATH="\n')
        for portion, path in output_portion_paths:
            f.write(f"{portion:.9f} {path}\n")
        if is_sh:
            f.write('"\n')

    print('\n' + '=' * 72)
    print(f"输出 {len(output_portion_paths)} 行 -> {args.output_file}")
    print(f"输出总配比 = {total_portion:.9f}")
    print(f"类型=舍弃 的行（已丢弃，不出现在输出）: {len(skipped_discarded)}")
    print(f"portion<=0 的行: {skipped_zero}（按 path 占位写出，权重=0）")
    print(f"upsample 解析路径失败: {skipped_resolve}")
    print(f"upsample layout 探测失败: {skipped_layout}")

    if skipped_discarded:
        print(f"\n类型=舍弃 的数据集（共 {len(skipped_discarded)} 个）:")
        for text, cls, portion in skipped_discarded:
            print(f"  {text[:40]:40s}  类别={cls[:14]:14s}  原配比={portion:.9f}")

    if effective_min > 0:
        print(f"\n因条数 <= {effective_min} 被丢弃的分桶（共 {dropped_low_entry_total} 个）:")
        for text, dn, bkt, ec in dropped_low_entry_details:
            print(f"  {text[:40]:40s}  ds={dn[:40]:40s}  桶={bkt:12s}  条数={ec}")
        if dropped_full_dataset:
            print(f"\n[警告] 全部桶都被丢弃 -> 整个数据集失去权重 (共 {len(dropped_full_dataset)} 个):")
            for text, dn, portion in dropped_full_dataset:
                print(f"  {text[:40]:40s}  ds={dn[:40]:40s}  原配比={portion:.9f}")
        if no_entry_info_datasets:
            print(f"\n[警告] 在 entry_count_json 中找不到的 multi_buckets 数据集 (共 {len(no_entry_info_datasets)} 个，未做过滤):")
            for text, dn in no_entry_info_datasets:
                print(f"  {text[:40]:40s}  ds={dn}")

    if reference_grouped is not None:
        print(f"\n非上采样: 使用 reference 自然配比展开 {nonups_used_ref} 个数据集")
        if nonups_fallback:
            print(f"非上采样: reference 中未找到，回退到 csv path 输出（共 {len(nonups_fallback)} 个）:")
            for text, dn, portion in nonups_fallback:
                print(f"  {text[:40]:40s}  ds={dn[:40]:40s}  配比={portion:.9f}")

    if appended_from_ref:
        n_merge_small = sum(1 for _, p in appended_from_ref if '/merge_small/' in p)
        n_other = len(appended_from_ref) - n_merge_small
        print(f"\n从 reference 额外补齐的路径明细 (共 {len(appended_from_ref)} 个，"
              f"其中 merge_small/* {n_merge_small} 个, 其它 {n_other} 个):")
        for w, p in appended_from_ref:
            print(f"  {w:.9f}  {p}")

    def _classify_path(path):
        """Return (type, cls) label for an output path.

        优先级：
          1) merge_small/decay/* -> 'merge_small_decay'
          2) merge_small/*       -> 'merge_small_stable'
          3) 该 path 由 CSV 行 _emit 出来：使用记录的 type/cls
          4) 该 path 来自 ref 补齐：用 data_name 反查 CSV 行的 type/cls
          5) 兜底：'unknown'
        """
        if '/merge_small/decay/' in path:
            return 'merge_small_decay', 'merge_small_decay'
        if '/merge_small/' in path:
            return 'merge_small_stable', 'merge_small_stable'
        if path in path_to_type:
            t, c = path_to_type[path], path_to_cls[path]
        else:
            dn = ref_path_to_dataname.get(path)
            if dn and dn in csv_ds_by_dataname:
                d = csv_ds_by_dataname[dn]
                t, c = d['type'], d['cls']
            else:
                t, c = 'unknown', 'unknown'
        return (t or '(空type)'), (c or '(空class)')

    all_by_type = defaultdict(float)
    all_by_cls = defaultdict(float)
    all_cnt_by_type = defaultdict(int)
    all_cnt_by_cls = defaultdict(int)
    for w, p in output_portion_paths:
        t, c = _classify_path(p)
        all_by_type[t] += w
        all_by_cls[c]  += w
        all_cnt_by_type[t] += 1
        all_cnt_by_cls[c]  += 1

    denom = total_portion if total_portion > 0 else 1.0
    print('\n' + '=' * 72)
    print(f"全部输出 - 按 类型 汇总  (总权重 {total_portion:.9f}):")
    print(f"  {'type':22s}  {'#行':>6s}  {'权重':>14s}  {'占比':>9s}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*14}  {'-'*9}")
    for t, w in sorted(all_by_type.items(), key=lambda x: -x[1]):
        print(f"  {t:22s}  {all_cnt_by_type[t]:6d}  {w:14.9f}  {w/denom*100:8.4f}%")

    print(f"\n全部输出 - 按 类别 汇总:")
    print(f"  {'class':22s}  {'#行':>6s}  {'权重':>14s}  {'占比':>9s}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*14}  {'-'*9}")
    for c, w in sorted(all_by_cls.items(), key=lambda x: -x[1]):
        print(f"  {c:22s}  {all_cnt_by_cls[c]:6d}  {w:14.9f}  {w/denom*100:8.4f}%")

    if upsample_total <= 0:
        print('\n没有需要上采样的行')
        return

    print('\n' + '=' * 72)
    print(f"上采样总配比 = {upsample_total:.9f} "
          f"({upsample_total/total_portion*100:.4f}% of all)")
    print(f"long_{args.ld_len} 配置: divider_k={cfg['divider_k']}k, "
          f"short_ratio={cfg['short_ratio']}, long_ratio={cfg['long_ratio']}")

    print("\n上采样行的 layout 分布:")
    for label, cnt in sorted(layout_summary.items()):
        print(f"  {label:15s}  {cnt:4d}")

    print("\n上采样详情:")
    print(f"  {'文本':40s}  {'类别':12s}  {'类型':10s}  {'layout':14s}  "
          f"{'src':6s}  {'配比':>12s}  {'#bkt':>5s}  {'#sh':>4s}  {'#lg':>4s}  {'#dr':>4s}")
    print(f"  {'-'*40}  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*6}  "
          f"{'-'*12}  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*4}")
    for d in sorted(upsample_records, key=lambda x: -x['portion']):
        print(f"  {d['text'][:40]:40s}  {d['cls'][:12]:12s}  {d['type'][:10]:10s}  "
              f"{d['layout']:14s}  {d['src']:6s}  {d['portion']:12.9f}  "
              f"{d['n_bucket']:5d}  {d['n_short']:4d}  {d['n_long']:4d}  "
              f"{d['n_dropped']:4d}")

    by_cls = defaultdict(float)
    by_type = defaultdict(float)
    for d in upsample_records:
        by_cls[d['cls']]   += d['portion']
        by_type[d['type']] += d['portion']

    print("\n上采样行 - 按 类别 汇总:")
    for cls, w in sorted(by_cls.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s}  {w:.9f}  ({w/upsample_total*100:.4f}% of upsample, "
              f"{w/total_portion*100:.4f}% of all)")

    print("\n上采样行 - 按 类型 汇总:")
    for t, w in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t:20s}  {w:.9f}  ({w/upsample_total*100:.4f}% of upsample, "
              f"{w/total_portion*100:.4f}% of all)")


if __name__ == "__main__":
    args = parse_args()
    gen_minicpm5_longdecay(args)
