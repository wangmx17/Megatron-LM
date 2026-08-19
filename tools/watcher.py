import json
import os
import urllib.request
import urllib.error
import time
import logging

# 配置 logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("watcher.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("watcher")

os.environ["http_proxy"] = "http://whitelist-proxy.cybertron.svc.cluster.local:7891"
os.environ["https_proxy"] = "http://whitelist-proxy.cybertron.svc.cluster.local:7891"

url = "https://open.feishu.cn/open-apis/bot/v2/hook/45bde4d2-0745-4f97-b748-a4eaf0b6f563"


def send_feishu_message(text: str, job_id: str) -> dict:
    """向飞书机器人 Webhook 发送文本消息。

    Args:
        text: 要发送的消息内容。

    Returns:
        飞书 API 的响应结果（dict）。
    """
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": "MiniCPM5 Training Status",
                    "content": [[{
                            "tag": "a",
                            "text": "点击此处跳转至任务",
                            "href": f"https://cybertron.modelbest.co/job-detail?id={job_id}&projectId=1682&cluster=paratera_train"
                        }],
                        [{
                            "tag": "text",
                            "text": f"任务告警信息: {text}"
                        }]
                    ]
                }
            }
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    logger.info(f"正在发送飞书消息, job_id={job_id}, text={text[:80]}...")
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            logger.info(f"飞书消息发送成功: {result}")
            return result
    except urllib.error.HTTPError as e:
        logger.error(f"飞书消息发送失败 - HTTP 错误: {e.code} - {e.read().decode('utf-8')}")
        raise
    except urllib.error.URLError as e:
        logger.error(f"飞书消息发送失败 - URL 错误: {e.reason}")
        raise


if __name__ == "__main__":
    # send_feishu_message("test", "1234567890")
    # exit(1)
    latest_log_update_info = {
        "latest_update_time": None,
        "latest_update_file": None,
        "latest_checked_lines": 0
    }
    current_job_id = None

    logged_slow_iters = dict()
    logged_grad_norm = dict()
    logged_loss = dict()

    threshold_loss = 1.55
    threshold_grad_norm = 0.5
    threshold_iter_time = 1.6
    threshold_slow_iters = 20
    threshold_log_update_time = 5 #分钟

    checkpoint_interval = 60 #秒

    logger.info("监控程序已启动，检查间隔: %d 秒", checkpoint_interval)
    logger.info("阈值配置 - loss: %.2f, grad_norm: %.2f, iter_time: %.2f, slow_iters: %d, log_update_time: %d min",
                threshold_loss, threshold_grad_norm, threshold_iter_time, threshold_slow_iters, threshold_log_update_time)

    # check job status every 60 seconds
    while True:
        time.sleep(checkpoint_interval)
        logger.info("=" * 60)
        logger.info("开始新一轮检查...")

        # get latest running job id
        cmd = """curl --location --request GET 'https://cybertron.modelbest.co/api/project/jobs/?id=1682&page=1&limit=20&resource_pool_id=336' \
    --header 'User-Agent: Apifox/1.0.0 (https://apifox.com)' \
    --header 'Authorization: Bearer feishu-bot-minicpm5'"""

        logger.debug("正在请求任务列表...")
        try:
            response = os.popen(cmd).read()
            response = json.loads(response)
            logger.info("成功获取任务列表，共 %d 个任务", len(response["data"]["jobs"]))
        except Exception as e:
            logger.error("获取任务列表失败: %s", e)
            continue

        current_job_id = None
        for job in response["data"]["jobs"]:
            num_replica = job["replicas"]["master"]["replicas"] + job["replicas"]["worker"]["replicas"]
            logger.debug("任务 %s: status=%s, num_replica=%d", job["id"], job["status"], num_replica)
            if job["status"] == "Running" and num_replica == 96:
                current_job_id = job["id"]
                logger.info("找到目标运行任务: job_id=%s, num_replica=%d", current_job_id, num_replica)
                break
            if job["id"] == current_job_id and job["status"] == "Failed":
                send_feishu_message(f"任务 {current_job_id} 失败，请检查！！！", current_job_id)
                continue    
        
        # 没有检测到任务正在running的任务
        if not current_job_id:
            logger.warning("没有监测到任何 running 中的训练任务")
            send_feishu_message("没有监测到任何running中的训练任务，请检查！！！", current_job_id)
            continue

        # 获取任务最新日志
        log_dir = f"/projects/1682-minicpm5/{current_job_id}/logs/pytorchjob-minicpm5-{current_job_id}-worker-94/"
        logger.info("日志目录: %s", log_dir)
        # 找到日期最新的log文件
        latest_log_date, latest_log_file = None, None
        try:
            for file in os.listdir(log_dir):
                if file.startswith("pytorch.") and file.endswith(".log"):
                    date = file.split(".")[1]
                    if latest_log_date is None:
                        latest_log_date = date
                        latest_log_file = os.path.join(log_dir, file)
                    else:
                        if date > latest_log_date:
                            latest_log_date = date
                            latest_log_file = os.path.join(log_dir, file)
        except FileNotFoundError:
            logger.error("日志目录不存在: %s", log_dir)
            continue
        except Exception as e:
            logger.error("读取日志目录失败: %s", e)
            continue

        logger.info("最新日志文件: %s", latest_log_file)
        if latest_log_file != latest_log_update_info["latest_update_file"]:
            latest_log_update_info["latest_update_file"] = latest_log_file
            latest_log_update_info["latest_checked_lines"] = 0
        
        # 获取文件的更新时间
        update_time = os.path.getmtime(latest_log_file)
        logger.info("日志文件最后更新时间: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(update_time)))
        if latest_log_update_info["latest_update_time"] is None:
            latest_log_update_info["latest_update_time"] = update_time

        # print(update_time, flush=True)
        # print(latest_log_update_info["latest_update_time"], flush=True)
        # print(update_time - latest_log_update_info["latest_update_time"], flush=True)
        if (update_time - latest_log_update_info["latest_update_time"]) > threshold_log_update_time * 60:
            logger.warning("日志已经 %.1f 秒没有更新，超过阈值 %d 分钟", update_time - latest_log_update_info["latest_update_time"], threshold_log_update_time)
            send_feishu_message(f"日志已经{update_time - latest_log_update_info["latest_update_time"]}秒没有更新了，请检查！！！", current_job_id)
            continue
        latest_log_update_info["latest_update_time"] = update_time

        train_data_recoder = []
        logger.info("开始解析日志文件")
        try:
            with open(latest_log_file, "r") as f:
                lines = f.readlines()
                # for line in lines[latest_log_update_info["latest_checked_lines"]:]:
                for line in lines[-200:]:
                    if "elapsed time per iteration (ms):" in line:
                        iteration = line.split("|")[0].split("iteration")[1].split("/")[0].strip()
                        elapsed_time = float(line.split("|")[2].split(":")[1].strip())/1000
                        lm_loss = float(line.split("|")[5].split(":")[1].strip())
                        grad_norm = float(line.split("|")[-4].split(":")[1].strip())
                        train_data_recoder.append({
                            "iteration": iteration,
                            "elapsed_time": elapsed_time,
                            "lm_loss": lm_loss,
                            "grad_norm": grad_norm,
                        })
                latest_log_update_info["latest_checked_lines"] = len(lines)
        except Exception as e:
            logger.error("解析日志文件失败: %s", e)
            continue

        logger.info("解析到 %d 条训练数据记录", len(train_data_recoder))
        if train_data_recoder:
            latest = train_data_recoder[-1]
            logger.info("最新训练状态 - iteration: %s, loss: %.4f, grad_norm: %.4f, elapsed_time: %.3f s",
                        latest["iteration"], latest["lm_loss"], latest["grad_norm"], latest["elapsed_time"])
        
        # 检查loss和grad norm的异常值
        loss_warning_str = ""
        grad_norm_warning_str = ""
        for d in train_data_recoder:
            if d["lm_loss"] > threshold_loss:
                if (current_job_id, d["iteration"]) not in logged_loss:
                    logger.warning("loss 异常: %.4f (阈值: %.2f), iteration: %s", d["lm_loss"], threshold_loss, d["iteration"])
                    loss_warning_str += f"loss异常: {d['lm_loss']}, iteration: {d['iteration']}\n"
                logged_loss[(current_job_id, d["iteration"])] = 0
            if d["grad_norm"] > threshold_grad_norm:
                if (current_job_id, d["iteration"]) not in logged_grad_norm:
                    logger.warning("grad_norm 异常: %.4f (阈值: %.2f), iteration: %s", d["grad_norm"], threshold_grad_norm, d["iteration"])
                    grad_norm_warning_str += f"grad norm异常: {d['grad_norm']}, iteration: {d['iteration']}\n"
                logged_grad_norm[(current_job_id, d["iteration"])] = 0
        if loss_warning_str:
            send_feishu_message(f"{loss_warning_str}", current_job_id)
        if grad_norm_warning_str:
            send_feishu_message(f"{grad_norm_warning_str}", current_job_id)

        # 检查迭代时间是否连续n步超出预期值
        consecutive_slow_iters, begin_iter, end_iter = 0, None, None
        for i in range(len(train_data_recoder)):
            if (current_job_id, train_data_recoder[i]["iteration"]) in logged_slow_iters:
                continue

            if train_data_recoder[i]["elapsed_time"] > threshold_iter_time:
                if consecutive_slow_iters == 0:
                    begin_iter = train_data_recoder[i]["iteration"]
                end_iter = train_data_recoder[i]["iteration"]
                consecutive_slow_iters += 1
            else:
                consecutive_slow_iters = 0

        if consecutive_slow_iters > threshold_slow_iters:
            for i in range(int(begin_iter), int(end_iter) + 1):
                logged_slow_iters[(current_job_id, i)] = 0
            logger.warning("连续 %d 步 (%s-%s) 迭代时间超出预期值 %.2f s",
                            consecutive_slow_iters, begin_iter, end_iter, threshold_iter_time)
            send_feishu_message(f"连续{consecutive_slow_iters}步({begin_iter}-{end_iter})迭代时间超出预期值（{threshold_iter_time}s），请检查！！！", current_job_id)

        logger.info("本轮检查完成")
