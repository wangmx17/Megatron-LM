"""Submit scaling bench (PPL) evaluation jobs to Cybertron.

Usage:
    python tools/long_context/eval_scaling_bench.py \
        --model_paths /user/yanhui/ckpts/minicpm5/16a3b/job_208301.step_1593000

    # Multiple models
    python tools/long_context/eval_scaling_bench.py \
        --model_paths \
            /user/yanhui/ckpts/minicpm5/16a3b/job_196207.step_1284000 \
            /user/yanhui/ckpts/minicpm5/16a3b/job_208301.step_1593000

    # Dry run (only print, do not submit)
    python tools/long_context/eval_scaling_bench.py --model_paths ... --dry_run
"""

import argparse
import json
import requests

CYBERTRON_API = "https://cybertron.modelbest.co/api/job"
TOKEN = "EPUfWL0zmpv3hFuQYZgvrA"
PROJECT_ID = 680

DATASET = "scaling_bench_all_v2_gen_828809"
INFER_TYPE = "vllm_scalingbench_ppl"


def build_entry(model_path: str, infer_type: str, dataset: str) -> str:
    return (
        "cd /user/linbiyuan/opencompass && "
        "ENV_NAME=sglang_minicpm5 source init.sh && "
        "CUDA_HOME=/usr/local/cuda TORCH_COMPILE_DISABLE=1 SGLANG_DISABLE_CUDNN_CHECK=1 OPENAI_API_KEY=KEY "
        f"python run_model.py "
        f"--dataset {dataset} "
        f"--model_path {model_path} "
        f"--infer_type {infer_type}"
    )


def build_job_payload(model_path: str, infer_type: str, dataset: str,
                      gpu_num: int, image: str,
                      resource_pool_id: int = 336,
                      priority: str = "normal",
                      cluster: str = "paratera_train") -> dict:
    return {
        "entry": build_entry(model_path, infer_type, dataset),
        "cluster": cluster,
        "namespace": "training",
        "priority": priority,
        "image": image,
        "training_type": "pytorchjob",
        "code_type": "image",
        "resource_pool_id": resource_pool_id,
        "replicas": {
            "master": {
                "replicas": 1,
                "gpu_num": gpu_num,
                "cpu_num": 80,
                "memory_size": 800,
                "gpu_series": "h100",
            },
            "worker": {
                "replicas": 0,
                "gpu_num": gpu_num,
                "cpu_num": 80,
                "memory_size": 800,
                "gpu_series": "h100",
            },
        },
    }


def submit_job(payload: dict, project_id: int, token: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    resp = requests.post(
        f"{CYBERTRON_API}?project_id={project_id}",
        headers=headers,
        json=payload,
    )
    return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}


def main():
    parser = argparse.ArgumentParser(description="Submit scaling bench PPL evaluation jobs to Cybertron")
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="HuggingFace model paths to evaluate")
    parser.add_argument("--infer_type", default=INFER_TYPE,
                        help=f"Infer type config name (default: {INFER_TYPE}). "
                             "Append _2gpu/_4gpu for multi-GPU tensor parallelism.")
    parser.add_argument("--dataset", default=DATASET,
                        help=f"Dataset config name (default: {DATASET})")
    parser.add_argument("--gpu_num", type=int, default=8,
                        help="Number of GPUs per node (default: 8)")
    parser.add_argument("--project_id", type=int, default=PROJECT_ID)
    parser.add_argument("--token", default=TOKEN)
    parser.add_argument("--image", default="modelbest/opencompass20250424:vllm-202506140057-27dedf")
    parser.add_argument("--resource_pool_id", type=int, default=336,
                        help="Resource pool ID (default: 336 = model-next)")
    parser.add_argument("--cluster", default="paratera_train",
                        help="Cluster name (default: paratera_train)")
    parser.add_argument("--priority", default="normal",
                        choices=["preemptable", "normal", "high"],
                        help="Job priority (default: normal)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print job payloads without submitting")
    args = parser.parse_args()

    results = []
    for model_path in args.model_paths:
        payload = build_job_payload(
            model_path=model_path,
            infer_type=args.infer_type,
            dataset=args.dataset,
            gpu_num=args.gpu_num,
            image=args.image,
            resource_pool_id=args.resource_pool_id,
            priority=args.priority,
            cluster=args.cluster,
        )

        print(f"\n{'='*60}")
        print(f"Model: {model_path}")
        print(f"Infer: {args.infer_type}  Dataset: {args.dataset}")
        print(f"Entry: {payload['entry']}")

        if args.dry_run:
            print("[DRY RUN] Skipping submission")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            continue

        resp = submit_job(payload, args.project_id, args.token)
        print(f"Response: {json.dumps(resp, ensure_ascii=False)}")
        results.append({"model": model_path, "response": resp})

    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"Submitted {len(results)} scaling bench jobs")
        for r in results:
            resp_data = r["response"].get("data", {})
            job_id = resp_data.get("id", r["response"].get("raw", "unknown"))
            print(f"  {r['model']} => job {job_id}")


if __name__ == "__main__":
    main()
