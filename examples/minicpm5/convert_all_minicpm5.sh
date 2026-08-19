#!/bin/bash
# Batch-convert every iter_* dist checkpoint under $SRC into its own HF ckpt.
# Model config is MiniCPM5 16A3B (matches examples/minicpm5/convert_16a3b.sh).
#
# 必须在装好 Megatron 依赖(torch/transformer_engine/grouped_gemm...)且有 8 张卡的环境里运行。
# 用法示例:
#   SRC=/user/licheng/models/minicpm5 \
#   OUT=/user/licheng/models/minicpm5_hf \
#   TOKENIZER=/path/to/minicpm5_hf_tokenizer_dir \
#   HF_CONFIG=/path/to/hf_template_dir \
#   bash examples/minicpm5/convert_all_minicpm5.sh

set -euo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=.:$PYTHONPATH

# -------- configurable --------
SRC=${SRC:-/user/licheng/models/minicpm5}          # 含 iter_* 的父目录
OUT=${OUT:-/user/licheng/models/minicpm5_hf}        # 输出根目录
TOKENIZER=${TOKENIZER:?请设置 TOKENIZER=HF tokenizer 目录}   # 必填
HF_CONFIG=${HF_CONFIG:-}                             # 可选: 含 config.json/tokenizer 的 HF 模板目录, 会拷进每个输出
ITERS=${ITERS:-}                                     # 留空则自动扫描 SRC 下所有 iter_*
GPUS_PER_NODE=${GPUS_PER_NODE:-8}                    # EP=8, 需要 8 个 rank
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

REPO=/user/licheng/megatron/Megatron-LM
cd "$REPO"

# 自动发现要转换的 iteration
if [ -z "$ITERS" ]; then
    ITERS=$(ls -d "$SRC"/iter_* 2>/dev/null | sed -E 's#.*/iter_0*([0-9]+)$#\1#' | sort -n)
fi
echo "[info] iterations to convert: $ITERS"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes 1
    --node_rank 0
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MODEL_ARGS=(
    --use-mcore-models
    --vocab-size 130560
    --make-vocab-size-divisible-by 1
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 4096
    --num-layers 28
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 10000
    --norm-epsilon 1e-6
    --elementwise-attn-output-gate
)

MOE_ARGS=(
    --num-experts 160
    --moe-router-topk 16
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 512
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-4
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-layer-freq [0]+[1]*27
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0.001
    --moe-router-pre-softmax
    --moe-router-topk-scaling-factor 3.66
    --moe-router-dtype fp32
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER"
    --split 99990,8,2
    --use-modelbest-sdk
    --no-load-data-state
    --dataloader-type external
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --lr 2e-4
    --min-lr 2e-5
    --train-iters 40000
    --lr-decay-iters 40000
    --lr-wsd-decay-iters 3000
    --lr-decay-style WSD
    --lr-wsd-decay-style exponential
    --weight-decay 0.1
    --lr-warmup-iters 400
    --clip-grad 1.0
    --bf16
    --optimizer muon
    --muon-split-qkv
    --muon-coefficient-type quintic
    --adam-beta1 0.9
    --adam-beta2 0.95
    --finetune
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --log-task-loss-interval 100
    --async-save
    --ckpt-format torch_dist
)

for N in $ITERS; do
    padded=$(printf "iter_%07d" "$N")
    if [ ! -d "$SRC/$padded" ]; then
        echo "[skip] $SRC/$padded not found"
        continue
    fi
    save="$OUT/$padded"
    if [ -f "$save/pytorch_model.bin" ]; then
        echo "[skip] already converted: $save"
        continue
    fi
    echo "==================== converting $padded ===================="
    # 用一个只含该 iteration 的临时目录, 让 Megatron 精确加载这一步
    stage=$(mktemp -d)
    ln -s "$SRC/$padded" "$stage/$padded"
    echo "$N" > "$stage/latest_checkpointed_iteration.txt"
    mkdir -p "$save"

    torchrun "${DISTRIBUTED_ARGS[@]}" tools/checkpoint/dist_ckpt_to_hf_minicpm.py \
        "${MODEL_ARGS[@]}" \
        "${MOE_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${LOGGING_ARGS[@]}" \
        --load "$stage" \
        --save "$save"

    # 合并 8 个 EP 分片 -> 单个 pytorch_model.bin
    python tools/checkpoint/ckpt_to_hf_with_ep.py \
        --num_layer 28 \
        --ep_num 160 \
        --ep_size 8 \
        --in_dir "$save" \
        --save_path "$save"

    rm -f "$save"/pytorch_model.ep*
    if [ -n "$HF_CONFIG" ]; then
        cp "$HF_CONFIG"/* "$save"/ || true
    fi
    rm -rf "$stage"
    echo "[done] -> $save"
done

echo "ALL DONE. HF checkpoints under: $OUT"
