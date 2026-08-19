#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=.:$PYTHONPATH

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

# train job
GPUS_PER_NODE=${GPUS_PER_NODE:-`nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l`}
DATA_PATH="0.190032317 /user/zhangyixuan/data/minicpm5/minicpm5_tokenizer.128k.dedup_link_all/Chinese_CC_HQ_TextCNN_merge"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
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
    # --moe-shared-expert-overlap
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
    --tokenizer-model /user/yanhui/utils/tokenizer_v9.6/
    # --data-path $DATA_PATH
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
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    # --log-throughput \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH \
    # --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --log-task-loss-interval 100 \
    --async-save \
    --ckpt-format torch_dist \
    # --use-persistent-ckpt-worker
)

set -ex
torchrun ${DISTRIBUTED_ARGS[@]} tools/checkpoint/dist_ckpt_to_hf_minicpm.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    $@

python tools/checkpoint/ckpt_to_hf_with_ep.py \
    --num_layer 28 \
    --in_dir $4 \
    --save_path $4

rm $4/pytorch_model.ep*
cp /user/yanhui/ckpts/minicpm5/demo.16a3b.4k.hf/* $4