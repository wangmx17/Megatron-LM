#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=.:$PYTHONPATH

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

# train job
GPUS_PER_NODE=${GPUS_PER_NODE:-`nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l`}
DATA_PATH="0.190032317 /user/traindoc/text/by_minicpm3_tokenizer.74k/the_stack_v2_rule_filtered_transformed"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --vocab-size 73448
    --make-vocab-size-divisible-by 1
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 4096
    --num-layers 24
    --hidden-size 1536
    --ffn-hidden-size 4608
    --num-attention-heads 16
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
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model /user/yanhui/models/MiniCPM4.1-8B/
    # --data-path $DATA_PATH
    --split 99990,8,2
    --use-modelbest-sdk
    --no-load-data-state
    --dataloader-type external
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
    --bf16
    --lr 1e-2
    --min-lr 0.000625
    --train-iters 5000
    --lr-warmup-iters 100
    --lr-decay-style WSD
    --lr-wsd-decay-style exponential
    --lr-wsd-decay-iters 600
    --lr-decay-iters 5000
)

MODEL_PARALLEL_ARGS=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-distributed-optimizer
    --tensor-model-parallel-size 1
    # --use-dist-ckpt
    # --dist-ckpt-format torch_dist
    # --async-save
    # --ckpt-format torch
    --sequence-parallel
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    # --log-task-loss-interval 10
    --save-interval 1000
    --eval-interval 1000000000
    # --save /user/yanhui/ckpts/mtp/job_25295/
    --eval-iters 100
    # --load /projects/429-minicpm4-deliver/25295/checkpoints/
    --finetune
    --async-save \
    --ckpt-format torch_dist \
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

set -ex
torchrun ${DISTRIBUTED_ARGS[@]} tools/checkpoint/dist_ckpt_to_hf_minicpm.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    $@