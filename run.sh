# export CUDA_VISIBLE_DEVICES=0,1
export GPUS_PER_NODE=8
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=23456
export CP_SIZE=2
export MEGATRON_CP_SIZE=${CP_SIZE}
export GLOBAL_BATCH_SIZE=2
export SEQ_LENGTH=4096
export NUM_LAYERS=1
export NNODES=${WORLD_SIZE:-"1"}
export NODE_RANK=${RANK:-"0"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

pip install transformers==4.57.1 -i https://pypi.hs1.paratera.com/root/pypi/+simple
pip install /user/yanhui/code/DeepEP/dist/*.whl -i https://pypi.hs1.paratera.com/root/pypi/+simple
pip install /user/yanhui/share_user_long/qiqi/modelbest_sdk/dist/modelbest_sdk-0.3+cp.pad.fix-py3-none-any.whl --force-reinstall
pip install /user/yanhui/whls/infllm_v2-0.0.0-cp312-cp312-linux_x86_64.whl -i https://pypi.hs1.paratera.com/root/pypi/+simple

source examples/minicpm5/data_conf/260306.sft.sh && \
bash examples/minicpm5/train_16a3b.128k.sh \
    --tensorboard-dir /data/tensorboard/ \
    --save /user/yanhui/minicpm5/16a3b/${JOB_ID}/ \
    --micro-batch-size 1 \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --train-iters 12500 \
    --lr-warmup-iters 500 \
    --lr-decay-style WSD \
    --lr-wsd-decay-style exponential \
    --lr-wsd-decay-iters 12000 \
    --lr-decay-iters 12500 \
    --log-task-loss-interval 100 \
    --log-hidden-rms-interval 100 \
    --log-interval 1 \
    --save-interval 200 \
    --distributed-timeout-minutes 60 \
    --data-path $DATA_PATH \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --no-create-attention-mask-in-dataloader \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size ${MEGATRON_CP_SIZE} \
    --moe-aux-loss-coeff 0 \
    --moe-router-bias-update-rate 0 \
    --moe-token-dispatcher-type flex \
    --moe-flex-dispatcher-backend hybridep \
    --use-span-based-attn \
    --override-opt_param-scheduler \
    --no-load-data-state \
    # --load /user/yanhui/minicpm5/16a3b/196577/ \
    # --ckpt-step 2983 \
