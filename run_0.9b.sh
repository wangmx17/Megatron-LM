export GPUS_PER_NODE=8
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=23456
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LONGROPE_LONG_FACTOR="1.0010494258559746 1.1436603353159471 1.3154655424059682 1.5223889965537758 1.7715329912081192 2.0714018485306207 2.4321632151234898 2.8659510456482296 3.3872136600538414 4.013108831254348 4.763945341554351 5.6636663700989995 6.740363842504566 8.026803764273394 9.560929764361573 11.386294813826394 13.552348827116225 16.114482714889405 19.133698814550215 22.67574698448923 26.80954169028272 31.604668947790593 37.127818259142856 43.438052017492026 50.58097048841974 58.58205244282615 67.43973712153787 77.11911722722213 87.54735236617748 98.61197903327556 110.16308615553697 122.01980354101696 133.9807806186324 145.83749800411235 157.3886051263738 168.45323179347184 178.88146693242723 188.56084703811146 197.41853171682317 205.4196136712296 212.5625321421573 218.8727659005065 224.39591521185875 229.1910424693666 233.3248371751601 236.8668853450991 239.8861014447599 242.44823533253313 244.61428934582293 246.43965439528776 247.97378039537594 249.26022031714479 250.33691778955034 251.236638818095 251.987475328395 252.6133704995955 253.13463311400113 253.56842094452585 253.9291823111187 254.2290511684412 254.47819516309556 254.6851186172434 254.85692382433336 254.99953473379335"

pip install transformers==4.57.1 -i https://pypi.hs1.paratera.com/root/pypi/+simple
pip install /user/yanhui/code/DeepEP/dist/*.whl -i https://pypi.hs1.paratera.com/root/pypi/+simple
pip install /user/yanhui/share_user_long/qiqi/wheels/modelbest_sdk-0.3+feat.queue-py3-none-any.whl --force-reinstall
pip install /user/yanhui/whls/infllm_v2-0.0.0-cp312-cp312-linux_x86_64.whl -i https://pypi.hs1.paratera.com/root/pypi/+simple

##############################################################################
# Phase 1: Long Decay 32k (75B tokens)
#   lr: 5.22e-05 -> 9.28e-06 (总降 10x 的前 75%)
#   GBS=128, seq=32768, tokens/step=4M (与 short decay 一致)
#   128 * 32768 * 18000 = 75,497,472,000 ≈ 75.5B tokens
#   数据: bingxing.260321.long_decay_32k.split.sh
#   Load: short decay ckpt (job 212612, iter 528000)
#   iter 528000 -> 546000, WSD 最后 18000 步指数衰减
##############################################################################
source examples/minicpm5/data_conf/bingxing.260321.long_decay_32k.split.sh && \
bash examples/minicpm5/train_0.9b.sh \
    --tensorboard-dir /data/tensorboard/ \
    --save /user/${USER_NAME}/minicpm5/0.9b/${JOB_ID}/ \
    --load /para_hdd/user/qiqi/projects/1682-minicpm5/212612/checkpoints/shortdecay/ \
    --ckpt-step 528000 \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --lr 0.0000522 \
    --min-lr 0.00000928 \
    --train-iters 546000 \
    --lr-warmup-iters 0 \
    --lr-decay-style WSD \
    --lr-wsd-decay-style exponential \
    --lr-wsd-decay-iters 18000 \
    --lr-decay-iters 546000 \
    --log-task-loss-interval 100 \
    --log-hidden-rms-interval 100 \
    --log-interval 1 \
    --save-interval 3000 \
    --distributed-timeout-minutes 60 \
    --no-load-optim \
    --override-opt_param-scheduler \
    --clear-sampler-state \
    --use-rope-scaling \
    --rope-type longrope \
    --longrope-long-factor $LONGROPE_LONG_FACTOR \
    --use-span-based-attn \
    --num-workers 1 \
    --data-path $DATA_PATH \
    --timing-log-level 2 \
    --timing-log-option minmax

##############################################################################
# Phase 2: Long Decay 128k (25B tokens)
#   lr: 9.28e-06 -> 5.22e-06 (总降 10x 的后 25%)
#   256 GPU, TP2 CP2, DP=64, GBS=64, seq=131072, tokens/step=8.4M
#   64 * 131072 * 3000 = 25,165,824,000 ≈ 25.2B tokens
#   数据: bingxing.260321.long_decay_128k.split.sh
#   Load: 32k longdecay ckpt (iter 546000)
#   iter 546000 -> 549000, WSD 最后 3000 步指数衰减
##############################################################################
# export CP_SIZE=2
# source examples/minicpm5/data_conf/bingxing.260321.long_decay_128k.split.sh && \
# bash examples/minicpm5/train_0.9b.sh \
#     --tensorboard-dir /data/tensorboard/ \
#     --save /user/${USER_NAME}/minicpm5/0.9b/${JOB_ID}/ \
#     --load <32k_longdecay_ckpt_path> \
#     --ckpt-step 546000 \
#     --micro-batch-size 1 \
#     --global-batch-size 64 \
#     --seq-length 131072 \
#     --max-position-embeddings 131072 \
#     --tensor-model-parallel-size 2 \
#     --context-parallel-size ${CP_SIZE} \
#     --lr 0.00000928 \
#     --min-lr 0.00000522 \
#     --train-iters 549000 \
#     --lr-warmup-iters 0 \
#     --lr-decay-style WSD \
#     --lr-wsd-decay-style exponential \
#     --lr-wsd-decay-iters 3000 \
#     --lr-decay-iters 549000 \
#     --log-task-loss-interval 100 \
#     --log-hidden-rms-interval 100 \
#     --log-interval 1 \
#     --save-interval 1000 \
#     --distributed-timeout-minutes 60 \
#     --no-load-optim \
#     --override-opt_param-scheduler \
#     --clear-sampler-state \
#     --use-rope-scaling \
#     --rope-type longrope \
#     --longrope-long-factor $LONGROPE_LONG_FACTOR \
#     --use-span-based-attn \
#     --num-workers 1 \
#     --data-path $DATA_PATH \
#     --timing-log-level 2 \
#     --timing-log-option minmax

##############################################################################
# Phase 3: LongCOT Stage1 64k (100B tokens)
#   lr: 0 -> 5.22e-05 -> 5.22e-06 (warmup 500 步, WSD exp decay 23500 步)
#   128 GPU, TP2, DP=64, GBS=64, seq=65536, tokens/step=4M
#   64 * 65536 * 24000 = 100,663,296,000 ≈ 100.7B tokens
#   数据: 260318.long_cot_sft.stage1.sh (复用 16A3B stage1 数据)
#   Load: 128k longdecay ckpt (iter 549000)
#   train-iters 24000, save-interval 3000
#   注: lr 比例与旧 0.9B 一致 (lr=stable/10, min-lr=stable/109)
##############################################################################
# source examples/minicpm5/data_conf/260318.long_cot_sft.stage1.sh && \
# bash examples/minicpm5/train_0.9b.sh \
#     --tensorboard-dir /data/tensorboard/ \
#     --save /user/${USER_NAME}/minicpm5/0.9b/${JOB_ID}/ \
#     --load <128k_longdecay_ckpt_path> \
#     --ckpt-step 549000 \
#     --micro-batch-size 1 \
#     --global-batch-size 64 \
#     --seq-length 65536 \
#     --max-position-embeddings 131072 \
#     --tensor-model-parallel-size 2 \
#     --lr 0.0000522 \
#     --min-lr 0.00000522 \
#     --train-iters 24000 \
#     --lr-warmup-iters 500 \
#     --lr-decay-style WSD \
#     --lr-wsd-decay-style exponential \
#     --lr-wsd-decay-iters 23500 \
#     --lr-decay-iters 24000 \
#     --log-task-loss-interval 100 \
#     --log-hidden-rms-interval 100 \
#     --log-interval 1 \
#     --save-interval 3000 \
#     --distributed-timeout-minutes 60 \
#     --no-load-optim \
#     --override-opt_param-scheduler \
#     --finetune \
#     --clear-sampler-state \
#     --no-load-data-state \
#     --use-rope-scaling \
#     --rope-type longrope \
#     --longrope-long-factor $LONGROPE_LONG_FACTOR \
#     --use-span-based-attn \
#     --log-throughput \
#     --num-workers 1 \
#     --data-path $DATA_PATH \
#     --timing-log-level 2 \
#     --timing-log-option minmax
