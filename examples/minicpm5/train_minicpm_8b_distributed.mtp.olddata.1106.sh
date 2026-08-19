#!/bin/bash

# sudo chmod 777 -R /local/apps/Megatron-LM/
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TRITON_PTXAS_PATH=/usr/local/cuda-13.1/bin/ptxas
export TRITON_LIBCUDA_PATH=/usr/local/cuda-13.1/lib64/libcuda.so
export TRITON_CUDA_HOME=/usr/local/cuda-13.1
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CP_SIZE=4
# train job
pip install /user/yanhui/whls/modelbest_sdk-0.3.1-py3-none-any.whl
# Apply ModelBest SDK patches for attention mask handling
MODELBEST_SDK_PATH=$(python -c "import os, modelbest_sdk; print(os.path.dirname(modelbest_sdk.__file__))")
cp modelbest_sdk_patches/megatron_batch_packer.py ${MODELBEST_SDK_PATH}/dataset/batch_packer/megatron_batch_packer.py
cp modelbest_sdk_patches/base_doc.py ${MODELBEST_SDK_PATH}/dataset/thrift_wrapper/base_doc.py

# FP8 env prepare ###########################c
# if [[ ${USE_FP8_GEMM} -eq 1 ]]; then
#     echo ">>> Preparing DeepSeek FP8 env ......"
#     pip install /user/yanhui/whls/deep_gemm-1.0.0-py3-none-any.whl
#     pip install /user/yanhui/whls/fp8_quant-0.0.1-cp39-abi3-manylinux2014_x86_64.whl
#     cd tools/fp8_gemm/ && python patch.py && cd -
#     echo ">>> DeepSeek FP8 env installed ......" 
#     cp tools/te_fp8_patch/distributed.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/distributed.py
#     cp tools/te_fp8_patch/layernorm_linear.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/module/layernorm_linear.py
#     cp tools/te_fp8_patch/layernorm_mlp.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/module/layernorm_mlp.py
#     cp tools/te_fp8_patch/linear.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/module/linear.py
# fi
#############################################

GPUS_PER_NODE=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
TOKENIZER_MODEL=/user/pretrain/120k_v2.model


VALID_DATA_PATH='1.0 //user/xuxiaoyue/LongPPL/sst/govreport'
# for debug
# pip install /mnt/data/user/tc_agi/whl/modelbest_sdk/modelbest_sdk-0.2.5.7-py3-none-any.whl
# export CUDA_VISIBLE_DEVICES=0
# GPUS_PER_NODE=2
# MASTER_ADDR=localhost
# MASTER_PORT=6420
# WORLD_SIZE=1
# RANK=0
# CHECKPOINT_PATH='./temp/'
# TOKENIZER_MODEL=/mnt/data/user/tc_agi/dzn/tokenizer/models/120k_v2.model
# DATA_PATH="
#     0.75 /home/jeeves/tangpeijun/refactor/Megatron-LM/c4_sample_20w_text_document
#     0.25 /home/jeeves/tangpeijun/refactor/Megatron-LM/redpajama_sample_20w_text_document
# "
# TENSORBOARD_LOGS_PATH="./tensorboard"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    # --nnodes $WORLD_SIZE 
    # --node_rank $RANK
    # --master_addr $MASTER_ADDR 
    # --master_port $MASTER_PORT
    --standalone
)

MAX_LEN=32768
GPT_MODEL_ARGS=(
    --vocab-size 73448
    --make-vocab-size-divisible-by 1
    --num-layers 8
    --hidden-size 4096
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 2
    --kv-channels 128
    --ffn-hidden-size 16384
    --seq-length $MAX_LEN
    --max-position-embeddings $MAX_LEN
    --attention-dropout 0
    --hidden-dropout 0
    --swiglu
    --position-embedding-type rope
    --disable-bias-linear
    --use-flash-attn
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    # --use-mup
    # --mup-emb-scale 12
    # --mup-depth-scale 1.4
    --init-method-std 0.1
    --use-rope-scaling
    --rope-type longrope
    --longrope-long-factor 0.9977997200264581 1.014658295992452 1.0349680404997148 1.059429246056193 1.0888815016813513 1.1243301355211495 1.166977103606075 1.2182568066927284 1.2798772354275727 1.3538666751582975 1.4426259039919596 1.5489853358570191 1.6762658237220625 1.8283407612492941 2.0096956085876183 2.225478927469756 2.481536379650452 2.784415934557119 3.1413289096347365 3.560047844772632 4.048719380066383 4.615569542115128 5.2684819496549835 6.014438591970396 6.858830049237097 7.804668263503327 8.851768731513417 9.99600492938444 11.228766118181639 12.536757560834843 13.902257701387796 15.303885189125953 16.717837610115794 18.119465097853947 19.484965238406907 20.792956681060105 22.02571786985731 23.16995406772833 24.217054535738416 25.16289275000465 26.007284207271347 26.753240849586767 27.40615325712662 27.973003419175363 28.461674954469114 28.880393889607006 29.237306864684626 29.540186419591297 29.79624387177199 30.01202719065413 30.193382037992453 30.34545697551969 30.47273746338473 30.579096895249787 30.66785612408345 30.741845563814174 30.80346599254902 30.85474569563567 30.897392663720595 30.932841297560394 30.962293553185553 30.986754758742034 31.007064503249293 31.02392307921529
)

TRAINING_ARGS=(
    # --micro-batch-size 8
    # --global-batch-size 8
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
    --bf16
    # --lr 1e-2
    # --min-lr 0.000625
    # --train-iters 5000
    # --lr-warmup-iters 100
    # --lr-decay-style WSD
    # --lr-wsd-decay-style exponential
    # --lr-wsd-decay-iters 600
    # --lr-decay-iters 5000
    --use-mcore-models
    --norm-epsilon 1e-6
)

MODEL_PARALLEL_ARGS=(
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 1
    --use-distributed-optimizer
    # --overlap-grad-reduce
    # --overlap-param-gather
    # --tensor-model-parallel-size 2
    # --sequence-parallel
    # --tp-comm-overlap
    --use-dist-ckpt
    --dist-ckpt-format torch_dist
    --async-save
    # --fp8-format hybrid
    # --fp8-amax-history-len 1024
    # --fp8-amax-compute-algo max
    # --log-throughput
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer  
    --tokenizer-model /user/sunao/zhangyan/120k_v2.model
    # --data-path $DATA_PATH
    --split 99990,8,2
    --use-modelbest-sdk
    # --no-load-data-state
    --dataloader-type external
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    # --log-task-loss-interval 1
    # --save-interval 1000
    --eval-interval 5000000
    # --save temp/ 
    --eval-iters 1
    --log-validation-ppl-to-tensorboard
    # --load $CHECKPOINT_PATH
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)
export PYTHONPATH=/user/sunao/zhangyan/Megatron-LM/infllmv2_cuda_impl:$PYTHONPATH
set -ex
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_minicpm.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    $@
