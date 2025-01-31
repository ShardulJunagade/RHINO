#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=${GPUS:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Specify the GPUs to use
export CUDA_VISIBLE_DEVICES=2,3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/val.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch ${@:3}


# bash tools/dist_val.sh 'configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_brickkilns.py' 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_brickkilns1/epoch_36.pth'