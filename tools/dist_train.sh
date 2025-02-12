#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Set the CUDA_VISIBLE_DEVICES to the specified GPUs (GPUS should be a comma-separated string, e.g. "0,2,3")
export CUDA_VISIBLE_DEVICES=$GPUS

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Ensure GPUS is passed as an integer to nproc_per_node, not as a string like "0,1"
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
