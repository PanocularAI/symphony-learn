#!/usr/bin/bash

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}
TRAIN_FILE=${TRAIN_FILE:-"train"}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

MASTER_ADDR=${MASTER_ADDR:-"localhost"} # Set to the public IP of the master node (e.g. tailscale ip)
MASTER_PORT=${MASTER_PORT:-"29500"}
LOCAL_ADDR=${LOCAL_ADDR:-"localhost"} # Set to the public IP of the local node (e.g. tailscale ip)


NNODES=${NNODES:-"1"} # Total number of nodes within an island
ISHOST=${ISHOST:-"true"} # Set to true for the master node, false for other nodes in the same island

export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"tailscale0"} # Hint Gloo to use desired network interface, in this case tailscale
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"tailscale0"} # Hint NCCL to use desired network interface, in this case tailscale


PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
LOCAL_ADDR=${LOCAL_ADDR} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NNODES=${NNODES} \
ISHOST=${ISHOST} \
torchrun --nproc_per_node=${NGPU} --nnodes ${NNODES} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
--local_addr=${LOCAL_ADDR} --rdzv-conf is_host=${ISHOST} --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@"
