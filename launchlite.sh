#!/usr/bin/env bash
# Usage: ./launchlite.sh <gpu_ids> [grad_accum_steps]
#   gpu_ids            Comma-separated list of GPU indices to use, e.g. "0,1,2,3"
#   grad_accum_steps   (optional) number of gradient-accumulation steps per optimizer step (default: 1)

set -euo pipefail

GPU_IDS=${1:-0}
ACCUM=${2:-1}

# make GPU list absolute (remove trailing/leading spaces)
GPU_IDS=$(echo "$GPU_IDS" | tr -d '[:space:]')

# derive GPU count
IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARR[@]}

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NUM_GPUS="$NUM_GPUS"
export GRAD_ACCUM_STEPS="$ACCUM"

echo "Launching with GPUs: $GPU_IDS (count=$NUM_GPUS) | grad_accum_steps=$ACCUM"

torchrun --standalone --nproc_per_node=$NUM_GPUS lite_gpt.py
