#!/usr/bin/env bash
# Usage: ./launchlite.sh <gpu_ids> [grad_accum_steps] [joinedbin] [--save-checkpoint] [--checkpoint-folder <folder_name>] [--run-description <description>]
#   gpu_ids            Comma-separated list of GPU indices to use, e.g. "0,1,2,3"
#   grad_accum_steps   (optional) number of gradient-accumulation steps per optimizer step (default: 1)
#   joinedbin          (optional) path to single .bin file for one-pass training (default: none)
#   --save-checkpoint  (optional) enable saving model checkpoint at end of training
#   --checkpoint-folder (optional) folder name for saving checkpoints (e.g., "rand-1pct")
#   --run-description  (optional) descriptive name to include in checkpoint folder (e.g., "baseline_model")

set -euo pipefail

GPU_IDS=${1:-0}
ACCUM=${2:-1}
JOINEDBIN=${3:-}
SAVE_CHECKPOINT=false
CHECKPOINT_FOLDER=""
RUN_DESCRIPTION=""

# Parse arguments for flags
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --save-checkpoint)
            SAVE_CHECKPOINT=true
            ;;
        --checkpoint-folder)
            i=$((i + 1))
            if [ $i -le $# ]; then
                CHECKPOINT_FOLDER="${!i}"
            else
                echo "Error: --checkpoint-folder requires a folder name"
                exit 1
            fi
            ;;
        --run-description)
            i=$((i + 1))
            if [ $i -le $# ]; then
                RUN_DESCRIPTION="${!i}"
            else
                echo "Error: --run-description requires a description"
                exit 1
            fi
            ;;
    esac
    i=$((i + 1))
done

# make GPU list absolute (remove trailing/leading spaces)
GPU_IDS=$(echo "$GPU_IDS" | tr -d '[:space:]')

# derive GPU count
IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARR[@]}

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NUM_GPUS="$NUM_GPUS"
export GRAD_ACCUM_STEPS="$ACCUM"
export SAVE_CHECKPOINT="$SAVE_CHECKPOINT"
export CHECKPOINT_FOLDER="$CHECKPOINT_FOLDER"
export RUN_DESCRIPTION="$RUN_DESCRIPTION"

if [ -n "$JOINEDBIN" ]; then
    export JOINEDBIN="$JOINEDBIN"
    echo "Launching with GPUs: $GPU_IDS (count=$NUM_GPUS) | grad_accum_steps=$ACCUM | joinedbin=$JOINEDBIN | save_checkpoint=$SAVE_CHECKPOINT | checkpoint_folder=$CHECKPOINT_FOLDER | run_description=$RUN_DESCRIPTION"
else
    echo "Launching with GPUs: $GPU_IDS (count=$NUM_GPUS) | grad_accum_steps=$ACCUM | save_checkpoint=$SAVE_CHECKPOINT | checkpoint_folder=$CHECKPOINT_FOLDER | run_description=$RUN_DESCRIPTION"
fi

torchrun --standalone --nproc_per_node=$NUM_GPUS lite_gpt.py
