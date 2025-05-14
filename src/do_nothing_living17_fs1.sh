#!/usr/bin/env bash
set -e
(
CUDA_VISIBLE_DEVICES=0 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1]_forget_id-1_lr-10000.0_model-resnet18_optimizer-sgd_unlearning_method-do_nothing.yml &
wait
) &
wait
