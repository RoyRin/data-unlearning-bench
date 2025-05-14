#!/usr/bin/env bash
set -e
(
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-1e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-0.05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-0.0001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-0.01_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-1e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-0.05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1]_forget_id-1_lr-10000.0_model-resnet18_optimizer-sgd_unlearning_method-do_nothing.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1]_forget_id-4_lr-10000.0_model-resnet18_optimizer-sgd_unlearning_method-do_nothing.yml &
wait
) &
(
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-0.0001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-0.01_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-1e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-0.05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-0.0001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-0.01_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-1e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1]_forget_id-2_lr-10000.0_model-resnet18_optimizer-sgd_unlearning_method-do_nothing.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1]_forget_id-5_lr-10000.0_model-resnet18_optimizer-sgd_unlearning_method-do_nothing.yml &
wait
) &
(
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-3_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-32_forget_id-5_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-3_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.005_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-scrubnew.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-living17_epochs-[5,7,10]_forget_batch_size-64_forget_id-5_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-0.05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-0.0001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-0.01_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-1e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-0.05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-0.0001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-0.01_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1]_forget_id-3_lr-10000.0_model-resnet18_optimizer-sgd_unlearning_method-do_nothing.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c test_living17.yml &
wait
) &
wait
