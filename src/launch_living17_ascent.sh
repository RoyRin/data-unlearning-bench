#!/usr/bin/env bash
set -e
(
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-0.0001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-0.05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-1e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-0.01_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-0.0001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-0.05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
) &
(
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-1e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-0.01_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-0.0001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-0.05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-1e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
) &
(
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-0.01_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-0.0001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-0.05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-1e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
) &
(
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-1_lr-0.01_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-0.0001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-2_lr-0.05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-0.0005_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-3_lr-1e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-0.001_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-4_lr-5e-05_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=7 python run.py --c N-100_batch_size-64_dataset-living17_epochs-[1,3,5,7,10]_forget_id-5_lr-0.01_model-resnet18_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
) &
wait
