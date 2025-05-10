#!/usr/bin/env bash
set -e
(
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-6_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-8_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-9_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-6_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-7_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-8_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-9_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-6_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-7_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-9_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-7_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-8_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=1 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-9_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
) &
(
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-7_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-8_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-9_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-6_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-7_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-8_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-6_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-8_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-9_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-6_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-7_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-8_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-9_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
) &
(
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-6_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-7_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-8_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-9_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-6_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-7_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-9_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-7_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-8_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-9_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-6_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-7_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-8_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
) &
(
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-6_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-7_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-8_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-6_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-8_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-9_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-6_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-7_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-8_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-9_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-6_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-7_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=5 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-9_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
) &
(
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-2_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-6_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-7_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-9_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-7_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-8_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-3_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-9_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-1_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-4_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-6_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-7_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-32_forget_id-8_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-1_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-2_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-4_lr-0.005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-6_lr-5e-05_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-8_lr-0.0005_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
CUDA_VISIBLE_DEVICES=6 python run.py --c N-100_ascent_epochs-5_batch_size-64_dataset-cifar10_epochs-[5,7,10]_forget_batch_size-64_forget_id-9_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-scrub.yml &
wait
) &
wait
