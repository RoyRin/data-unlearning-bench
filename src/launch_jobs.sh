#!/usr/bin/env bash
set -e
(
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-1_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-1_lr-1e-05_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-2_lr-0.01_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-4_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-4_lr-1e-05_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-6_lr-0.01_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-7_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-7_lr-1e-05_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-8_lr-0.01_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-9_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-9_lr-1e-05_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1]_forget_id-2_lr-10000.0_model-resnet9_optimizer-sgd_unlearning_method-do_nothing.yml &
wait
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1]_forget_id-6_lr-10000.0_model-resnet9_optimizer-sgd_unlearning_method-do_nothing.yml &
CUDA_VISIBLE_DEVICES=2 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1]_forget_id-8_lr-10000.0_model-resnet9_optimizer-sgd_unlearning_method-do_nothing.yml &
wait
) &
(
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-1_lr-0.01_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-2_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-2_lr-1e-05_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-4_lr-0.01_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-6_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-6_lr-1e-05_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-7_lr-0.01_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-8_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-8_lr-1e-05_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
wait
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-9_lr-0.01_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1]_forget_id-1_lr-10000.0_model-resnet9_optimizer-sgd_unlearning_method-do_nothing.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1]_forget_id-4_lr-10000.0_model-resnet9_optimizer-sgd_unlearning_method-do_nothing.yml &
wait
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1]_forget_id-7_lr-10000.0_model-resnet9_optimizer-sgd_unlearning_method-do_nothing.yml &
CUDA_VISIBLE_DEVICES=4 python run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1]_forget_id-9_lr-10000.0_model-resnet9_optimizer-sgd_unlearning_method-do_nothing.yml &
wait
) &
wait
