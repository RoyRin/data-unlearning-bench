CUDA_VISIBLE_DEVICES=5 p run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-3_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml;
CUDA_VISIBLE_DEVICES=5 p run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-3_lr-0.01_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 p run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-3_lr-1e-05_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml

CUDA_VISIBLE_DEVICES=5 p run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-5_lr-0.001_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml;
CUDA_VISIBLE_DEVICES=5 p run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-5_lr-0.01_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml &
CUDA_VISIBLE_DEVICES=5 p run.py --c N-100_batch_size-64_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-5_lr-1e-05_model-resnet9_optimizer-sgd_unlearning_method-ascent_forget.yml
