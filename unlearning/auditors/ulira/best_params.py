best_params_GD = {
    "method_name": "benchmark_GD_wrapper",
    "unlearning_algo": "benchmark_GD_wrapper",
    "num_epochs": 3,
    "learning_rate": 1e-4,
    "forget_batch_size": 64,
}

best_params_GA = {
    "method_name": "benchmark_GA_wrapper",
    "unlearning_algo": "benchmark_GA_wrapper",
    "num_epochs": 3,
    #"learning_rate": 1e-6,
    "learning_rate": 5e-8,
    "batch_size": 64,
    "forget_batch_size": 64,
}
best_params_SCRUB = {
    "method_name": "scrub",
    "unlearning_algo": "scrub",
    "num_epochs": 10,
    "learning_rate": 0.01,
    "forget_batch_size": 32,
    "beta": 0.999,
    "retain_batch_size": 64,
    #"forget_batch_size": 32,
    "maximization_epochs": 3
}

best_params_SCRUB__large = {
    "method_name": "scrub",
    "unlearning_algo": "scrub",
    "num_epochs": 5,
    #"learning_rate": 0.01,
    "learning_rate": 0.01,
    "forget_batch_size": 32,
    "beta": 0.999,
    "retain_batch_size": 64,
    #"forget_batch_size": 32,
    "maximization_epochs": 3
}

best_params_SCRUB_resnet18 = {
    "method_name": "scrub",
    "unlearning_algo": "scrub",
    "num_epochs": 10,
    "learning_rate": 0.01,
    "forget_batch_size": 32,
    "beta": 0.999,
    "retain_batch_size": 64,
    #"forget_batch_size": 32,
    "maximization_epochs": 3,
    "model_name": "resnet18"
}
