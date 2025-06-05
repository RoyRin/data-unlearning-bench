# Data Unlearning Benchmark

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow?style=for-the-badge)](https://huggingface.co/datasets/royrin/KLOM-models/tree/main) [![Licence](https://img.shields.io/badge/MIT_License-lightgreen?style=for-the-badge)](./LICENSE)

## Installation 📦
```bash
git clone git@github.com:RoyRin/data-unlearning-bench.git
cd data-unlearning-bench
pip install -e .
```

## Quickstart ⚡️⚡️
```bash
python unlearning_bench/run.py --c ascent_descent_example.yml
```

## Usage Guide 🚀

This benchmark provides a flexible way to run data unlearning experiments. The main workflow is:

1.  **Configure**: Define hyperparameter search spaces to generate configuration files for your experiments.
2.  **Launch**: Generate a script to run these experiments, potentially in parallel across multiple GPUs.
3.  **Execute**: Run the generated script to start the unlearning process and evaluation.

### 1. Configure Your Experiments

Experiments are defined by YAML files in the `config/` directory. You can generate these files by defining hyperparameter search spaces in `unlearning_bench/config.py`.

First, open `unlearning_bench/config.py` and modify the parameter dictionaries (e.g., `cifar_params`, `living_params`, `ascent_configs`) to suit your needs.

Then, generate the configuration files:
```bash
python unlearning_bench/config.py
```
This will populate the `config/` directory with YAML files, each representing a unique experiment configuration. For example:
`config/unlearning_method-ascent_forget_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-1_lr-1e-05_model-resnet9_optimizer-sgd_N-100_batch_size-64.yml`

### 2. Launch Multiple Experiments

To run all your configured experiments, especially across multiple GPUs, use `unlearning_bench/launching.py` to generate a shell script.

For example, to distribute jobs across 4 GPUs, running 2 jobs per GPU, and filtering for `ascent_forget` experiments on `cifar10`:

```bash
python unlearning_bench/launching.py --gpus 0,1,2,3 --jobs-per-gpu 2 --filters ascent_forget,cifar10
```
This will create a `launch_jobs.sh` script.

Now, simply run the script:
```bash
bash launch_jobs.sh
```
This will start executing the experiments. The results (KLOM scores) will be saved as `.pt` files in the `data/eval/<dataset>/<model>/` directory.

### 3. Run a Single Experiment

If you want to run a single experiment, you can directly use `unlearning_bench/run.py` with a specific configuration file.

```bash
python unlearning_bench/run.py --c config/your_config_file.yml
```

For instance:
```bash
python data-unlearning/run.py --c "config/unlearning_method-ascent_forget_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-1_lr-1e-05_model-resnet9_optimizer-sgd_N-100_batch_size-64.yml"

* (Note: Wrapping the filename in quotes can help your shell handle special characters like `[` and `]`.)*

