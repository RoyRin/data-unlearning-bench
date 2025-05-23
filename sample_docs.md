# Anonymous Code Submission: Unlearning Evaluation Framework

This codebase provides a framework for evaluating machine unlearning methods. It focuses on computing the KL divergence between the output distributions (specifically, classification margins) of an "unlearned" model and an "oracle" model (a model retrained from scratch without the data to be forgotten).

## Core Functionality

The primary script, `run.py`, orchestrates the unlearning and evaluation process. It performs the following key steps:

1.  **Loads Configuration**: Reads a YAML configuration file specifying the dataset, model, unlearning method, hyperparameters, and other parameters.
2.  **Loads Data**:
    - Retrieves pre-trained model checkpoints.
    - Fetches "oracle" margins (pre-computed margins from models trained without the forget set).
    - Loads indices of data points to be "forgotten."
    - These files are expected to be in specific directories (see **Directory Structure**). If not found locally, the framework will attempt to download them automatically from a designated Hugging Face repository.
3.  **Performs Unlearning**:
    - If unlearned model margins are not already pre-computed and saved, it applies the specified unlearning method (e.g., gradient ascent, scrub, scrubnew) to the pre-trained models.
    - Computes and saves the margins for the unlearned models across specified epochs.
4.  **Evaluates Unlearning**:
    - Calculates the KL divergence between the oracle margins and the unlearned model margins. This metric is referred to as "KLOM" (KL on Margins).
    - Saves these KLOM results.

## File Overview

- `run.py`: Main script to execute unlearning experiments and evaluations.
- `config.py`: Handles loading, checking, and generating YAML configuration files for experiments.
- `unlearning.py`: Implements various unlearning algorithms (`do_nothing`, `ascent_forget`, `scrub`, `scrubnew`) and defines optimizers.
- `eval.py`: Contains functions for computing classification margins and KL divergence between margin distributions.
- `datasets.py`: Provides data loaders for supported datasets (`cifar10`, `living17`).
- `models.py`: Defines model architectures (`ResNet9`, `ResNet18`).
- `paths.py`: Defines directory paths for data, configurations, checkpoints, and results. It also includes functionality to download necessary files from a remote registry if not found locally.
- `launching.py`: Generates a bash script to run multiple experiments (defined by YAML configuration files) in parallel across specified GPUs. It takes GPU assignments and job distribution parameters as input.

## Setup

### Dependencies

The project uses standard Python libraries. Install them using pip:

```bash
pip install -r requirements.txt
```

A `requirements.txt` file is provided, listing the necessary packages and their suggested versions (e.g., PyTorch, NumPy, torchvision, etc.).

### Data and Pre-trained Models

The framework relies on several pre-computed files:

- **Datasets**: CIFAR-10 (downloaded automatically by torchvision) and Living-17 (expected as `raw_tensors_tr_new.pt` and `raw_tensors_val_new.pt` in `data/living17/`).
- **Pre-trained Model Checkpoints**: Stored under `data/checkpoints/<dataset>/<model>/pretrain_checkpoints.pt`.
- **Oracle Margins**: Stored under `data/oracles/<dataset>/<model>/oracle_margins_<forget_id>.pt`.
- **Forget Set Indices**: Stored under `data/forget_set_indices/<dataset>/forget_indices_<forget_id>.pt`.

If these files are not present locally in the specified paths, the script will attempt to download them automatically from a designated Hugging Face repository. You will need to ensure you have an internet connection for this fallback mechanism to work.

### Directory Structure

The expected directory structure (relative to the repository root) is:

```
.
├── config/             # YAML configuration files
├── data/
│   ├── checkpoints/
│   │   └── <dataset>/
│   │       └── <model>/
│   │           └── pretrain_checkpoints.pt
│   ├── eval/
│   │   └── <dataset>/
│   │       └── <model>/
│   │           └── <config_specific_name_klom>.pt
│   ├── forget_set_indices/
│   │   └── <dataset>/
│   │       └── forget_indices_<forget_id>.pt
│   ├── margins/
│   │   └── <dataset>/
│   │       └── <model>/
│   │           └── <config_specific_name_margins>.pt
│   ├── oracles/
│   │   └── <dataset>/
│   │       └── <model>/
│   │           └── oracle_margins_<forget_id>.pt
│   └── living17/       # For Living-17 dataset
│       ├── raw_tensors_tr_new.pt
│       └── raw_tensors_val_new.pt
├── datasets.py
├── eval.py
├── config.py
├── launching.py
├── models.py
├── paths.py
├── run.py
└── unlearning.py
```

## Usage

The recommended experimental pipeline involves defining hyperparameter search spaces in `config.py`, generating a runnable bash script using `launching.py`, and finally executing that script.

### 1. Define Hyperparameter Search Space (in `config.py`)

The `config.py` script is the starting point for defining your experiments. It allows you to specify lists of hyperparameters that you want to explore.

- Open `config.py`.
- Modify the dictionaries (e.g., `cifar_params`, `living_params`, `ascent_configs`, `scrub_configs`) to define the ranges or specific values for parameters like learning rate, epochs, model architecture, dataset, etc.
- When you run `python config.py`, it will generate a set of individual YAML configuration files in the `config/` directory. Each file represents a unique combination of the hyperparameters you defined.

Example `config/` file naming convention:
`unlearning_method-ascent_forget_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-1_lr-1e-05_model-resnet9_optimizer-sgd_N-100_batch_size-64.yml`

This systematic generation ensures that all desired experimental conditions are covered.

### 2. Generate Experiment Execution Script (with `launching.py`)

Once your configuration files are generated by `config.py`, you use `launching.py` to create a bash script that will run all these experiments, potentially in parallel across multiple GPUs.

To generate the execution script (e.g., `launch_jobs.sh`):

```bash
python launching.py --gpus <gpu_ids_comma_separated> --jobs-per-gpu <num_jobs> --output <your_script_name.sh> --filters <comma_separated_filters_for_config_names>
```

**Arguments for `launching.py`:**

- `--gpus`: Comma-separated list of GPU IDs to use (e.g., `0,1,2,3`).
- `--jobs-per-gpu`: (Optional, default: 1) Number of concurrent jobs to run on each specified GPU.
- `--output`: (Optional, default: `launch_jobs.sh`) Name of the output bash script that will be generated.
- `--filters`: (Optional) Comma-separated list of strings. Only config files whose names contain _all_ these filter strings will be included in the launch script. This is useful for running a subset of your generated configs. For example, `--filters ascent_forget,cifar10` would only include configs for `ascent_forget` on `cifar10`.

This script will intelligently distribute the experiment runs (defined by the `.yml` files in `config/`) across the specified GPUs.

### 3. Launch Experiments

After `launching.py` has created your bash script (e.g., `launch_jobs.sh`), you can execute it to start all your experiments:

```bash
bash <your_script_name.sh>
```

Or, if you used the default output name:

```bash
bash launch_jobs.sh
```

The script will then sequentially or concurrently (based on your `--jobs-per-gpu` setting and number of GPUs) execute `run.py` for each relevant configuration file.

Each `run.py` instance will:

- Load its specific configuration.
- Attempt to load necessary data (checkpoints, oracle margins, forget indices), downloading if configured and not found locally.
- Perform unlearning (or load pre-computed unlearning margins).
- Compute and save the KLOM results to the `data/eval/<dataset>/<model>/` directory. The filename will be based on the configuration.

### Individual Run (Alternative to `launching.py`)

If you wish to run a single experiment without using the `launching.py` script, you can still do so directly with `run.py` after ensuring the desired configuration file exists in `config/`:

```bash
python run.py --c config/<your_config_file_name>.yml
```

For example:

```bash
python run.py --c config/unlearning_method-ascent_forget_dataset-cifar10_epochs-\[1,3,5,7,10\]_forget_id-1_lr-1e-05_model-resnet9_optimizer-sgd_N-100_batch_size-64.yml
```

_(Note: You might need to escape characters like `[` and `]` in the filename depending on your shell.)_

### Configuration Parameters

Key parameters in the YAML configuration files include:

- `unlearning_method`: (string) Name of the unlearning method (e.g., `ascent_forget`, `scrub`, `scrubnew`, `do_nothing`).
- `lr`: (float) Learning rate for the unlearning process.
- `epochs`: (list of int) List of epochs at which to save the unlearned model and compute margins.
- `dataset`: (string) Name of the dataset (e.g., `cifar10`, `living17`).
- `optimizer`: (string) Optimizer to use (currently only `sgd` is supported in `check_config`).
- `model`: (string) Model architecture (e.g., `resnet9`, `resnet18`).
- `N`: (int) Number of pre-trained models/oracle margins to use (influences the number of samples for KLOM calculation).
- `forget_id`: (int) Identifier for the forget set to be used.
- `batch_size`: (int) Batch size for data loaders.
- `ascent_epochs`: (int, required for `scrub` and `scrubnew`) Number of epochs to perform gradient ascent on the forget set.
- `forget_batch_size`: (int, optional, can be used by `scrub` and `scrubnew`) Specific batch size for the forget loader.

### Output

The main output is the KLOM (KL divergence on Margins) scores, saved as `.pt` files in the `data/eval/<dataset>/<model>/` directory. The filename is generated based on the experiment configuration (see `get_checkpoint_name` in `unlearning.py`).

Intermediate unlearned model margins are saved in `data/margins/<dataset>/<model>/`.

## Anonymity

- The previous mechanism using `paths.BASE_dbREQ_URL` for downloading data has been replaced with automatic fetching from a Hugging Face repository if files are not found locally.
- No user-specific paths or identifiers should be present in the core logic.

## Notes

- The `launching.py` script (as described in the **Usage** section) facilitates running multiple experiments, potentially in parallel on multiple GPUs, by generating a bash script based on the configurations in the `config/` directory.
- The code uses `torch.cuda` if available. Ensure your environment is set up correctly if you intend to use GPUs.
- The `living17` dataset requires pre-processed tensor files (`raw_tensors_tr_new.pt`, `raw_tensors_val_new.pt`) to be placed in the `data/living17/` directory.
- In `datasets.py`, the function `get_living17_dataloader` is defined twice. It is recommended to remove one of the definitions (e.g., the first one from lines 41-60).
