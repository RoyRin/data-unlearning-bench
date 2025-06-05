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
python data-unlearning/run.py --c ascent_descent_example.yml
```

## Usage Guide 🚀

This benchmark provides a flexible way to run data unlearning experiments. The main workflow is:

1.  **Configure**: Define hyperparameter search spaces to generate configuration files for your experiments.
2.  **Launch**: Generate a script to run these experiments, potentially in parallel across multiple GPUs.
3.  **Execute**: Run the generated script to start the unlearning process and evaluation.

### 1. Configure Your Experiments

Experiments are defined by YAML files in the `config/` directory. You can generate these files by defining hyperparameter search spaces in `data-unlearning/config.py`.

First, open `data-unlearning/config.py` and modify the parameter dictionaries (e.g., `cifar_params`, `living_params`, `ascent_configs`) to suit your needs.

Then, generate the configuration files:
```bash
python data-unlearning/config.py
```
This will populate the `config/` directory with YAML files, each representing a unique experiment configuration. For example:
`config/unlearning_method-ascent_forget_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-1_lr-1e-05_model-resnet9_optimizer-sgd_N-100_batch_size-64.yml`

### 2. Launch Multiple Experiments

To run all your configured experiments, especially across multiple GPUs, use `data-unlearning/launching.py` to generate a shell script.

For example, to distribute jobs across 4 GPUs, running 2 jobs per GPU, and filtering for `ascent_forget` experiments on `cifar10`:

```bash
python data-unlearning/launching.py --gpus 0,1,2,3 --jobs-per-gpu 2 --filters ascent_forget,cifar10
```
This will create a `launch_jobs.sh` script.

Now, simply run the script:
```bash
bash launch_jobs.sh
```
This will start executing the experiments. The results (KLOM scores) will be saved as `.pt` files in the `data/eval/<dataset>/<model>/` directory.

### 3. Run a Single Experiment

If you want to run a single experiment, you can directly use `data-unlearning/run.py` with a specific configuration file.

```bash
python data-unlearning/run.py --c config/your_config_file.yml
```

For instance:
```bash
python data-unlearning/run.py --c "config/unlearning_method-ascent_forget_dataset-cifar10_epochs-[1,3,5,7,10]_forget_id-1_lr-1e-05_model-resnet9_optimizer-sgd_N-100_batch_size-64.yml"
```
<<<<<<< HEAD
unlearning_algos/
├── base_nn.py - holds dictionary that maps names to function calls
├── dm_direct.py - datamodel direct algo
├── dummies.py - `do-nothing` and `load-an-oracle` function calls
├── grad_ascent.py - `gradient ascent`
├── oracle_matching.py - `oracle matching` algorithm
└── utils.py
```

Code to run the different unlearning algorithms, and evaluate them over a range of hyperparameter values (e.g. `python ga_gd_eval.py 3`)
```
eval/
├── baselines_eval.py
├── dm_eval.py
├── ga_gd_eval.py
├── nn_evals.py
├── om_eval.py
└── scrub_eval.py
```

## Evaluation code - KLOM
```
auditors/
├── __init__.py
├── accuracies.py
├── basic.py
├── direct.py
├── eval_suite.py
├── logit_plots.py
├── margin_distributions_over_time.py
└── utils.py
```
* `eval_suite.py` - this is the main evaluation file. It is the lynchpin that calls all the other functions (it is called from evals like `dm_eval.py`). It runs the unlearning algorithm `N` times, and then compares the margins of the unlearned algorithm to the oracle's margins.

* `margin_distributions_over_time.py` - this is used to measure how well an individual data point is unlearned, over training time (this is akin to KLOM-over-time). In practice, rather than KLOM, it's measuring how well the margin of a data point    



# Implementing ULIRA
ULIRA was introduced by prior work that does not release open-source ULIRA code. We re-implemented our own version of ULIRA, and verified correctness by reproducingh the main results from the original ULIRA paper, and consulting with the original authors on our implementation to ensure correctness in our implementation results.

For the benefit of future researchers, we release our version of ULIRA. We view that it is more valuable to share something for future researchers, as implementing ULIRA is both subtle, difficult, and important. However, as this is not our work, we do not spend as long setting up a code cleaniliness and clarity.

Please reach out if you have issues reproducing results.

* First run `precomputing/make_ulira_forget_masks.ipynb` in order to create a set of forget-masks for all the ULIRA masks and unlearnings.
* Train oracle models for ulira using `unlearning/auditors/ulira/ulira.py`
* Train unlearnings for ulira using `unlearning/auditors/ulira/ulira`
* In order to train original models for ULIRA `precomputing/ulira_oracles.slrm`
    which calls `precomputing/compute_ulira_logits.py`
* In order to do unlearnings for ULIRA run `run_ulira_unlearnings.py`
* In order to compute ULIRA scores, we call the notebook: `compute_ulira_scores.ipynb` or `compute_ulira_scores.py`
    * We note that this file is very long and does not folow best coding practices, however, despite this, it is the main program containing ULIRA logic and we make it public to provide a reference for others implement ULIRA in case that is helpful.


```
├── ulira
│   ├── best_params.py
│   ├── compute_ulira_margins.py
│   ├── compute_ulira_scores.py
│   ├── compute_ulira_scores.slrm
│   ├── run_ulira_unlearnings.py
│   ├── run_ulira_unlearnings.slrm
│   ├── ulira.py
│   ├── ulira_pipeline_readme.md
│   └── ulira_plans.py
```

## Helper functions

```
training/
├── __init__.py
├── train.py - holds code for training both base models and oracle models (that exclude forget set), also code for saving and storing margins (for ease of KLOM computation)
models/
└── resnet9.py 
```


# Questions and Contributing
For contributions, feel free to create a pull-request. And for questions, feel invited to reach out to the corresponding authors on the paper.

Note, there are many assets associated this paper (original models, oracles, and datamodels), pelase reach out if there are difficulties reproducing them



# Running Living17 benchmarks from scratch

You will need to install FFCV (https://ffcv.io/)

To do so 
```
conda create -n ffcv python=3.9 cupy pkg-config libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.6 numba -c conda-forge -c pytorch && conda activate ffcv && conda update ffmpeg && pip install ffcv
                
```
extra imports required:
```
    pyyaml
    requests
    fastarg
    torchmetrics  
```
=======
*(Note: Wrapping the filename in quotes can help your shell handle special characters like `[` and `]`.)*
>>>>>>> MUGEN_workshop
