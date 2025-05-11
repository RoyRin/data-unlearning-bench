# Data Unlearning Benchmark

## How to Run - Quick start
The main evaluation of unlearning algortihms is orchestrated through `eval_suite.py`.

`eval_suite.py` does 3 things:
1. Initalizes the original models (N times), which it assumes are pre-trained, and place in `BASE_DIR`, defined in `unlearning.auditors.utils.py` (set as a global variable)
2. run unlearning algorithm for a specific forget set (both specified by the config file) and saves the models.
3. Loads the oracles (fully retrained models) for that forget set
4. For a set of data point (in the forget, validation, and retain sets), eval_suite computes the margins of the oracle-models and the retain-models.
5. Computes and stores the KLOM scores for each data point that one has margins for.


To evaluate an unlearning method:
```
from unlearning.auditors.eval_suite import eval_suite
config = {
    "results_dir": "/path/to/output_dir",
    "dataset": "cifar10",
    "forgot_set_id": 5,
    "unlearning_algo": "oracle_matching",
    "unlearning_algo_kwargs": {
         ...
    },
    "run_direct_eval": True,
    "use_submitit_for_direct_eval": True,
    "save_unlearned_margins": False,
    "N_models_for_direct": 100,
}
eval_suite(config_dict=config)
```

The `unlearning_algo` name is searched in dictionary `NAME_TO_ALGO`, defined inside the file `unlearning.unlearning_algos.base_nn.py`. 

`eval_suite.py` expects that `NAME_TO_ALGO` is a function that can be called with the following interface:

```
    unlearned_model = NAME_TO_ALGO[algo_name](
        model=model,
        train_dataloader=train_loader,
        forget_dataloader=forget_loader,
        forget_indices=forget_set_indices,
        **unlearning_kwargs,
    )
```


Unlearning Algorithms are located in `unlearning.unlearning_algos`, and are specified in `base_nn.py`. In our implementations, `eval_suite.py` is called from python files in `unlearning/evals` (e.g. `unlearning/evals/om_eval.py`)

SCRUB is reimplemented from `https://github.com/meghdadk/SCRUB/tree/main`, largely copying directly, but with minor edits made for compatibility reasons and to fix a batch-order issue we found.



## Outline of Codebase:
```
├── LICENSE
├── README.md
├── datamodels - code for generating datamodels 
├── example_configs
│   └── oracle_matching_example_config.yaml - example config
├── extra_installs.txt
├── poetry.lock
├── pyproject.toml
└── unlearning
    ├── __init__.py
    ├── auditors - code for evaluating unlearning methods 
    ├── datasets - code for managing datasets
    ├── eval - code for actually running unlearning methods, and their evaluations
    ├── models - code for model specification
    ├── training - code for training
    ├── unlearning_algos - code for our proposed algorithms
    ├── unlearning_benchmarks - code for benchmarks
    └── utils.py
```


## Forget Set Indices:
While by default we only include training code for CIFAR10, we include the forget set indices for CIFAR10, LIVING17, and QNLI in 
`forget_set_indices`.

They were originally created through `precomputing/forget_sets.ipynb`


## Precomputing original and oracle (retrained) models

Original models and oracle (fully retrained) models are computed through `precomputing/oracles_and_full_models.ipynb`. We train on Resnet9 and Reset18.


## Algorithms
```
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
pyyaml
requests
fastarg
torchmetrics 