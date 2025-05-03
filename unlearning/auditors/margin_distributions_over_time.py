import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import shutil
import yaml
import numpy as np
import torch as ch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import logging
import pprint
from contextlib import redirect_stdout, redirect_stderr

from unlearning.auditors.utils import (
    model_factory,
    loader_factory,
    load_forget_set_indices,
    get_full_model_paths,
    get_oracle_paths,
    make_results_dir,
)
from unlearning.auditors.accuracies import eval_accuracy
from unlearning.auditors.logit_plots import compute_logits, plot_logits
from unlearning.auditors.basic import plot_margins

from unlearning.datasets import DATASET_SIZES, DATASET_VAL_SIZES
from unlearning.unlearning_algos.base_nn import NAME_TO_ALGO
from unlearning.unlearning_algos.utils import get_margins
from unlearning.auditors.utils import load_model_from_dict

from unlearning import BASE_DIR, LOG_DIR, ULIRA_BASE_DIR, dm_scores_path, ORACLE_BASE_DIR


def read_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(path, model_factory, ds_name):
    model = model_factory(ds_name)
    loaded_model = ch.load(path)
    if ds_name == 'QNLI':
        model.model.load_state_dict(loaded_model)
        return model

    first_key = list(loaded_model.keys())[0]
    print('> first_key:', first_key)
    if "model" in first_key:
        model.load_state_dict(loaded_model)
    else:
        # add ".model" to each key in k,vs
        loaded_model = {f"model.{k}": v for k, v in loaded_model.items()}
        model.load_state_dict(loaded_model)
    return model


def record_u_margins_over_time(model,
                               ckpt_path,
                               unlearn_fn,
                               unlearning_kwargs,
                               train_loader,
                               eval_loader,
                               forget_set_indices,
                               callback_save_dir=None,
                               callback_index=0,
                               callback_epochs=[-1]):

    if len(forget_set_indices) < 5:
        raise Exception(
            f"You probably forgot to set the forget_set_indices. - {forget_set_indices}"
        )

    load_model_from_dict(model, ckpt_path)
    model = model.cuda().eval()
    print(f"unlearning_kwargs - {unlearning_kwargs}")

    def callback(model, epoch):
        # compute margins on self,
        margins = get_margins(model, eval_loader)
        callback_savepath = callback_save_dir / f"{callback_index}__{epoch}__margins.npy"
        # save margins to path
        # save it
        np.save(callback_savepath, margins.cpu().numpy())

    # pass in a call back into unlearn_fn, as well as what to do every x steps
    unlearned_model = unlearn_fn(
        model=model,
        train_dataloader=train_loader,
        forget_dataloader=None,
        forget_indices=forget_set_indices,
        # val_loader = eval_loader,
        callback=callback,
        callback_epochs=callback_epochs,
        **unlearning_kwargs)
    return get_margins(unlearned_model, eval_loader), unlearned_model


def margins_over_time(config_yaml_file=None,
                      config_dict=None,
                      overwrite=False,
                      unlearn_fn=None):
    ####### SETUP ########
    if config_yaml_file is not None:
        config = read_yaml(config_yaml_file)
    elif config_dict is not None:
        config = config_dict
    else:
        raise ValueError("Must pass in either a yaml file or a dictionary")

    # which margins to record
    callback_epochs = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20
    ]

    results = {}
    results["params"] = {}

    pprint.pp(config)
    RESULTS_DIR = make_results_dir(config)
    # save config to results dir
    with open(RESULTS_DIR / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Overwrite check (checks only for `direct`, but we can change this if we want)
    direct_results_file = RESULTS_DIR / "direct" / "9__1__margins.npy"
    if not overwrite and direct_results_file.exists():
        raise FileExistsError(
            f"{direct_results_file} already exists. Set overwrite=True to overwrite."
        )

    logger = logging.getLogger("EvalSuite")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.CRITICAL,  # ignore submitit warnings
        handlers=[
            logging.FileHandler(RESULTS_DIR / "eval_suite.log"),
            logging.StreamHandler(),
        ],
    )
    logger.setLevel(logging.INFO)
    logger.info("Setting up eval suite..")
    rng = np.random.RandomState(0)

    ds_name = config["dataset"]

    # for now, let's tie the model to the dataset, so we have fewer moving pieces
    print(f"model!")
    model = model_factory(ds_name)  # on cuda, in eval mode
    logger.info(f"Loaded model.")

    forget_set_indices = load_forget_set_indices(ds_name,
                                                 config["forget_set_id"])

    results["params"]["forget_set_indices"] = forget_set_indices
    if unlearn_fn is None:
        unlearn_fn = NAME_TO_ALGO[config["unlearning_algo"]]

    unlearning_kwargs = config["unlearning_algo_kwargs"]
    if unlearning_kwargs is None:
        unlearning_kwargs = {}
    logger.info(f"Loaded unlearning algo: {config['unlearning_algo']}")

    method_name = config["unlearning_algo"]
    drop_last = False
    with redirect_stdout(open("/dev/null", "w")):

        # no shuffling, no augmentation
        if ds_name == 'QNLI':
            train_indices = np.arange(10_000)
        elif ds_name == 'LIVING17':
            train_indices = np.arange(44_200)
        else:
            train_indices = np.arange(50_000)

        train_loader = loader_factory(ds_name,
                                      indices=train_indices,
                                      indexed=True,
                                      drop_last=drop_last)

        val_loader = loader_factory(ds_name, split="val", indexed=True)
        forget_loader = loader_factory(
            ds_name,
            indices=forget_set_indices,
            batch_size=50,
            indexed=True,
            drop_last=drop_last,
        )

        eval_set_inds = np.arange(DATASET_SIZES[ds_name] +
                                  DATASET_VAL_SIZES[ds_name])
        eval_loader = loader_factory(ds_name,
                                     split="train_and_val",
                                     indices=eval_set_inds,
                                     indexed=True)

    logger.info(f"Created loaders.")
    ####### END OF SETUP ########

    ####### LOAD PRETRAINED MODELS ########
    splits = ["train", "val"]

    f_ckpt_paths, f_logit_paths, f_margins_paths = get_full_model_paths(
        ds_name, splits=splits)

    (
        o_ckpt_0_path,  # we only need a single oracle checkpoint
        o_logit_paths,
        o_margins_paths,
    ) = get_oracle_paths(ds_name, config["forget_set_id"], splits=splits)
    logger.info(f"Loaded paths of pretrained models.")

    #print('f_ckpt_path[0]:', f_ckpt_paths[0])
    model = load_model(f_ckpt_paths[0], model_factory, ds_name)

    _inds_to_plot = rng.choice(np.arange(len(forget_set_indices)), 20)
    inds_to_plot = forget_set_indices[_inds_to_plot]

    logger.info(f"Loaded a pretrained model.")

    DIRECT_DIR = RESULTS_DIR / "direct"
    DIRECT_DIR.mkdir(parents=True, exist_ok=True)
    # run the expensive direct eval
    # this involves running the unlearning algo many times
    # (once on each checkpoint from full_model_ckpt_paths)
    all_unlearned_margins = []

    f_ckpt_paths = f_ckpt_paths[:config["N_models_for_direct"]]

    print(f"\n\n\nwe are saving things to {DIRECT_DIR}\n\n\n")

    for index, f_ckpt_path in tqdm(enumerate(f_ckpt_paths),
                                   desc="unlearning models.."):
        print(f"margins over time: {f_ckpt_path}")
        _m, __unlearned_model = record_u_margins_over_time(
            model,
            f_ckpt_path,
            unlearn_fn,
            unlearning_kwargs,
            train_loader,
            eval_loader,
            forget_set_indices,
            callback_save_dir=DIRECT_DIR,
            callback_index=index,
            callback_epochs=callback_epochs)
        all_unlearned_margins.append(_m)


dataset = "CIFAR10"

ORACLE_BASE_DIR = ORACLE_BASE_DIR / dataset

best_params_SCRUB = {
    "method_name": "scrub",
    "unlearning_algo": "scrub",
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "forget_batch_size": 32,
    "beta": 0.999,
    "retain_batch_size": 64,
    #"forget_batch_size": 32,
    "maximization_epochs": 3
}

best_params_OM = {
    "dataset": "CIFAR10",
    "loss_type": "MSE",
    "num_epochs": 10,
    "learning_rate": 1e-3,  # 1e-4,
    "batch_size": 512,
    "optimizer": "adam",
    "retain_multiplier": 5.,
    "forget_multiplier": 1.,
    "num_oracles": 1,
    "wd_lambda": 0.,
    "shuffle": True,
}
best_params_DM = {
    "dataset": "CIFAR10",
    "loss_type": "MSE",
    "num_epochs": 10,
    "learning_rate": 1e-3,  # 1e-4,
    "batch_size": 512,
    "optimizer": "adam",
    "retain_multiplier": 5.,
    "forget_multiplier": 1.,
    "num_oracles": 1,
    "wd_lambda": 0.,
    "shuffle": True,
}

dm_matching_config = {
    "dataset": "CIFAR10",
    #"oracles_path": str(ORACLE_BASE_DIR/ f"forget_set_{forget_set_id}"), # dm_scores_path,
    "loss_type": "MSE",
    "num_epochs": 10,  # 4
    "learning_rate": 1e-4,
    "batch_size": 512,
    "optimizer": "adam",
    "retain_multiplier": 20.,
    "forget_multiplier": 5.,
    "num_oracles": 1,
    "wd_lambda": 0.,
    "shuffle": True,
    "dm_scores_path": dm_scores_path,
    "multiplier": 1.,
}

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
    "learning_rate": 1e-6,
    "batch_size": 64,
    "forget_batch_size": 64,
}

if __name__ == "__main__":
    import sys
    learning_rates_overrides = [
        None, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5
    ]

    methods = [
        "benchmark_GD_wrapper", "benchmark_GA_wrapper", "scrub",
        "oracle_matching", "dm_matching"
    ]
    forget_set_ids = [5, 6]

    from itertools import product

    tups = list(product(methods, learning_rates_overrides, forget_set_ids))
    for ii, tup in enumerate(tups):
        print(f"{ii} - {tup}")

    index = 0
    try:
        index = int(sys.argv[1])
    except:
        print(f"Using default index {index}")

    #method = methods[index]
    tup = tups[index]
    method, learning_rate_override, forget_set_id = tup

    N_models_for_direct = 10
    num_epochs_override = 10  # manual
    CWD = Path.cwd()
    BASE_DIR = CWD.parent.parent

    config_file = BASE_DIR / "configs" / "test_oracle_matching.yaml"

    config_file.exists()

    config_dict = read_yaml(config_file)

    if method == "scrub":
        num_epochs = best_params_SCRUB["num_epochs"]
        learning_rate = best_params_SCRUB["learning_rate"]
        beta = best_params_SCRUB["beta"]
        retain_bs = best_params_SCRUB["retain_batch_size"]
        forget_bs = best_params_SCRUB["forget_batch_size"]
        max_epochs = best_params_SCRUB["maximization_epochs"]
        #if forget_set_id == 3:
        #    config_dict["unlearning_algo_kwargs"]["drop_last"] = False # True

        config_dict["unlearning_algo_kwargs"]["num_epochs"] = num_epochs
        config_dict["unlearning_algo_kwargs"]["learning_rate"] = learning_rate
        config_dict["unlearning_algo_kwargs"]["beta"] = beta
        config_dict["unlearning_algo_kwargs"]["retain_batch_size"] = retain_bs
        config_dict["unlearning_algo_kwargs"]["forget_batch_size"] = forget_bs
        config_dict["unlearning_algo_kwargs"][
            "maximization_epochs"] = max_epochs
    elif method == "oracle_matching":
        for k, v in best_params_OM.items():
            config_dict["unlearning_algo_kwargs"][k] = v
    elif method == "dm_matching":
        for k, v in dm_matching_config.items():
            config_dict["unlearning_algo_kwargs"][k] = v
    elif method == "benchmark_GD_wrapper":
        for k, v in best_params_GD.items():
            config_dict["unlearning_algo_kwargs"][k] = v
            config_dict["unlearning_algo_kwargs"][
                "num_epochs"] = num_epochs_override

    elif method == "benchmark_GA_wrapper":
        for k, v in best_params_GA.items():
            config_dict["unlearning_algo_kwargs"][k] = v
            config_dict["unlearning_algo_kwargs"][
                "num_epochs"] = num_epochs_override

    else:
        raise ValueError(f"method {method} not recognized")

    config_dict["unlearning_algo"] = method
    config_dict["dataset"] = dataset
    config_dict["unlearning_algo_kwargs"]["dataset"] = dataset

    #config_dict["results_dir"] = "./scrub_results/"
    config_dict["forget_set_id"] = forget_set_id
    config_dict["unlearning_algo_kwargs"]["oracles_path"] = str(
        ORACLE_BASE_DIR / f"forget_set_{forget_set_id}")

    config_dict["unlearning_algo_kwargs"]["forget_set_id"] = forget_set_id

    config_dict["run_direct_eval"] = True

    config_dict["unlearning_algo_kwargs"][
        "N_models_for_direct"] = N_models_for_direct
    config_dict["N_models_for_direct"] = N_models_for_direct

    config_dict["results_dir"] = str(RESULTS_DIR)

    #
    if learning_rate_override is not None:
        config_dict["unlearning_algo_kwargs"][
            "learning_rate"] = learning_rate_override
    pprint.pp(config_dict)

    margins_over_time(config_yaml_file=None,
                      config_dict=config_dict,
                      overwrite=False,
                      unlearn_fn=None)
