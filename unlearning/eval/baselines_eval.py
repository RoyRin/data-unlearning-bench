import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from unlearning.auditors.utils import (
    load_forget_set_indices, )
from unlearning.unlearning_algos.base_nn import NAME_TO_ALGO
from importlib import reload
from scipy import stats


def read_yaml(f):
    with open(f, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


from unlearning.auditors import eval_suite
import sys
from itertools import product

DATASET = "CIFAR10"
#DATASET = "LIVING17"
#DATASET = "QNLI"

N_models_for_direct = 100

held_out_start = 100
held_out_end = 200

from unlearning import BASE_DIR, LOG_DIR, ULIRA_BASE_DIR, ORACLE_BASE_DIR, RESULTS_DIR


def do_baselines():
    import sys
    from itertools import product
    index = int(sys.argv[1])

    try:
        dataset = sys.argv[2]
    except:
        dataset = DATASET
        print(f"Using default dataset {dataset}")

    if dataset == "CIFAR10":
        forget_set_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif dataset == "LIVING17":
        forget_set_ids = [1, 2, 3]
    elif dataset == "QNLI":
        forget_set_ids = [1, 2, 3, 4]

    else:
        raise ValueError("Unknown dataset")

    #num_epochs_groups = [5, 10, 15, 20]
    method_names = ["do_nothing", "load_an_oracle"]

    #method_names = ["do_nothing"]
    #method_names = ["load_an_oracle"]
    num_epochs = 10

    tups = list(product(forget_set_ids, method_names))
    print(f"tups are :{tups}")
    for i, tup in enumerate(tups):
        print(f"{i} - {tup}")
    tup = tups[index]
    forget_set_id, method_name = tup

    CWD = Path.cwd()
    BASE_DIR = CWD.parent.parent

    config_file = BASE_DIR / "configs" / "test_oracle_matching.yaml"

    #config_file.exists()
    ###o
    config_dict = read_yaml(config_file)
    config_dict["dataset"] = dataset
    config_dict["unlearning_algo_kwargs"]["dataset"] = dataset

    #config_dict["results_dir"] = "./scrub_results/"
    config_dict["forget_set_id"] = forget_set_id

    if method_name == "load_an_oracle":

        school2_ORACLE_DIR = Path(
            f"school2_path/unlearning/precomputed_models/oracles/{dataset}/")
        MIT_ORACLE_DIR = Path(f"school1_path/MATCHING/oracles/{dataset}/")
        if school2_ORACLE_DIR.exists():
            ORACLE_BASE_DIR = school2_ORACLE_DIR
        elif MIT_ORACLE_DIR.exists():
            ORACLE_BASE_DIR = MIT_ORACLE_DIR
        else:
            raise ValueError("No oracle directory found")

        config_dict["unlearning_algo_kwargs"]["oracles_path"] = str(
            ORACLE_BASE_DIR / f"forget_set_{forget_set_id}")

    config_dict["unlearning_algo_kwargs"]["forget_set_id"] = forget_set_id

    config_dict["run_direct_eval"] = True
    config_dict["unlearning_algo_kwargs"]["num_epochs"] = num_epochs
    config_dict["unlearning_algo"] = method_name
    config_dict["N_models_for_direct"] = N_models_for_direct
    config_dict["unlearning_algo_kwargs"][
        "N_models_for_direct"] = N_models_for_direct
    config_dict["unlearning_algo_kwargs"]["held_out_start"] = held_out_start
    config_dict["unlearning_algo_kwargs"]["held_out_end"] = held_out_end

    #config_dict[]

    config_dict["results_dir"] = str(RESULTS_DIR)

    print(f"running config_dict ")
    for k, v in config_dict.items():
        print(f"{k} - {v}")

    results_dir, results = eval_suite.eval_suite(config_dict=config_dict)


if __name__ == "__main__":
    do_baselines()
