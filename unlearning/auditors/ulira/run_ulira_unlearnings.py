import re
import numpy as np
import torch as ch
from tqdm import tqdm
from pathlib import Path

from unlearning.auditors.direct import get_u_margins
from unlearning.auditors.utils import loader_factory
from unlearning.auditors.accuracies import eval_accuracy

import yaml
import numpy as np
import torch as ch
from tqdm import tqdm
from pathlib import Path
import logging
import pprint
from contextlib import redirect_stdout, redirect_stderr

from unlearning.auditors.utils import (
    model_factory,
    loader_factory,
    make_results_dir,
)
from unlearning.auditors.direct import (
    get_u_margins, )
from unlearning.auditors import ulira
from unlearning.unlearning_algos.base_nn import NAME_TO_ALGO

from contextlib import redirect_stdout, redirect_stderr

from unlearning.auditors.ulira_plans import get_ulira_forget_masks, load_ulira_forget_masks
import datetime

from unlearning import BASE_DIR, LOG_DIR, ULIRA_BASE_DIR, dm_scores_path


def read_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def write_yaml(yaml_file, config):
    with open(yaml_file, "w") as f:
        yaml.dump(config, f)


def load_model(path, model_factory, ds_name):
    model = model_factory(ds_name)
    loaded_model = ch.load(path)
    first_key = list(loaded_model.keys())[0]
    if "model" in first_key:
        model.load_state_dict(loaded_model)

    else:
        # add ".model" to each key in k,vs
        loaded_model = {f"model.{k}": v for k, v in loaded_model.items()}
        model.load_state_dict(loaded_model)
    return model


def sort_key(path):
    # Extract numerical parts and convert to integers
    numbers = re.findall(r'\d+', path.name)
    return [int(num) for num in numbers]


def do_several_ulira_unlearnings(start=0,
                                 num_unlearnings=25,
                                 config_yaml_file=None,
                                 config_dict=None,
                                 overwrite=False,
                                 models_dir=None,
                                 unlearn_fn=None,
                                 unlearning_forget_set_size=200,
                                 use_resnet18=True,
                                 unlearnings_per_model=40,
                                 trained_epochs=24,
                                 forget_set_id=None):
    """
    pass in either a yaml file or a dictionary

    expects that unlearn_fn has the interface:
        unlearned_model = unlearn_fn(model,
                                     train_dataloader,
                                     forget_dataloader,
                                     forget_indices,
                                     **kwargs)
    """
    ####### SETUP ########
    if config_yaml_file is not None:
        config = read_yaml(config_yaml_file)
    elif config_dict is not None:
        config = config_dict
    else:
        raise ValueError("Must pass in either a yaml file or a dictionary")

    results = {}
    results["params"] = {}

    pprint.pp(config)

    RESULTS_DIR = make_results_dir(config)

    ULIRA_RESULTS_DIR = RESULTS_DIR / "ulira"
    # make dir
    ULIRA_RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    print(f"saving to {ULIRA_RESULTS_DIR}")

    ulira_results_save_path = ULIRA_RESULTS_DIR / f"{num_unlearnings}__ulira_margins__{start}.pt"

    ulira_accuracies_save_path = ULIRA_RESULTS_DIR / f"{num_unlearnings}__ulira_margins__{start}__accuracy.yaml"

    if ulira_results_save_path.exists():
        print(f"already exists {ulira_results_save_path}")
        return

    with open(RESULTS_DIR / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Overwrite check (checks only for `direct`, but we can change this if we want)
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
    model = model_factory(ds_name)  # on cuda, in eval mode
    logger.info(f"Loaded model.")

    if unlearn_fn is None:
        unlearn_fn = NAME_TO_ALGO[config["unlearning_algo"]]

    unlearning_kwargs = config.get("unlearning_algo_kwargs", {})

    logger.info(f"Loaded unlearning algo: {config['unlearning_algo']}")

    with redirect_stdout(open("/dev/null", "w")):
        # no shuffling, no augmentation

        # DROPPING LAST SO THAT SCRUB WILL WORK!
        method_name = config["unlearning_algo"]
        drop_last = False

        train_loader = loader_factory(ds_name, indexed=True)
        #,drop_last=drop_last)

        val_loader = loader_factory(ds_name, split="val", indexed=True)
        #drop_last=drop_last

        eval_set_inds = np.arange(
            len(train_loader.dataset) + len(val_loader.dataset))
        eval_loader = loader_factory(ds_name,
                                     split="train_and_val",
                                     indices=eval_set_inds,
                                     indexed=True)
        #drop_last=drop_last)
    logger.info(f"Created loaders.")
    ####### END OF SETUP ########

    ####### LOAD PRETRAINED MODELS ########
    # f stands for "full model",-  the model that was trained on the enitre dataset (retain + forget)
    # o stands for "oracle model" - the model that was trained on the oracle dataset (retain only)

    splits = ["train", "val"]

    if models_dir is None:
        ulira_model_ckpt_paths, _, _ = ulira.get_ulira_model_paths(
            ds_name, splits=splits)
    else:
        ulira_model_ckpt_paths = sorted(list(
            models_dir.glob(f"sd_*_epoch_{trained_epochs-1}.pt")),
                                        key=sort_key)

    ulira_mask = ulira.get_ulira_masks()
    #ulira_forget_mask = ulira.get_ulira_forget_mask(ds_name)

    logger.info(f"Loaded paths of pretrained models.")

    # resnet 9!

    if not use_resnet18:
        print(f"loading resnet9!")
        model = load_model(ulira_model_ckpt_paths[0], model_factory, ds_name)
    else:
        print(f"laoding resnet18")
        from torchvision import models as torchvision_models
        import torch
        from unlearning.models.resnet9 import WrappedModel
        # resnet 18
        model = torchvision_models.resnet18(num_classes=10)
        #model = ResNet9(num_classes=10)
        model = WrappedModel(model).cuda()
        #model = model.load_state_dict(torch.load(ulira_model_ckpt_paths[0]))

    print(f"model is :{model}")
    logger.info(f"Loaded a pretrained model.")
    ####### END OF LOADING PRETRAINED MODELS ########

    ####### DIRECT EVAL ########

    DIRECT_DIR = RESULTS_DIR / "direct"
    DIRECT_DIR.mkdir(parents=True, exist_ok=True)
    # run the expensive direct eval
    # this involves running the unlearning algo many times
    # (once on each checkpoint from full_model_ckpt_paths)
    all_unlearned_margins = []

    # NOTE! we do 40 unlearnings per model! (this is hardcoded)
    # be aware!

    print(
        f"Note! we do {unlearnings_per_model} unlearnings per model! (this is hardcoded)"
    )
    if False:
        all_ulira_forget_masks = get_ulira_forget_masks(
            ds_name,
            original_model_count=256,
            class_5_range=1000,
            unlearnings_per_model=unlearnings_per_model,
            unlearning_forget_set_size=unlearning_forget_set_size,
            overwrite=False)
    all_ulira_forget_masks = load_ulira_forget_masks(
        256, unlearning_forget_set_size, forget_set_id)
    accuracies = []
    for i in tqdm(range(start, start + num_unlearnings),
                  desc="unlearning models.."):

        original_model_i = i // unlearnings_per_model
        forget_index = i % unlearnings_per_model

        forget_set_mask = all_ulira_forget_masks[original_model_i][
            forget_index]
        # forget_set should be the set intersection of the forget mask and the ulira mask at index i
        training_set_mask = ulira_mask[original_model_i]  # i]

        #forget_set_mask = ulira_forget_mask[i]
        print(f"forget_set_mask= {forget_set_mask.shape}")
        print(f"training_set_mask= {training_set_mask.shape}")
        print(f"type - {type(forget_set_mask)}")
        print(f"type - {type(training_set_mask)}")
        # bitwise and the two masks
        train_and_forget = np.array(forget_set_mask * training_set_mask,
                                    dtype=bool)
        #train_and_forget = np.dot(forget_set_mask, training_set_mask)
        print(f"number of forget points (real): {train_and_forget.sum()}")
        print(f"train_and_forget - {train_and_forget}")

        forget_set_indices = (train_and_forget).nonzero()[0]

        print(f"forget_set_indices- {forget_set_indices}")

        ulira_model_ckpt_path = ulira_model_ckpt_paths[original_model_i]
        print(f"original_model_i-   {original_model_i}")
        start_time_for_1 = datetime.datetime.now()
        print(f"unlearning_kwargs - {unlearning_kwargs}")
        _m, unlearned_model = get_u_margins(
            model=model,
            ckpt_path=ulira_model_ckpt_path,
            unlearn_fn=unlearn_fn,
            unlearning_kwargs=unlearning_kwargs,
            train_loader=train_loader,
            eval_loader=eval_loader,
            forget_set_indices=forget_set_indices,
        )
        now = datetime.datetime.now()
        print(f"time for 1 margin computation= {now-start_time_for_1}")
        all_unlearned_margins.append(_m)

        ####### EVALUATE ACCURACIES ########
        forget_loader = loader_factory(
            ds_name,
            indices=forget_set_indices,
            batch_size=50,
            indexed=True,
            drop_last=False,
        )
        # eval the unlearned model accuracy on train/val/forget set
        train_acc, val_acc, forget_acc = eval_accuracy(
            unlearned_model,
            train_loader,
            val_loader,
            forget_loader,
        )
        print("=" * 20)
        print(f"Train acc:  {train_acc:.2f} ")
        print(f"Val acc:    {val_acc:.2f} ")
        print(f"Forget acc: {forget_acc:.2f}")
        print("=" * 20)
        accuracies.append({
            "train": float(train_acc),
            "val": float(val_acc),
            "forget": float(forget_acc)
        })

    all_unlearned_margins = ch.stack(all_unlearned_margins)

    # save all_unlearned_margins  to ULIRA_RESULTS_DIR
    print(f"saving to :{ULIRA_RESULTS_DIR}")
    ch.save(all_unlearned_margins, ulira_results_save_path)
    write_yaml(ulira_accuracies_save_path, accuracies)

    print(f"donezo!")


dm_direct_config = {
    "dataset": "CIFAR10",
    #"forget_set_id": FID,
    "unlearning_algo": "dm_direct",
    "method_name": "dm_direct",
    "unlearning_algo_kwargs": {
        "dm_scores_path": dm_scores_path,
        "multiplier": 1.,
    },
    "only_direct_eval": False,
    "run_direct_eval": True,
    "use_submitit_for_direct_eval": False,
    "N_models_for_direct": 200,
}

dm_matching_config = {
    "dataset": "CIFAR10",
    #"forget_set_id": FID,
    "unlearning_algo": "dm_matching",
    "method_name": "dm_matching",
    "unlearning_algo_kwargs": {
        "dataset": "CIFAR10",
        "oracles_path": dm_scores_path,
        "loss_type": "MSE",
        "num_epochs": 4,
        "learning_rate": 1e-4,
        "batch_size": 512,
        "optimizer": "adam",
        "retain_multiplier": 20.,
        "forget_multiplier": 5.,
        "num_oracles": 1,
        "wd_lambda": 0.,
        "shuffle": True,
    },
    "run_direct_eval": True,
    "only_direct_eval": False,
    "use_submitit_for_direct_eval": False,
    "N_models_for_direct": 200,
}

from unlearning.auditors.ulira import best_params
if __name__ == "__main__":
    import sys
    index = int(sys.argv[1])

    # Parameters selected based on the lowest forget set CE on forget set 3
    # see `scrap.ipynb`

    # Forget Sets :
    ## Random forget sets
    # 0-1000: 200
    # 1000-2000: 50
    # 2000-3000: 100
    # 3000-4000: 400
    ## non-random forget sets
    # 4000-5000: forget set 3
    # 5000-6000: forget set 6
    # 6000-7000: forget set 9
    forget_set_id = None

    model_dirname = "resnet_long"  # resnet 18
    model_dirname = "resnet9_25_epochs"  # resnet 9

    lr_for_100_fs = 0.005

    best_params_GA = best_params.best_params_GA

    if index < 1000:
        unlearning_forget_set_size = 70

    elif index < 2000:
        unlearning_forget_set_size = 100
        fs_size = unlearning_forget_set_size

    elif index < 3000:
        unlearning_forget_set_size = 200
        fs_size = unlearning_forget_set_size
        best_params_SCRUB = best_params.best_params_SCRUB__large
        lr = lr_for_100_fs * (100 / fs_size)
        best_params_SCRUB["learning_rate"] = lr

        best_params_GA["learning_rate"] = best_params.best_params_GA[
            "learning_rate"] * (100 / fs_size)
    elif index < 4000:

        forget_set_id = 3
        fs_size = 1000
        best_params_SCRUB = best_params.best_params_SCRUB__large
        lr = lr_for_100_fs * (100 / fs_size) * .5
        best_params_SCRUB["learning_rate"] = lr
        unlearning_forget_set_size = 1000  # this is a filler number

        best_params.best_params_GA[
            "learning_rate"] = best_params.best_params_GA["learning_rate"] * (
                100 / fs_size)
    elif index < 5000:
        forget_set_id = 6
        fs_size = 500
        best_params_SCRUB = best_params.best_params_SCRUB__large
        lr = lr_for_100_fs * (100 / fs_size) * .5
        best_params_SCRUB["learning_rate"] = lr
        unlearning_forget_set_size = 1000  # this is a filler number

        best_params_GA["learning_rate"] = best_params.best_params_GA[
            "learning_rate"] * (100 / fs_size)
    elif index < 6000:
        forget_set_id = 9
        fs_size = 500

        best_params_SCRUB = best_params.best_params_SCRUB__large
        lr = lr_for_100_fs * (100 / fs_size) * .5
        best_params_SCRUB["learning_rate"] = lr
        unlearning_forget_set_size = 1000  # this is a filler number
        forget_set_id = 9

        best_params_GA["learning_rate"] = best_params.best_params_GA[
            "learning_rate"] * (100 / fs_size)

    elif index < 7000:
        unlearning_forget_set_size = 400
        fs_size = unlearning_forget_set_size

        best_params_SCRUB = best_params.best_params_SCRUB__large
        lr = lr_for_100_fs * (100 / fs_size) * .5
        best_params_SCRUB["learning_rate"] = lr

        best_params_GA["learning_rate"] = best_params.best_params_GA[
            "learning_rate"] * (100 / fs_size)

    else:
        raise ValueError("index out of range")

    best_params_SCRUB["drop_last"] = False

    index = index % 1000

    MODELS_PATH = ULIRA_BASE_DIR / model_dirname
    use_resnet9 = ("resnet9" in model_dirname)
    use_resnet18 = not use_resnet9

    dirname = f"results__{unlearning_forget_set_size}__{model_dirname}__"
    if forget_set_id is not None:
        dirname += f"__forget_set_{forget_set_id}"
    dirname += "__drop_last"

    RESULTS_DIR = ULIRA_BASE_DIR / dirname

    # make it
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    #method_params = [best_params_GA, best_params_GD, best_params_SCRUB]

    method_params = [
        dm_direct_config,  # 0 -100
        dm_matching_config,  # 100 - 200
        best_params_SCRUB,  # 200 - 300
        best_params.best_params_GD,  # 300 - 399
        best_params_GA,  # 400 - 499
    ]
    num_unlearnings = 2000
    unlearning_group_size = 20
    num_groups = num_unlearnings // unlearning_group_size
    method_i = index // num_groups

    method_param = method_params[method_i]
    unlearning_i = index % num_groups

    unlearning_start_i = unlearning_i * unlearning_group_size
    unlearning_stop_i = (unlearning_i + 1) * unlearning_group_size

    print(
        f"running index {index} out of a max of {num_unlearnings*3/ unlearning_group_size}"
    )
    method_name = method_param["method_name"]
    print(
        f"we are running :{method_name} - indices: {unlearning_start_i} - {unlearning_stop_i}"
    )
    file_dir = Path(__file__).resolve()

    BASE_DIR = file_dir.parent.parent.parent

    config_dict = method_param
    config_dict["dataset"] = "CIFAR10"
    config_dict["results_dir"] = RESULTS_DIR

    #config_dict["results_dir"] = "./scrub_results/"
    print(f"running config_dict ")
    for k, v in config_dict.items():
        print(f"{k} - {v}")

    print("---\n" * 3)
    if "unlearning_algo_kwargs" not in config_dict:
        config_dict["unlearning_algo_kwargs"] = {}
    for k, v in method_param.items():
        if k not in ["unlearning_algo_kwargs"]:
            config_dict["unlearning_algo_kwargs"][k] = v

    config_dict["unlearning_algo"] = method_param.get("method_name")

    config_dict["MODELS_PATH"] = MODELS_PATH
    config_dict["unlearning_forget_set_size"] = unlearning_forget_set_size
    config_dict["model_dirname"] = model_dirname
    #config_dict["results_dir"] = ULIRA_BASE_DIR

    print(f"running config_dict ")
    for k, v in config_dict.items():
        print(f"{k} - {v}")

    start = datetime.datetime.now()

    trained_epochs = 24
    if use_resnet18:
        trained_epochs = 100

    trials = 5
    for i in range(trials):
        try:
            do_several_ulira_unlearnings(
                start=unlearning_start_i,
                num_unlearnings=unlearning_group_size,
                config_dict=config_dict,
                models_dir=MODELS_PATH,
                unlearning_forget_set_size=unlearning_forget_set_size,
                use_resnet18=use_resnet18,
                forget_set_id=forget_set_id)
            break
        except Exception as e:
            print(f"failed on trial {i}")
            # sleep for a minute
            print(f"error: {e}")
            import time
            time.sleep(50)
            print("trying again")

    end = datetime.datetime.now()
    print(f"Time taken: {end-start}")
