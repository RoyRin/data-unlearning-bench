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
from unlearning.auditors.direct import (
    direct_audit_precomputed,
    get_u_margins,
    plot_margins_direct,
)
from unlearning.datasets import DATASET_SIZES, DATASET_VAL_SIZES
from unlearning.unlearning_algos.base_nn import NAME_TO_ALGO


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
    if "model" in first_key:
        model.load_state_dict(loaded_model)
    else:
        # add ".model" to each key in k,vs
        loaded_model = {f"model.{k}": v for k, v in loaded_model.items()}
        model.load_state_dict(loaded_model)
    return model


def eval_suite(config_yaml_file=None,
               config_dict=None,
               overwrite=False,
               unlearn_fn=None):
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
    # save config to results dir
    with open(RESULTS_DIR / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Overwrite check (checks only for `direct`, but we can change this if we want)
    direct_results_file = RESULTS_DIR / "direct" / "ks_p_kl_ce.npy"
    if not overwrite and direct_results_file.exists():
        raise FileExistsError(
            f"{RESULTS_DIR} already exists. Set overwrite=True to overwrite.")

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

    # We tie the model to the dataset, so we have fewer moving pieces
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

    drop_last = False  # flag used specifically for SCRUB, which potentially behaves differently on the last batch
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
    # f stands for "full model",
    #   i.e. the model that was trained on the enitre dataset (retain + forget)
    # o stands for "oracle model"
    #   i.e. the model that was trained on the oracle dataset (retain only)

    splits = ["train", "val"]

    f_ckpt_paths, f_logit_paths, f_margins_paths = get_full_model_paths(
        ds_name, splits=splits)

    (
        o_ckpt_0_path,  # we only need a single oracle checkpoint
        o_logit_paths,
        o_margins_paths,
    ) = get_oracle_paths(ds_name, config["forget_set_id"], splits=splits)
    logger.info(f"Loaded paths of pretrained models.")

    model = load_model(f_ckpt_paths[0], model_factory, ds_name)

    _inds_to_plot = rng.choice(np.arange(len(forget_set_indices)), 20)
    inds_to_plot = forget_set_indices[_inds_to_plot]

    logger.info(f"Loaded a pretrained model.")
    ####### END OF LOADING PRETRAINED MODELS ########
    if not config.get("only_direct_eval", True):

        ####### RUN UNLEARNING ALGO ONCE ########
        # run the unlearning algo on the model we just created once
        logger.info(f"Running unlearning algo..")

        unlearned_model = unlearn_fn(
            model=model,
            train_dataloader=train_loader,
            forget_dataloader=forget_loader,
            forget_indices=forget_set_indices,
            **unlearning_kwargs,
        )
        logger.info(f"Done running unlearning algo.")
        ####### END OF RUNNING UNLEARNING ALGO ########

        ####### EVALUATE ACCURACIES ########
        # eval the unlearned model accuracy on train/val/forget set
        train_acc, val_acc, forget_acc = eval_accuracy(
            unlearned_model,
            train_loader,
            val_loader,
            forget_loader,
        )
        oracle_model = load_model(o_ckpt_0_path, model_factory, ds_name)

        oracle_model.cuda().eval()  # can't be too sure
        oracle_train_acc, oracle_val_acc, oracle_forget_acc = eval_accuracy(
            oracle_model, train_loader, val_loader, forget_loader)

        print("=" * 20)
        print(f"Train acc:  {train_acc:.2f} (oracle: {oracle_train_acc:.2f})")
        print(f"Val acc:    {val_acc:.2f} (oracle: {oracle_val_acc:.2f})")
        print(
            f"Forget acc: {forget_acc:.2f} (oracle: {oracle_forget_acc:.2f})")
        print("=" * 20)
        # save csv with the accuracies
        # columns are forget_set_id, unlearning_algo, train_acc, val_acc, forget_acc
        # and there are two rows, one with the unlearned model, one with the oracle
        evaluations = {
            "forget_set_id":
            [config["forget_set_id"], config["forget_set_id"]],
            "unlearning_algo": [config["unlearning_algo"], "oracle"],
            "train_acc": [train_acc, oracle_train_acc],
            "val_acc": [val_acc, oracle_val_acc],
            "forget_acc": [forget_acc, oracle_forget_acc],
        }
        df = pd.DataFrame(evaluations)
        df.to_csv(RESULTS_DIR / "accuracies.csv")
        logger.info(f"Saved accuracies to {RESULTS_DIR / 'accuracies.csv'}")
        results.update(evaluations)

        ####### END OF EVALUATING ACCURACIES ########
        print(f"size of forget_set_indices - {forget_set_indices.shape}")

    ####### DIRECT EVAL ########
    if config.get("run_direct_eval", False):

        DIRECT_DIR = RESULTS_DIR / "direct"
        DIRECT_DIR.mkdir(parents=True, exist_ok=True)
        # run the expensive direct eval
        # this involves running the unlearning algo many times
        # (once on each checkpoint from full_model_ckpt_paths)
        all_unlearned_margins = []

        f_ckpt_paths = f_ckpt_paths[:config["N_models_for_direct"]]
        for f_ckpt_path in tqdm(f_ckpt_paths, desc="unlearning models.."):
            _m, __unlearned_model = get_u_margins(
                model,
                f_ckpt_path,
                unlearn_fn,
                unlearning_kwargs,
                train_loader,
                eval_loader,
                forget_set_indices,
            )
            all_unlearned_margins.append(_m)
        all_unlearned_margins = ch.stack(all_unlearned_margins)

        logger.info(f"Done computing margins for direct eval..")
        if config.get("save_unlearned_margins", True):
            np.save(
                DIRECT_DIR / "direct_unlearned_margins.npy",
                all_unlearned_margins.numpy(),
            )

        logger.info("Loading oracle margins..")
        assert len(o_margins_paths) == 2
        assert o_margins_paths[0].stem.startswith("train")
        assert o_margins_paths[1].stem.startswith("val")
        all_oracle_margins = ch.cat(
            [ch.load(path) for path in o_margins_paths], dim=1)
        all_oracle_margins = all_oracle_margins[:, eval_set_inds]

        fig_margins_direct = plot_margins_direct(all_unlearned_margins,
                                                 all_oracle_margins,
                                                 inds_to_plot)
        fig_margins_direct.savefig(DIRECT_DIR / "direct_margins_hist.png")

        direct_results = direct_audit_precomputed(all_unlearned_margins,
                                                  all_oracle_margins)

        np.save(DIRECT_DIR / "ks_p_kl_ce.npy", direct_results)

        direct_ulira_results = {
            "all_unlearned_margins": all_unlearned_margins,
            "direct_results": direct_results,
            #"ulira_strong": ulira_strong_results,
        }
        results["direct_ulira"] = direct_ulira_results

        fig_direct_results, dir_axes = plt.subplots(1, 3)
        for i, col in enumerate(["ks pval", "t pval", "KL divergence"]):
            print(f"direct_results.shape - {direct_results.shape}")
            dir_axes[i].hist(direct_results[:, i], bins=20)
            dir_axes[i].set_title(col)
        fig_direct_results.savefig(DIRECT_DIR / "direct_results_hist.png")
        # save dictionary direct_results as npz
        np.savez(DIRECT_DIR / "direct_results.npz", **direct_ulira_results)

    return RESULTS_DIR, results
