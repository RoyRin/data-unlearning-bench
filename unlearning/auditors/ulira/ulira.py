"""
This is online lira
"""

import re
import numpy as np
import torch as ch
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional

from unlearning.unlearning_algos.utils import get_margin
from unlearning.auditors.direct import (
    direct_audit_precomputed,
    get_u_margins,
    u_margin_job,
    plot_margins_direct,
)

from scipy.stats import norm
import torch
from unlearning.auditors.utils import loader_factory,
import os
import shutil
import yaml
import numpy as np
import torch as ch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from unlearning.auditors.utils import loader_factory,

from unlearning.auditors.direct import (
    get_u_margins, )
from unlearning.unlearning_algos.base_nn import NAME_TO_ALGO

from contextlib import redirect_stdout, redirect_stderr

from unlearning import BASE_DIR, LOG_DIR, ULIRA_BASE_DIR


def to_cuda(x):
    if isinstance(x, (list, tuple)):
        return [to_cuda(xi) for xi in x]
    return x.cuda()


def get_ulira_masks():
    masks_path = ULIRA_BASE_DIR / "training_masks.npy"
    masks = np.load(masks_path)
    return masks


def sort_key(path):
    # Extract numerical parts and convert to integers
    numbers = re.findall(r'\d+', path.name)
    return [int(num) for num in numbers]


def get_ulira_model_paths(dataset, splits=["train", "val"]):
    DATA_DIR = ULIRA_BASE_DIR / "final_models"
    ulira_model_ckpt_paths = sorted(list(DATA_DIR.glob("sd_*_epoch_23.pt")),
                                    key=sort_key)
    # Sort paths using the custom key
    # train and val
    ulira_model_logit_paths = [
        DATA_DIR / f"{split}_logits_all.pt" for split in splits
    ]
    ulira_model_margins_paths = [
        DATA_DIR / f"{split}_margins_all.pt" for split in splits
    ]
    return ulira_model_ckpt_paths, ulira_model_logit_paths, ulira_model_margins_paths


def get_forget_and_val_margins_from_paths(
    margin_paths: List[str],
    forget_set_indices: np.ndarray,
    val_set_indices: Optional[np.ndarray] = None,
):
    assert len(margin_paths) == 2
    assert margin_paths[0].stem.startswith("train")
    assert margin_paths[1].stem.startswith("val")
    forget_margins = ch.load(margin_paths[0])[:, forget_set_indices]
    val_margins = ch.load(margin_paths[1])
    print(f"forget margin size - {forget_margins.shape}")
    print(f"val margin size - {val_margins.shape}")
    if val_set_indices is not None:
        val_margins = val_margins[:, val_set_indices]
    all_margins = ch.cat([forget_margins, val_margins], dim=1)
    return all_margins


def cheap_ulira_audit_precomputed(
    dataset_name: str,
    loader_factory: callable,
    unlearned_model: ch.nn.Module,
    full_margins_paths: List[str],
    oracle_margins_paths: List[str],
    forget_set_indices: np.ndarray,
    val_set_indices: Optional[np.ndarray] = None,
    val_subsample_random_seed: int = 0,
    val_set_size: int = 10_000,
    **kwargs,
):
    """
    margins paths start with either "train_" or "val_"

    if val_indices is None, we randomly sample a subset from the val set of the
    same size as forget_set_indices
    """
    if len(kwargs) > 0:
        print(f"Received unused kwargs: {kwargs}")

    if val_set_indices is None:
        np.random.seed(val_subsample_random_seed)
        val_set_indices = np.random.choice(np.arange(val_set_size),
                                           len(forget_set_indices))

    # collect pre-computed oracle and full margins
    all_oracle_margins = get_forget_and_val_margins_from_paths(
        oracle_margins_paths, forget_set_indices, val_set_indices)

    all_full_margins = get_forget_and_val_margins_from_paths(
        full_margins_paths, forget_set_indices, val_set_indices)

    oracle_means = all_oracle_margins.mean(dim=0)
    oracle_vars = all_oracle_margins.var(dim=0)

    full_means = all_full_margins.mean(dim=0)
    full_vars = all_full_margins.var(dim=0)

    # compute unlearned margins
    forget_loader = loader_factory(
        dataset=dataset_name,
        split="train",
        indices=forget_set_indices,
        batch_size=100,
        shuffle=False,
    )
    val_loader = loader_factory(
        dataset=dataset_name,
        split="val",
        indices=val_set_indices,
        batch_size=100,
        shuffle=False,
    )
    forget_unlearned_margins = []
    for x, y, inds in forget_loader:
        x = to_cuda(x)
        y = y.cuda()
        with ch.no_grad():
            unlearned_margins = get_margin(unlearned_model, x, y, inds)
        forget_unlearned_margins.append(unlearned_margins.cpu())
    val_unlearned_margins = []
    for x, y, inds in val_loader:
        x = to_cuda(x)
        y = y.cuda()
        with ch.no_grad():
            unlearned_margins = get_margin(unlearned_model, x, y, inds)
        val_unlearned_margins.append(unlearned_margins.cpu())
    all_unlearned_margins = ch.cat(
        [ch.cat(forget_unlearned_margins),
         ch.cat(val_unlearned_margins)])

    ratios = []
    print("Calculating likelihood ratios")

    def gaussian_probability(margin, mean, var):
        return np.exp(-((margin - mean)**2) / (2 * var)) / ((var)**0.5)

    for eval_sample in tqdm(range(len(all_unlearned_margins)),
                            desc="calcuating likelihood ratios.."):
        # do a likelihood ratio test
        margin = all_unlearned_margins[eval_sample].item()
        full_mean = full_means[eval_sample].item()
        full_var = full_vars[eval_sample].item()
        oracle_mean = oracle_means[eval_sample].item()
        oracle_var = oracle_vars[eval_sample].item()

        full_p = gaussian_probability(margin, full_mean, full_var)
        oracle_p = gaussian_probability(margin, oracle_mean, oracle_var)

        lr = full_p / oracle_p
        ratios.append(lr)

    return (
        ch.tensor(ratios),
        all_unlearned_margins,
        all_full_margins,
        all_oracle_margins,
    )


def generate_ulira_forget_mask(dataset_name,
                               training_mask,
                               SEED=42,
                               unlearning_forget_set_count=2000,
                               unlearning_forget_set_size=200,
                               class_5_range=1_000):
    """
    class_5_range is the number of points from class 5 we want to consider forgetting
    """
    train_loader = loader_factory(dataset_name, indexed=True)

    np.random.seed(SEED)
    train_targets = np.array(train_loader.dataset.original_dataset.targets)
    class_5_mask = (train_targets == 5)
    N = len(train_targets)

    class_5_indices = class_5_mask.nonzero()[0]
    # pick 1000 points from class 5
    class_5_indices = np.random.choice(class_5_indices,
                                       class_5_range,
                                       replace=False)
    class_5_mask = np.zeros(N)
    class_5_mask[class_5_indices] = 1
    ####
    ulira_forget_mask = []
    for mask_i in range(unlearning_forget_set_count):
        indiv_training_mask_ = training_mask[mask_i]  # .nonzero()[0]

        train_and_5 = np.array(indiv_training_mask_ * class_5_mask, dtype=bool)

        class_5_trained_on_mask = np.array(train_and_5)
        class_5_trained_on_indices = class_5_trained_on_mask.nonzero()[0]
        indices = np.random.choice(class_5_trained_on_indices,
                                   unlearning_forget_set_size,
                                   replace=False)
        mask = np.zeros(N)
        mask[indices] = 1
        ulira_forget_mask.append(mask)

    ulira_forget_mask = np.array(ulira_forget_mask)
    return ulira_forget_mask


def get_ulira_training_masks():
    masks_path = ULIRA_BASE_DIR / "training_masks.npy"
    ulira_mask = np.load(masks_path)
    return ulira_mask


def get_ulira_forget_mask(dataset_name,
                          class_5_range=1000,
                          unlearning_forget_set_size=50,
                          overwrite=False):
    training_mask = get_ulira_training_masks(dataset_name)
    ulira_forget_mask_f = ULIRA_BASE_DIR / f"all_forget_masks__{unlearning_forget_set_size}.npy"

    if not ulira_forget_mask_f.exists() or overwrite:
        print(f"generating and saving")
        ulira_forget_mask = generate_ulira_forget_mask(
            dataset_name,
            class_5_range=class_5_range,
            unlearning_forget_set_size=unlearning_forget_set_size,
            training_mask=training_mask)
        np.save(ulira_forget_mask_f, ulira_forget_mask)
    else:
        print("loading ")
        ulira_forget_mask = np.load(ulira_forget_mask_f)

    return ulira_forget_mask


def gaussian_probability(mean, sigma, value):
    return norm.pdf(value, mean, sigma)


def single_ulira(
    all_unlearned_margins: ch.Tensor,  # T x n
    all_oracle_margins: ch.Tensor,  # T x n
    hold_out_model: int,
    threshold=0.5,
):
    """
    Compute the ULIRA (Unlearning Likelihood Ratio) results for each sample. for a single unlearned model.

    Note:
        1: (x,y) is likely a member of training (unlearned model)
        0: (x,y) is likely a member of the oracle (oracle model)

    Args:
        all_unlearned_margins (ch.Tensor): Tensor of shape T x n containing the margins of all unlearned models.
        all_oracle_margins (ch.Tensor): Tensor of shape T x n containing the margins of all oracle models.
        hold_out_model (int): Index of the hold-out model.
        threshold (float, optional): Threshold value for the likelihood ratio. Defaults to 0.5.

    Returns:
        np.ndarray: Array of ULIRA results for each sample, where 1 indicates the likelihood ratio is above the threshold, and 0 otherwise.
    """
    N = all_unlearned_margins.shape[1]
    ULIRA_results = np.zeros(N)

    for sample in range(N):
        oracle_arr = all_oracle_margins[:, sample].cpu().numpy()
        unlearned_arr = all_unlearned_margins[:, sample].cpu().numpy()

        #

        unlearned_model_margin = unlearned_arr[hold_out_model]
        other_unlearned_models = np.delete(unlearned_arr, hold_out_model)

        # fit gaussians
        oracle_mean, oracle_std = np.mean(oracle_arr), np.std(oracle_arr)
        unlearned_mean, unlearned_std = np.mean(
            other_unlearned_models), np.std(other_unlearned_models)

        # compute LIRA
        oracle_prob = gaussian_probability(oracle_mean, oracle_std,
                                           unlearned_model_margin)
        unlearned_prob = gaussian_probability(unlearned_mean, unlearned_std,
                                              unlearned_model_margin)

        likelihood_ratio = unlearned_prob / (unlearned_prob + oracle_prob)

        # save result
        ULIRA_results[sample] = (likelihood_ratio > threshold).astype(float)

    return ULIRA_results


def ulira_strong(
    all_unlearned_margins: ch.Tensor,
    all_oracle_margins: ch.Tensor,
):
    """
    """
    results = []
    model_counts = all_unlearned_margins.shape[0]

    # print(f"model counts: {model_counts}")

    # compute ULIRA for each unlearned model
    for hold_out_unlearned_model in range(model_counts):
        single_ulira_results = single_ulira(all_unlearned_margins,
                                            all_oracle_margins,
                                            hold_out_unlearned_model)
        results.append(single_ulira_results)

    # average over all unlearned models
    results = np.mean(np.stack(results), axis=0)
    return results


def get_masks_with_and_without(masks, index):
    m = masks[:, index].T
    with_index, without_index = np.where(m == 1)[0], np.where(m == 0)[0]
    return with_index, without_index


def get_precomputed_margins(model_ind):
    train_margins_path = ULIRA_BASE_DIR / \
        "final_models" / f"train_margins_{model_ind}.pt"
    val_margins_path = ULIRA_BASE_DIR / \
        "final_models" / f"val_margins_{model_ind}.pt"

    return torch.load(train_margins_path), torch.load(val_margins_path)


def load_margins_with_and_without(models_with, models_without):
    margins_with = []
    margins_without = []

    for model_ind in models_with:
        try:
            train_margins, val_margins = get_precomputed_margins(model_ind)
            all_margins = ch.cat([ch.cat(train_margins), ch.cat(val_margins)])
            margins_with.append(all_margins)

        except:
            continue

    for model_ind in models_without:
        try:
            train_margins, val_margins = get_precomputed_margins(model_ind)

            all_margins = ch.cat([ch.cat(train_margins), ch.cat(val_margins)])
            margins_without.append(all_margins)
        except:
            continue
    return np.array(margins_with), np.array(margins_without)


def load_all_ulira_margins(dataset_name="CIFAR10", dir_name="final_models"):
    train_f = ULIRA_BASE_DIR / dir_name / "train_margins_all.pt"
    val_f = ULIRA_BASE_DIR / dir_name / "val_margins_all.pt"
    return torch.load(train_f), torch.load(val_f)


def train_unlearnings(unlearn_fn,
                      unlearning_kwargs,
                      ulira_f_ckpt_paths,
                      eval_loader,
                      model_factory,
                      max_unlearnings=None,
                      dataset_name="CIFAR10"):
    ulira_mask = get_ulira_training_masks()
    ulira_forget_mask = get_ulira_forget_mask(dataset_name=dataset_name)

    model_count, N = ulira_mask.shape
    ulira_unlearned_margins = []

    if max_unlearnings is not None:
        model_count = min(model_count, max_unlearnings)

    for model_ind in range(model_count):
        # unlearn model with index model_ind
        train_mask = ulira_mask[model_ind]
        train_inds = train_mask.nonzero()[0]

        train_loader_ = loader_factory(
            dataset_name,
            indices=train_inds,
            batch_size=50,
            indexed=True,
        )
        ulira_f_ckpt_path = ulira_f_ckpt_paths[model_ind]

        forget_set_indices_ulira = ulira_forget_mask[model_ind].nonzero()[0]
        forget_set_indices_ulira = list(
            set(forget_set_indices_ulira).intersection(set(train_inds)))
        # unlearn model, the compute margins
        model = model_factory()  # model type
        # compute margins
        _m = get_u_margins(
            model,
            ulira_f_ckpt_path,
            unlearn_fn,
            unlearning_kwargs,
            train_loader_,
            eval_loader,
            forget_set_indices_ulira,
        )

        ulira_unlearned_margins.append(_m)
    ulira_unlearned_margins = ch.stack(ulira_unlearned_margins)
    return ulira_unlearned_margins


def read_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config


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


def get_models_with_pt_forgotten(masks, forget_mask, exclude_ind):
    """
    find the indices where the point is in the plan and in the forget mask
    """
    models_with = []

    models_inds_with, _ = get_masks_with_and_without(masks, exclude_ind)

    for model_ind in models_inds_with:
        if forget_mask[model_ind, exclude_ind] == 1:
            models_with.append(model_ind)
    return np.array(models_with)


def ulira_paper(
    all_unlearned_margins_ulira: ch.Tensor,
    all_oracle_margins_ulira: ch.Tensor,
    dataset_name="CIFAR10",
):
    """
    """
    results = []
    masks = get_ulira_masks()
    oracle_count, N = masks.shape
    forget_mask = get_ulira_forget_mask(dataset_name=dataset_name)

    for exclude_ind in range(N):

        models_inds_with = get_models_with_pt_forgotten(
            masks, forget_mask, exclude_ind
        )  # index that has the point in the plan, and in the forget mask
        _, models_inds_without = get_masks_with_and_without(masks, exclude_ind)
        unlearned_margins = all_unlearned_margins_ulira[models_inds_with]

        oracle_margins = all_oracle_margins_ulira[models_inds_without]

        model_counts = unlearned_margins.shape[0]
        for hold_out_unlearned_model in range(model_counts):
            single_ulira_results = single_ulira(unlearned_margins,
                                                oracle_margins,
                                                hold_out_unlearned_model)
            results.append(single_ulira_results)

        # average over all unlearned models
        results = np.mean(np.stack(results), axis=0)
    return results
