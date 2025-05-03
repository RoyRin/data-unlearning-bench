import numpy as np
import torch as ch
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
import torch

from unlearning.unlearning_algos.utils import get_margin

dataset_name = "CIFAR10"

from unlearning import BASE_DIR, LOG_DIR, ULIRA_BASE_DIR

masks_path = ULIRA_BASE_DIR / "training_masks.npy"
masks = np.load(masks_path)

#
from unlearning.auditors.utils import loader_factory

ds_name_ = "CIFAR10"
train_loader = loader_factory(ds_name_, indexed=True)
val_loader = loader_factory(ds_name_, split="val", indexed=True)
eval_set_inds = np.arange(len(train_loader.dataset) + len(val_loader.dataset))
eval_loader = loader_factory(ds_name_,
                             split="train_and_val",
                             indices=eval_set_inds,
                             indexed=True)

train_targets = train_loader.dataset.original_dataset.targets
val_targets = val_loader.dataset.original_dataset.targets


def get_train_logits(model_index, dataset_name):
    train_logits_path = ULIRA_BASE_DIR / dataset_name / f"train_logits_{model_index}.pt"
    if not train_logits_path.exists():
        return None
    return torch.load(train_logits_path)


def get_val_logits(model_index, dataset_name):
    val_logits_path = ULIRA_BASE_DIR / dataset_name / f"val_logits_{model_index}.pt"
    return torch.load(val_logits_path)


def get_logits(model_index, dataset_name):
    train_logits_path = ULIRA_BASE_DIR / dataset_name / f"train_logits_{model_index}.pt"
    val_logits_path = ULIRA_BASE_DIR / dataset_name / f"val_logits_{model_index}.pt"
    if not train_logits_path.exists():
        return None
    return torch.load(train_logits_path), torch.load(val_logits_path)


def compute_margins(logit, true_label):
    logit_other = logit.clone()
    logit_other[true_label] = -np.inf

    return logit[true_label] - logit_other.logsumexp(dim=-1)


def get_margins(model_ind, fake_ds_name="final_models"):
    train_logits = get_train_logits(model_ind, fake_ds_name)
    val_logits = get_val_logits(model_ind, fake_ds_name)
    train_margins = [
        compute_margins(logit, target)
        for (logit, target) in zip(train_logits, train_targets)
    ]
    val_margins = [
        compute_margins(logit, target)
        for (logit, target) in zip(val_logits, val_targets)
    ]
    return np.array(train_margins), np.array(val_margins)


all_models = ULIRA_BASE_DIR / "final_models"
all_models = all_models.glob("sd_*.pt")
nums = []
for model in all_models:

    #print(model)
    #print(model.name.split("_"))
    num = int(str(model.name).split("_")[1])
    nums.append(num)
    #print(num)
#nums = [int(str(model).split("_")[1]) for model in all_models]
nums = sorted(nums)
expected = set(range(len(nums)))
missing = expected - set(nums)
print(f"Missing: {sorted(missing)}")

import sys
try:
    start_index = int(sys.argv[1])
except:
    start_index = 0

print(f"start_index- {start_index}")

model_dir_name = "final_models"
model_dir_name = "resnet_long"

# MODEL COUNT HARDCODED
for model_ind in range(start_index, 525):

    train_path = ULIRA_BASE_DIR / model_dir_name / f"train_margins_{model_ind}.pt"
    val_path = ULIRA_BASE_DIR / model_dir_name / f"val_margins_{model_ind}.pt"
    if Path(train_path).exists():
        print(f"already computed {model_ind}")
        continue

    if model_ind % 10 == 0:
        print(model_ind)
    try:
        train_margins, val_margins = get_margins(model_ind,
                                                 fake_ds_name=model_dir_name)
    except Exception as e:
        print(e)

        print(f"fail on {model_ind}")
        continue
    # torch save

    torch.save(train_margins, train_path)
    torch.save(val_margins, val_path)
