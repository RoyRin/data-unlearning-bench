import torch as ch
import numpy as np
from typing import List
from pathlib import Path
from copy import deepcopy

from unlearning.training.train import train_cifar10
from unlearning.models.resnet9 import ResNet9
from unlearning.auditors.utils import load_model_from_dict, DATASET_TO_NUM_EPOCHS

from unlearning import BASE_DIR, LOG_DIR, ULIRA_BASE_DIR, ORACLE_BASE_DIR, RESULTS_DIR


def do_nothing(
    model: ch.nn.Module,
    train_dataloader: ch.utils.data.DataLoader = None,
    forget_indices: List[int] = None,
    forget_dataloader: ch.utils.data.Subset = None,
    **kwargs,
) -> ch.nn.Module:
    unlearned_model = deepcopy(model)
    return unlearned_model.eval()


def load_an_oracle(
    model: ch.nn.Module,
    train_dataloader: ch.utils.data.DataLoader = None,
    forget_indices: List[int] = None,
    forget_dataloader: ch.utils.data.Subset = None,
    dataset: str = "cifar10",
    forget_set_id: int = 0,
    held_out: bool = True,
    held_out_start: int = 200,
    held_out_end: int = 250,
    **kwargs,
) -> ch.nn.Module:

    oracles_list = []
    iterator = (range(held_out_start, held_out_end)
                if held_out else range(held_out_start))
    for i in iterator:
        last_ep = DATASET_TO_NUM_EPOCHS[dataset.upper()]
        p = BASE_DIR / f"{dataset}/forget_set_{forget_set_id}/sd_{i}____epoch_{last_ep-1}.pt"
        oracles_list.append(p)

    oracle_path = np.random.choice(oracles_list)
    oracle_sd = ch.load(oracle_path)
    unlearned_model = deepcopy(model)

    if dataset == 'LIVING17':
        unlearned_model.model.load_state_dict(oracle_sd)
    elif dataset == 'QNLI':
        unlearned_model.model.load_state_dict(oracle_sd)
    else:
        unlearned_model.load_state_dict(oracle_sd)
    return unlearned_model.eval()


def retrain_an_oracle(
    model: ch.nn.Module,
    train_dataloader: ch.utils.data.DataLoader = None,
    forget_indices: List[int] = None,
    forget_dataloader: ch.utils.data.Subset = None,
    dataset: str = "cifar10",
    forget_set_id: int = 0,
    batch_size: int = 512,
    num_epochs: int = 24,
    **kwargs,
) -> ch.nn.Module:

    retain_set_indices = np.setdiff1d(np.arange(len(train_dataloader.dataset)),
                                      forget_indices)
    ds = train_dataloader.dataset
    finetuning_dataset = ch.utils.data.Subset(
        dataset=ds,
        indices=retain_set_indices,
    )
    finetuning_dataloader = ch.utils.data.DataLoader(finetuning_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=True)

    model = ResNet9(num_classes=10, wrapped=True).cuda()

    if dataset.lower() != 'cifar10':
        raise NotImplementedError('Only cifar10 is supported for now')
    new_model = train_cifar10(model=model,
                              loader=finetuning_dataloader,
                              checkpoints_dir='/tmp',
                              checkpoint_epochs=[],
                              train_epochs=num_epochs,
                              eval_loader=None,
                              should_save_logits=True,
                              overwrite=True)
    return new_model.eval()
