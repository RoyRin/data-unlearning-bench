from typing import Iterable
from copy import deepcopy
import numpy as np
import torch as ch
from tqdm.auto import tqdm
from torch.cuda.amp import autocast

from unlearning.datasets.living17 import get_living17_dataloader


def to_cuda(x):
    if isinstance(x, (list, tuple)):
        return [to_cuda(xi) for xi in x]
    return x.cuda()


def construct_loader(
    ds_name, train_dataloader, indices, batch_size, num_workers=8, shuffle=True
):
    if ds_name.lower() == "living17":
        finetuning_dataloader = get_living17_dataloader(
            split="train",
            num_workers=2,
            batch_size=batch_size,
            shuffle=shuffle,
            indices=indices,
            indexed=True,
            drop_last=False,
        )
    else:
        ds = train_dataloader.dataset
        finetuning_dataset = ch.utils.data.Subset(
            dataset=ds,
            indices=indices,
        )
        finetuning_dataloader = ch.utils.data.DataLoader(
            finetuning_dataset, batch_size=batch_size, shuffle=shuffle
        )

        if ds_name == "QNLI":
            from unlearning.auditors.utils import WrappedDataLoader

            finetuning_dataloader = WrappedDataLoader(finetuning_dataloader)

    return finetuning_dataloader


def add_vector_to_parameters(
    vector: ch.Tensor, parameters: Iterable[ch.nn.Parameter]
) -> None:
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.data += vector[pointer : pointer + num_param].reshape(param.shape).clone()
        pointer += num_param



def get_margin_from_logits(
    single_logits: ch.Tensor,
    labels: ch.Tensor,
) -> ch.Tensor:
    """
    Compute margins directly from logits and labels.
    
    Args:
        logits: Tensor of shape [batch_size, num_classes] containing the logits
        labels: Tensor of shape [batch_size] containing the true class labels
        
    Returns:
        Tensor of shape [batch_size] containing the margins for each sample
        Margin is defined as: margin = logit_correct - log_sum_exp(logit_other)
    """
    bindex = ch.arange(single_logits.shape[0]).to(single_logits.device, non_blocking=False)
    logits_correct = single_logits[bindex, labels]

    cloned_logits = single_logits.clone()
    cloned_logits[bindex, labels] = ch.tensor(
        -ch.inf, device=cloned_logits.device, dtype=cloned_logits.dtype
    )

    return logits_correct - cloned_logits.logsumexp(dim=-1)


def get_margin(
    model: ch.nn.Module,
    images: ch.Tensor,
    labels: ch.Tensor,
    indices: ch.Tensor,
) -> ch.Tensor:
    """
    for each image in images, compute the margin of the correct class
    margin is defined as :
    margin = logit_correct - log_sum_exp(logit_other)
    """
    logits = model(images, indices)
    return get_margin_from_logits(logits, labels)
 


def get_margins_from_logits(logits: ch.Tensor, labels: ch.Tensor):
    # get margins for each logit in logits
    margins = []
    for logit, label in zip(logits, labels):
        margins.append(get_margin_from_logits(logit.unsqueeze(0), label))
    return ch.stack(margins)


def get_margins(model, loader):
    model.eval()

    all_margins = []
    with ch.no_grad():
        for x, y, idx in tqdm(loader, desc="getting margins.."):
            x, y = to_cuda(x), y.cuda()
            with autocast():
                margins = get_margin(model, x, y, idx)
            all_margins.append(margins.cpu())
    return ch.cat(all_margins)

