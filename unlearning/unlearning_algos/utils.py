from typing import Iterable
from copy import deepcopy
import numpy as np
import torch as ch
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from trak.gradient_computers import FunctionalGradientComputer
from trak.modelout_functions import ImageClassificationModelOutput
from trak.projectors import CudaProjector, ProjectionType
from trak.utils import vectorize

from unlearning.datasets.living17 import get_living17_dataloader


def to_cuda(x):
    if isinstance(x, (list, tuple)):
        return [to_cuda(xi) for xi in x]
    return x.cuda()


def construct_loader(ds_name,
                     train_dataloader,
                     indices,
                     batch_size,
                     num_workers=8,
                     shuffle=True):
    if ds_name.lower() == 'living17':
        finetuning_dataloader = get_living17_dataloader(split="train",
                                                        num_workers=2,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle,
                                                        indices=indices,
                                                        indexed=True,
                                                        drop_last=False)
    else:
        ds = train_dataloader.dataset
        finetuning_dataset = ch.utils.data.Subset(
            dataset=ds,
            indices=indices,
        )
        finetuning_dataloader = ch.utils.data.DataLoader(finetuning_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=shuffle)

        if ds_name == 'QNLI':
            from unlearning.auditors.utils import WrappedDataLoader
            finetuning_dataloader = WrappedDataLoader(finetuning_dataloader)

    return finetuning_dataloader


def add_vector_to_parameters(vector: ch.Tensor,
                             parameters: Iterable[ch.nn.Parameter]) -> None:
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.data += vector[pointer:pointer + num_param].reshape(
            param.shape).clone()
        pointer += num_param


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
    bindex = ch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
    logits_correct = logits[bindex, labels]

    cloned_logits = logits.clone()
    cloned_logits[bindex, labels] = ch.tensor(-ch.inf,
                                              device=cloned_logits.device,
                                              dtype=cloned_logits.dtype)

    return logits_correct - cloned_logits.logsumexp(dim=-1)


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


def get_xtx_inv_matrix_no_projection(
    model: ch.nn.Module,
    dataloader: ch.utils.data.DataLoader,
    lam: float = 1e-8,
    use_r: bool = False,
) -> ch.Tensor:
    try:
        model_to_featurize = deepcopy(model)
    except NotImplementedError:
        model_to_featurize = model.custom_deepcopy()

    grad_dim = sum(p.numel() for p in model_to_featurize.parameters())
    gradient_computer = FunctionalGradientComputer(
        model=model_to_featurize,
        task=ImageClassificationModelOutput(),
        grad_dim=grad_dim,
        dtype=ch.float32,
        device=ch.device("cuda"),
    )

    if use_r:
        p = []
        for images, labels in dataloader:
            batch = [images.cuda(), labels.cuda()]
            with ch.no_grad():
                outputs = model_to_featurize(batch[0])
                ps = ch.nn.functional.softmax(
                    outputs,
                    dim=1,
                )[np.arange(len(labels)), labels]
                p.append(ps)
        p = ch.cat(p, dim=0)
        r = (p * (1 - p)).cuda()
        print("R term:", r)
    else:
        r = ch.ones(len(dataloader.dataset)).cuda()

    xtx_inv_matrix = ch.zeros((grad_dim, grad_dim), device=ch.device("cuda"))
    for batch_i, (images, labels) in enumerate(
            tqdm(dataloader, desc="Computing X^T X inverse matrix")):
        batch = [images.cuda(), labels.cuda()]
        g = gradient_computer.compute_per_sample_grad(batch)
        g = vectorize(g)

        s, e = batch_i * dataloader.batch_size, (batch_i +
                                                 1) * dataloader.batch_size

        xtx_inv_matrix += ch.matmul(g.T, g * r[s:e, None])

    xtx_inv_matrix /= len(dataloader.dataset)
    xtx_inv_matrix += lam * ch.eye(grad_dim, device="cuda")
    print("XTX matrix:", xtx_inv_matrix)
    print("XTX[0]:", xtx_inv_matrix[0])
    xtx_inv_matrix = ch.linalg.inv(xtx_inv_matrix).cpu()
    return xtx_inv_matrix


def get_xtx_inv_matrix(
    model: ch.nn.Module,
    dataloader: ch.utils.data.DataLoader,
    proj_dim: int = 8192,
    lam: float = 1e-8,
    use_r: bool = False,
    seed: int = 0,
) -> ch.Tensor:
    try:
        model_to_featurize = deepcopy(model)
    except NotImplementedError:
        model_to_featurize = model.custom_deepcopy()

    grad_dim = sum(p.numel() for p in model_to_featurize.parameters())
    gradient_computer = FunctionalGradientComputer(
        model=model_to_featurize,
        task=ImageClassificationModelOutput(),
        grad_dim=grad_dim,
        dtype=ch.float32,
        device=ch.device("cuda"),
    )
    projector = CudaProjector(
        grad_dim=grad_dim,
        proj_dim=proj_dim,
        seed=seed,
        proj_type=ProjectionType.rademacher,
        device=ch.device("cuda"),
        max_batch_size=32,
    )

    normalize_factor = ch.from_numpy(np.array(np.sqrt(proj_dim))).cuda()

    if use_r:
        p = []
        for images, labels in dataloader:
            batch = [images.cuda(), labels.cuda()]
            with ch.no_grad():
                outputs = model_to_featurize(batch[0])
                ps = ch.nn.functional.softmax(
                    outputs,
                    dim=1,
                )[np.arange(len(labels)), labels]
                p.append(ps)
        p = ch.cat(p, dim=0)
        r = (p * (1 - p)).cuda()
        print("R term:", r)
    else:
        r = ch.ones(len(dataloader.dataset)).cuda()

    xtx_inv_matrix = ch.zeros((proj_dim, proj_dim), device=ch.device("cuda"))
    for batch_i, (images, labels) in enumerate(
            tqdm(dataloader, desc="Computing X^T X inverse matrix")):
        batch = [images.cuda(), labels.cuda()]
        g = gradient_computer.compute_per_sample_grad(batch)
        g = projector.project(g, model_id=0) / normalize_factor

        s, e = batch_i * dataloader.batch_size, (batch_i +
                                                 1) * dataloader.batch_size

        xtx_inv_matrix += ch.matmul(g.T, g * r[s:e, None])

    xtx_inv_matrix /= len(dataloader.dataset)
    xtx_inv_matrix += lam * ch.eye(proj_dim, device="cuda")
    print("XTX matrix:", xtx_inv_matrix)
    print("XTX[0]:", xtx_inv_matrix[0])
    xtx_inv_matrix = ch.linalg.inv(xtx_inv_matrix).cpu()
    return xtx_inv_matrix


def weighted_sum_gradients(
    loader,
    model,
    modelout_fn,
    alphas,
):
    """
    Aggregate the gradients across all examples, weighted by _alphas_.
    Used to compute the update in the original parameter space for TrakDualProjAlgo.
    """
    grads_sum = None
    i = 0
    for batch in loader:
        batch = [x.cuda() for x in batch]
        weights = alphas[i:i + len(batch[0])]
        outs = modelout_fn(model=model, batch=batch)
        weighted_output = outs @ weights
        grads = ch.autograd.grad(weighted_output, model.parameters())
        grads = vectorize(grads)
        grads_sum = grads if grads_sum is None else grads_sum + grads
        i += len(batch[0])
    return grads_sum
