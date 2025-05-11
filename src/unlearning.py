# stdlib dependencies
from copy import deepcopy
from typing import Dict, List

# third party deps
import torch
import numpy as np
import torch.nn.functional as F

def do_nothing(
    m,
    forget_loader,
    retain_loader,
    # to match basic signature
    optimizer_cls: torch.optim.Optimizer,
    optimizer_kwargs: Dict,
    epochs: List[int],
    **kwargs,
):
    return {1: deepcopy(m)}

def ascent_forget(
    m,
    forget_loader,
    retain_loader,
    optimizer_cls: torch.optim.Optimizer,
    optimizer_kwargs: Dict,
    epochs: List[int],
    loss_fn=torch.nn.functional.cross_entropy,
    device: str = "cuda",
    **kwargs,
):
    m = m.train().to(device)
    optimizer = optimizer_cls(m.parameters(), **optimizer_kwargs)
    epoch_models = {}
    for it in range(1, max(epochs)+1):
        for idx, (x, y) in enumerate(forget_loader):
            x, y = x.to(device), y.to(device)
            out = m(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()
        if it in epochs:
            # this could be made more efficient by computing margins on the fly
            # but looses simplicity and flexibility to compute other metrics
            # for resnet9 it takes around 0.07s to execute
            epoch_models[it] = deepcopy(m)
    return epoch_models

def scrub(
    m,
    forget_loader,
    retain_loader,
    optimizer_cls: torch.optim.Optimizer,
    optimizer_kwargs: Dict,
    epochs: List[int],
    loss_fn=torch.nn.functional.cross_entropy,
    device: str = "cuda",
    **kwargs,
):
    assert "ascent_epochs" in kwargs, "scrub requires ascent epochs in the config"
    ascent_epochs = kwargs["ascent_epochs"]
    m = m.train().to(device)
    optimizer = optimizer_cls(m.parameters(), **optimizer_kwargs)
    epoch_models = {}
    for it in range(1, max(epochs)+1):
        if it <= ascent_epochs:
            for idx, (x, y) in enumerate(forget_loader):
                x, y = x.to(device), y.to(device)
                out = m(x)
                loss = loss_fn(out, y)
                optimizer.zero_grad()
                (-loss).backward()
                optimizer.step()
        for idx, (x, y) in enumerate(retain_loader):
            x, y = x.to(device), y.to(device)
            out = m(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if it in epochs:
            # this could be made more efficient by computing margins on the fly
            # but looses simplicity and flexibility to compute other metrics
            # for resnet9 it takes around 0.07s to execute
            epoch_models[it] = deepcopy(m)
    return epoch_models

def adjust_learning_rate(epoch, lr_decay_epochs, lr_decay_rate, sgda_learning_rate, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = sgda_learning_rate
    if steps > 0:
        new_lr = sgda_learning_rate * (lr_decay_rate**steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr

def distill_kl_loss(y_s, y_t, T):
    """Distilling the Knowledge in a Neural Network"""
    p_s = F.log_softmax(y_s / T, dim=1)
    p_t = F.softmax(y_t / T, dim=1)
    loss = F.kl_div(p_s, p_t, reduction='batchmean') * (T**2)
    return loss

def scrub_new(
    m,
    forget_loader,
    retain_loader,
    optimizer_cls: torch.optim.Optimizer,
    optimizer_kwargs: Dict,
    epochs: List[int],
    loss_fn=torch.nn.functional.cross_entropy,
    device: str = "cuda",
    **kwargs,
):
    assert "ascent_epochs" in kwargs, "scrub requires ascent epochs in the config"
    cls_loss_fn = loss_fn
    gamma = 0.99
    alpha = 0.1
    lr_decay_epochs = [3, 5, 9]
    lr_decay_rate = 0.1
    kd_T = 4
    unlearn_model = deepcopy(m)
    m, unlearn_model = m.eval().to(device), unlearn_model.train().to(device)
    optimizer = optimizer_cls(m.parameters(), **optimizer_kwargs)
    epoch_models = {}
    for epoch in range(1, max(epochs)+1):
        lr = adjust_learning_rate(epoch, lr_decay_epochs, lr_decay_rate, optimizer_kwargs["lr"], optimizer)
        if epoch <= kwargs["ascent_epochs"]:
            for idx, (x, y) in enumerate(forget_loader):
                x, y = x.to(device), y.to(device)
                logit_s = unlearn_model(x)
                with torch.no_grad(): # already set to eval but just to be safe
                    logit_t = m(x)
                # max step on the KL loss
                loss = -distill_kl_loss(logit_s, logit_t, kd_T)
                # no param dist since args.smoothing was 0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        for idx, (x, y) in enumerate(retain_loader):
                x, y = x.to(device), y.to(device)
                logit_s = unlearn_model(x)
                with torch.no_grad(): # already set to eval but just to be safe
                    logit_t = m(x)
                # min step on gamma * cls_loss + alpha * kl_loss (since the kd term is set to zero)
                loss_cls = cls_loss_fn(logit_s, y)
                loss_div = distill_kl_loss(logit_s, logit_t, kd_T)
                loss = gamma * loss_cls + alpha * loss_div
                # no param dist since args.smoothing was 0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # no need to swa update since args.smoothing was 0

# TODO: later on if N > than the available margins then compute as many as necessary
def get_checkpoint_name(config, mode):
    assert mode in ["margins", "klom"]
    if config['unlearning_method'] == "do_nothing":
        if mode == "margins":
            name = f"do_nothing__{mode}"
        else:
            name = f"do_nothing__f{config['forget_id']}_{mode}"
    elif config['unlearning_method'] == "ascent_forget":
        name = f"ascent_forget__lr_{config['lr']}__ep_{config['epochs']}__f{config['forget_id']}__bs{config['batch_size']}__{mode}"
    elif config['unlearning_method'] == "scrub":
        name = f"scrub__lr_{config['lr']}__ep_{config['epochs']}__f{config['forget_id']}__bs{config['batch_size']}__ascent_epochs{config['ascent_epochs']}__{mode}"
    else:
        raise NotImplementedError(f"config {config['unlearning_method']} not implemented")
    if mode == "klom":
        name += f"_{config['N']}"
    name += ".pt"
    return name

UNLEARNING_METHODS = {
    "do_nothing": do_nothing,
    "ascent_forget": ascent_forget,
    "scrub": scrub,
}

OPTIMIZERS = {
    "sgd": torch.optim.SGD
}
