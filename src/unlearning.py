# stdlib dependencies
from copy import deepcopy
from typing import Dict, List

# third party deps
import torch
from torch.utils.data import DataLoader

def do_nothing(
    model,
    forget_loader,
    retain_loader,
    # to match basic signature
    optimizer_cls: torch.optim.Optimizer,
    optimizer_kwargs: Dict,
    epochs: int,
):
    return deepcopy(model)

def ascent_forget(
    model,
    forget_loader,
    retain_loader,
    optimizer_cls: torch.optim.Optimizer,
    optimizer_kwargs: Dict,
    epochs: List[int],
    loss_fn=torch.nn.functional.cross_entropy,
    device: str = "cuda",
):
    model = model.train().to(device)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    epoch_models = {}
    for it in range(1, max(epochs)+1):
        for idx, (x, y) in enumerate(forget_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()
        if it in epochs:
            # this could be made more efficient by computing margins on the fly
            # but looses simplicity and flexibility to compute other metrics
            # for resnet9 it takes around 0.07s to execute
            epoch_models[it] = deepcopy(model)
    return epoch_models

# TODO: later on if N > than the available margins then compute as many as necessary
def get_checkpoint_name(config, mode):
    assert mode in ["margins", "klom"]
    if config['unlearning_method'] == "do_nothing":
        name = "do_nothing__{mode}"
    if config['unlearning_method'] == "ascent_forget":
        name = f"ascent_forget__lr_{config['lr']}__ep_{config['epochs']}__f{config['forget_id']}__bs{config['batch_size']}__{mode}"
    else:
        raise NotImplementedError(f"config {config['unlearning_method']} not implemented")
    if mode == "klom":
        name += f"_{config['N']}"
    name += ".pt"
    return name

UNLEARNING_METHODS = {
    "do_nothing": do_nothing,
    "ascent_forget": ascent_forget,
}

OPTIMIZERS = {
    "sgd": torch.optim.SGD
}
