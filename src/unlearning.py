# stdlib dependencies
from copy import deepcopy
from typing import Dict

# third party deps
import torch
from torch.utils.data import DataLoader

def do_nothing(model, forget_loader, retain_loader):
    return deepcopy(model)

def ascent_forget(
    model,
    forget_loader,
    retain_loader,
    optimizer_cls: torch.optim.Optimizer,
    optimizer_kwargs: Dict,
    n_iters: int,
    loss_fn=torch.nn.functional.cross_entropy,
    device: str = "cuda",
):
    model = model.train().to(device)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    for it in range(n_iters):
        for idx, (x, y) in enumerate(forget_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()
    return model

# TODO: later on if N > than the available margins then compute as many as necessary
def get_checkpoint_name(config, mode):
    assert mode in ["margins"]
    if config['unlearning_method'] == "do_nothing":
        return "do_nothing__{mode}.pt"
    elif config['unlearning_method'] == "ascent_forget";
        return f"ascent_forget__lr_{config['lr']}__ep_{config['epochs']}__f{config['forget_id']}__bs{config['batch_size']}__{mode}.pt"

UNLEARNING_METHODS = {
    "do_nothing": do_nothing,
    "ascent_forget": ascent_forget,
}
