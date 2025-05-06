# stdlib dependencies
from copy import deepcopy
from typing import Dict

# third party deps
import torch
from torch.utils.data import DataLoader 
from tqdm import tqdm

def do_nothing(model, forget_loader, retain_loader):
    return deepcopy(model)

def ascent_forget(model, forget_loader, retain_loader, optimizer_cls: torch.optim.Optimizer, optimizer_kwargs: Dict, n_iters: int, loss_fn = torch.nn.functional.cross_entropy, device: str = "cuda"):
        model = model.train().to(device)
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        for it in tqdm(range(n_iters), desc="gradient ascent"):
            for idx, (x, y) in enumerate(forget_loader):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                optimizer.zero_grad()
                (-loss).backward()
                optimizer.step()
        return model

UNLEARNING_METHODS = {
        "do_nothing": do_nothing,
        "ascent_forget": ascent_forget,
}
