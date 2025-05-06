import requests
import numpy as np
import io
import torch
import os
from pathlib import Path
from torchvision import transforms, datasets
from tqdm import tqdm

tensor_from_url = lambda url, device="cpu", req_timeout_s = 30: torch.load(io.BytesIO(requests.get(url, timeout=req_timeout_s).content), map_location=device)

N = 100
DATA_DIR = "../../../data"
RECOMPUTE = True

def get_all_cifar_labels():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    val_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    # slow but could not access targets or labels directly
    return torch.tensor([label for _, label in dataset])

def margins_from_logits(logits, labels, device="cuda"):
    logits = logits.to(device)
    bindex = torch.arange(logits.shape[1]).to(logits.device, non_blocking=False)
    logits_correct = logits[:, bindex, labels]
    # Use clone to avoid modifying original logits if model is used elsewhere
    cloned_logits = logits #.clone()
    cloned_logits[:, bindex, labels] = torch.tensor(
        -torch.inf, device=cloned_logits.device, dtype=cloned_logits.dtype
    )
    return logits_correct - cloned_logits.logsumexp(dim=-1)

cifar_labels = get_all_cifar_labels()

for i in tqdm(range(1, 10), desc="getting margins for forget sets"):
    fgt_path = Path(f"./forget_set_{i}")
    logits_path = fgt_path / "all_logits.pt"
    if not os.path.exists(logits_path) or RECOMPUTE:
        train_contents = []
        val_contents = []
        for k in tqdm(range(N), desc="getting margins for models"):
            try:
                train_url = f"https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/oracles/CIFAR10/forget_set_{i}/{k}__train_logits__23.pt"
                train_contents.append(tensor_from_url(train_url))
                val_url = f"https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/oracles/CIFAR10/forget_set_{i}/{k}__val_logits_23.pt"
                val_contents.append(tensor_from_url(val_url))
            except Exception as e:
                print(e)
                print(i, k)
                import pdb; pdb.set_trace()
        train_contents = torch.stack(train_contents)
        val_contents = torch.stack(val_contents)
        all_contents = torch.concatenate([train_contents, val_contents], axis=1)
        assert all_contents.shape == (N, 60_000, 10)
        os.makedirs(fgt_path, exist_ok=True)
        torch.save(all_contents.cpu(), logits_path)
    else:
        all_contents = torch.load(logits_path, map_location="cpu")
    all_margins = margins_from_logits(all_contents, cifar_labels)
    margins_path = fgt_path / "all_margins.pt"
    torch.save(all_margins.cpu(), margins_path)

