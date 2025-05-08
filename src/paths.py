# stdlib deps
from pathlib import Path
import os
import io

# third party deps
import requests
import torch
import numpy as np

# Paths with respect to files inside the repo src/ folder
SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
DATA_DIR = REPO_DIR / "data"
EVAL_DIR = DATA_DIR / "eval"
os.makedirs(EVAL_DIR, exist_ok=True)
CONFIG_DIR = REPO_DIR / "config"
os.makedirs(CONFIG_DIR, exist_ok=True)
MARGINS_DIR = DATA_DIR / "margins"
os.makedirs(MARGINS_DIR, exist_ok=True)
FORGET_INDICES_DIR = DATA_DIR / "forget_set_indices"
os.makedirs(FORGET_INDICES_DIR, exist_ok=True)
ORACLES_DIR = DATA_DIR / "oracles"
os.makedirs(ORACLES_DIR, exist_ok=True)

BASE_HF_REQ_URL = "https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/"
HF_REGISTRY = {
    "oracle_margins": {
        "cifar10": {
            "resnet9": lambda forget_id: [f"oracles/CIFAR10/only_margins/forget_set_{forget_id}/{mode}_margins_all.pt" for mode in ["train", "val"]],
        }
    }
}

def check_hf_registry(config, mode):
    assert mode in HF_REGISTRY, "{mode} not in {HF_REGISTRY.keys()}"
    assert config['dataset'] in HF_REGISTRY[mode], "dataset not in {HF_REGISTRY[mode].keys()}"
    assert config['model'] in HF_REGISTRY[mode][config['dataset']], "model not in {HF_REGISTRY[mode][config['dataset']]}"
    urls_to_check = HF_REGISTRY[mode][config['dataset']][config['model']](config['forget_id'])
    all_contents = []
    for url in urls_to_check:
        req_url  = BASE_HF_REQ_URL + url
        out = requests.get(req_url, timeout=30).content
        if url.endswith(".pt"):
            contents = torch.load(io.BytesIO(out), map_location="cpu")
        elif url.endswith(".npy"):
            contents = np.load(io.BytesIO(out))
        else:
            raise NotImplementedError(f"support for {req_url} termination not implemented")
        all_contents.append(contents)
    if torch.is_tensor(all_contents[0]):
        all_contents = torch.cat(all_contents, dim=-1)
    return all_contents
