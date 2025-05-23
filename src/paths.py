# stdlib deps
from pathlib import Path
import os
import io

# project deps
from models import MODELS

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
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
FORGET_INDICES_DIR = DATA_DIR / "forget_set_indices"
os.makedirs(FORGET_INDICES_DIR, exist_ok=True)
ORACLES_DIR = DATA_DIR / "oracles"
os.makedirs(ORACLES_DIR, exist_ok=True)

BASE_HF_REQ_URL = "https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/"
HF_REGISTRY = {
    "oracle_margins": {
        "cifar10": {
            "resnet9": lambda config: [f"oracle_models/CIFAR10/only_margins/forget_set_{config['forget_id']}/{mode}_margins_all.pt" for mode in ["train", "val"]],
        },
        "living17": {
            "resnet18": lambda config: [f"oracle_models/LIVING17/only_margins/forget_set_{config['forget_id']}/{mode}_margins_all.pt" for mode in ["train", "val"]],
        },
    },
    "forget_indices": {
        "cifar10": lambda config: [f"forget_set_indices/CIFAR10/forget_set_{config['forget_id']}.npy"],
        "living17": lambda config: [f"forget_set_indices/LIVING17/forget_set_{config['forget_id']}.npy"],
    },
    "pretrain_checkpoints": {
        "cifar10": {
            "resnet9": lambda config: [f"full_models/CIFAR10/sd_{nn}____epoch_23.pt" for nn in range(config['N'])]
        },
        "living17": {
            "resnet18": lambda config: [f"full_models/LIVING17/sd_{nn}____epoch_24.pt" for nn in range(config['N'])]
        },
    },
}

def check_hf_registry(config, mode):
    assert mode in HF_REGISTRY, f"{mode} not in {HF_REGISTRY.keys()}"
    assert config['dataset'] in HF_REGISTRY[mode], "dataset not in {HF_REGISTRY[mode].keys()}"
    assert mode=="forget_indices" or config['model'] in HF_REGISTRY[mode][config['dataset']], "model not in {HF_REGISTRY[mode][config['dataset']]}"
    if mode == "forget_indices":
        urls_to_check = HF_REGISTRY[mode][config['dataset']](config)
    else:
        urls_to_check = HF_REGISTRY[mode][config['dataset']][config['model']](config)
    all_contents = []
    for url in urls_to_check:
        req_url  = BASE_HF_REQ_URL + url
        out = requests.get(req_url, timeout=30).content
        if url.endswith(".pt"):
            try:
                contents = torch.load(io.BytesIO(out), map_location="cpu")
            except:
                import pdb; pdb.set_trace()
            if "checkpoints" in mode:
                assert config['dataset'] in ["living17", "cifar10"], "dataset not in {['living17', 'cifar10']}"
                model = MODELS[config['model']](num_classes=17 if config['dataset'] == "living17" else 10)
                if config['model'] == "resnet9":
                    contents = { k.removeprefix("model.").removeprefix("module."): v
                    for k, v in contents.items()}
                try:
                    model.load_state_dict(
                            contents,
                            strict=True,
                        )
                    contents = model
                except:
                    import pdb; pdb.set_trace()
            elif mode == "oracle_margins":
                contents = contents[:config['N'], :]
        elif url.endswith(".npy"):
            contents = np.load(io.BytesIO(out))
            if mode == "forget_indices": return contents
        else:
            raise NotImplementedError(f"support for {req_url} termination not implemented")
        all_contents.append(contents)
    if torch.is_tensor(all_contents[0]):
        all_contents = torch.cat(all_contents, dim=-1)
    return all_contents

def get_living17_shapes(forget_id):
    import io; import requests;
    oracle_train_len = torch.load(io.BytesIO(requests.get(f"https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/oracle_models/LIVING17/only_margins/forget_set_{forget_id}/train_margins_all.pt").content)).shape[1]
    oracle_val_len = torch.load(io.BytesIO(requests.get(f"https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/oracle_models/LIVING17/only_margins/forget_set_{forget_id}/val_margins_all.pt").content)).shape[1]
    return oracle_train_len, oracle_val_len