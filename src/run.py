# stdlib deps
import io
from pathlib import Path
import os
import argparse

# project deps
from eval import kl_from_margins, get_margins
from datasets import DATASETS
from paths import CHECKPOINTS_DIR, EVAL_DIR, FORGET_INDICES_DIR, MARGINS_DIR, ORACLES_DIR, check_hf_registry
from config import load_config, check_config
from unlearning import get_checkpoint_name, UNLEARNING_METHODS, OPTIMIZERS
from models import MODELS

# external deps
import torch
from tqdm import tqdm
import numpy as np

def load_try_local_then_huggingface(path, config, mode):
    assert mode in ["pretrain_checkpoints", "oracle_margins", "forget_indices"]
    if os.path.exists(path):
        print(f"Loading {mode} from server")
        contents = torch.load(path, map_location="cpu")
        # maybe add a mode to compute the ones that are missing
        if mode!="forget_indices":
            assert len(contents) >= config['N'], f"not enough {mode} in server {len(contents)}/{config['N']}"
    else:
        try:
            print(f"Loading {mode} from huggingface")
            contents = check_hf_registry(config, mode)
            if mode!="forget_indices":
                assert len(contents) >= config['N'], f"not enough {mode} in huggingface {len(contents)}/{config['N']}"
            print(f"Saving {mode} from huggingface")
            torch.save(contents, path)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
    return contents[:config['N']]

def load_oracle_margins(config):
    oracle_margins_dir = ORACLES_DIR / config['dataset'] / config['model']
    os.makedirs(oracle_margins_dir, exist_ok=True)
    oracle_margins_path = oracle_margins_dir / f"oracle_margins_{config['forget_id']}.pt"
    return load_try_local_then_huggingface(oracle_margins_path, config, "oracle_margins")

def load_pretrain_checkpoints(config):
    pretrain_checkpoints_dir = CHECKPOINTS_DIR / config['dataset'] / config['model']
    os.makedirs(pretrain_checkpoints_dir, exist_ok=True)
    pretrain_checkpoints_path = pretrain_checkpoints_dir / "pretrain_checkpoints.pt"
    return load_try_local_then_huggingface(pretrain_checkpoints_path, config, "pretrain_checkpoints")

def load_forget_indices(config):
    forget_indices_dir = FORGET_INDICES_DIR / config['dataset']
    os.makedirs(forget_indices_dir, exist_ok=True)
    forget_indices_path = forget_indices_dir / f"forget_indices_{config['forget_id']}.pt"
    return load_try_local_then_huggingface(forget_indices_path, config, "forget_indices")

def load_unlearning_margins(config):
    unlearning_margins_dir = MARGINS_DIR / config['dataset'] / config['model']
    os.makedirs(unlearning_margins_dir, exist_ok=True)
    unlearning_margins_path = unlearning_margins_dir / get_checkpoint_name(config, "margins")
    if os.path.exists(unlearning_margins_path):
        print("Loading unlearning margins from server")
        contents = torch.load(unlearning_margins_path, map_location="cpu")
        assert all(len(contents[k]) >= config['N'] for k in contents), f"not enough margins in server {len(contents)}/{config['N']}"
        return {k: contents[k][:config['N']] for k in contents}
    # if not precomputed, we compute them
    pretrain_models = load_pretrain_checkpoints(config)
    forget_indices = load_forget_indices(config)
    assert config['unlearning_method'] != 'scrub' or 'forget_batch_size' in config, f"forget_batch_size is required in config, current keys {config.keys()}"
    forget_loader = DATASETS[config['dataset']]['loader'](indices=forget_indices, batch_size=config['batch_size'] if "forget_batch_size" not in config else config['forget_batch_size'])
    retain_indices = [idx for idx in range(DATASETS[config['dataset']]['train_size']) if idx not in forget_indices]
    retain_loader = DATASETS[config['dataset']]['loader'](indices=retain_indices, batch_size=config['batch_size'])
    all_dataloader = DATASETS[config['dataset']]['loader'](split="all")
    epoch_margins = {ep: [] for ep in config['epochs']} 
    for model in tqdm(pretrain_models, desc="Getting unlearning margins"):
        epoch_models = UNLEARNING_METHODS[config['unlearning_method']](model, forget_loader, retain_loader, OPTIMIZERS[config['optimizer']], optimizer_kwargs={"lr": config["lr"]}, **config)
        for ep, unlearn_model in epoch_models.items():
            unlearn_margins = get_margins(unlearn_model, all_dataloader)
            epoch_margins[ep].append(unlearn_margins)
    epoch_margins = {ep: torch.stack(ep_marg) for ep, ep_marg in epoch_margins.items()}
    torch.save(epoch_margins, unlearning_margins_path)
    return epoch_margins 

def load_klom(config):
    klom_dir = EVAL_DIR / config['dataset'] / config['model']
    os.makedirs(klom_dir, exist_ok=True)
    klom_path = klom_dir / get_checkpoint_name(config, "klom")
    if os.path.exists(klom_path):
        print("Results for this config already exist")
        return torch.load(klom_path)
    oracle_margins = load_oracle_margins(config)
    epoch_margins = load_unlearning_margins(config)
    epoch_kloms = {}
    for ep, margins in epoch_margins.items():
        res = kl_from_margins(oracle_margins, margins)
        epoch_kloms[ep] = res
    torch.save(epoch_kloms, klom_path)
    return epoch_kloms

parser = argparse.ArgumentParser(description="Run gradient ascent unlearning with a config file.")
parser.add_argument(
            '--c',
            type=str,
            required=True,
            help='Name of the YAML configuration file.'
        )
args = parser.parse_args()
config = load_config(args.c)
check_config(config)
klom = load_klom(config)
