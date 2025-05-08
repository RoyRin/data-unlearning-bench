import torch
import io
import requests
from tqdm import tqdm
import numpy as np
from eval import kl_from_margins, get_margins
from models import ResNet9
import os
from pathlib import Path
from datasets import get_cifar_dataloader

def do_nothing_test(all_margins, pretrain_margins, forget_indices):
    for i in tqdm(range(1,9)):
        res = kl_from_margins(all_margins[i], pretrain_margins)
        res_fgt = res[forget_indices[i]]
        res_ret = res[[k for k in range(50_000) if k not in forget_indices[i]]]
        res_val = res[50_000:]
        print(f"FGT {i}:-------------")
        print(f"klom forget: {np.percentile(res_fgt, 95)}")
        print(f"klom retain: {np.percentile(res_ret, 95)}")
        print(f"klom val: {np.percentile(res_val, 95)}")

tmp_dir = Path("./tmp")
os.makedirs(tmp_dir, exist_ok=True)
load_tensor_from_hf = lambda url, timeout = 30, device = "cpu": torch.load(io.BytesIO(requests.get(url, timeout=timeout).content), map_location=device)
get_oracle_margins_url = lambda forget_id, mode : f"https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/oracles/CIFAR10/only_margins/forget_set_{forget_id}/{mode}_margins_all.pt"
load_npy_from_hf = lambda url: np.load(io.BytesIO(requests.get(url).content))
get_forget_set_url = lambda forget_id: f"https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/forget_set_indices/CIFAR10/forget_set_{forget_id}.npy"
pretrain_margins_train = load_tensor_from_hf("https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/full_models/CIFAR10/train_margins_all.pt")
pretrain_margins_val = load_tensor_from_hf("https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/full_models/CIFAR10/val_margins_all.pt")
pretrain_margins = torch.cat([pretrain_margins_train, pretrain_margins_val], dim=-1)
N = 100
forget_indices = {}
all_margins = {}
oracle_margins_path = tmp_dir / "oracle_margins.pt"
forget_indices_path = tmp_dir / "forget_indices.pt"
RECOMPUTE = False
if not os.path.exists(oracle_margins_path) or RECOMPUTE:
    for i in tqdm(range(1,9)):
        forget_indices[i] = load_npy_from_hf(url=get_forget_set_url(i))
        train_margins = load_tensor_from_hf(url=get_oracle_margins_url(i, "train"))[:N, :]
        val_margins = load_tensor_from_hf(url=get_oracle_margins_url(i, "val"))[:N, :]
        joined_margins = torch.cat([train_margins, val_margins], dim=-1)
        all_margins[i] = joined_margins
    torch.save(all_margins, oracle_margins_path)
    torch.save(forget_indices, forget_indices_path)
else:
    print("oracle margins loaded from cache")
    all_margins = torch.load(oracle_margins_path)
    forget_indices = torch.load(forget_indices_path)
pretrain_margins = pretrain_margins[:N, :] # (model_id, point_id)
# do_nothing_test(all_margins, pretrain_margins)

# RUN the test on the pretrain checkpoints to sanity check the get margins logic
def load_pretrain_model(model_id):
    pretrain_url = f"https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/full_models/CIFAR10/sd_{model_id}____epoch_23.pt"
    tensor_contents = load_tensor_from_hf(pretrain_url)
    model = ResNet9().to("cuda")
    model.load_state_dict(
        {
            k.removeprefix("model.").removeprefix("module."): v
            for k, v in tensor_contents.items()
        },
        strict=True,
    )
    return model
all_loader = get_cifar_dataloader(split="all")
computed_pretrain_margins = []
pretrained_margins_path = tmp_dir / "pretrained_margins.pt"
if not os.path.exists(pretrained_margins_path):
    for n in tqdm(range(N), desc="computing pretrain margins"):
        model = load_pretrain_model(n)
        model_margins = get_margins(model, all_loader)
        computed_pretrain_margins.append(model_margins)
    computed_pretrain_margins = torch.cat(computed_pretrain_margins).view(N, 60_000)
    torch.save(computed_pretrain_margins, pretrained_margins_path)
else:
    computed_pretrain_margins = torch.load(pretrained_margins_path)
assert computed_pretrain_margins.shape == (N, 60_000)
pretrain_models = {}
for n in tqdm(range(N), desc="computing pretrain margins"):
    model = load_pretrain_model(n)
    pretrain_models[n] = model
pretrain_models_path = tmp_dir / "pretrain_models.pt"
torch.save(pretrain_models, pretrain_models_path)
# do_nothing_test(all_margins, computed_pretrain_margins, forget_indices)
