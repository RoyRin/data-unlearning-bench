from paths import BASE_HF_REQ_URL
import requests
import io
import torch
from tqdm import tqdm
from pathlib import Path
import os
from eval import get_margins_from_multimodel_logits, kl_from_margins
from paths import DATA_DIR, ORACLES_DIR, check_hf_registry
from torchvision import datasets

def load_try_local_then_huggingface(path, config, mode):
    assert mode in ["pretrain_checkpoints", "oracle_margins", "forget_indices"]
    if os.path.exists(path):
        print(f"Loading {mode} from server")
        contents = torch.load(path, map_location="cpu")
        # maybe add a mode to compute the ones that are missing
        if mode!="forget_indices":
            assert len(contents) >= config['N'], f"not enough {mode} in server {len(contents)}/{config['N']} for forget_set {config['forget_id']}"
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

forget_ids = [i for i in range(1,10)]
EARLY_EPOCHS = [10, 15]
N = 100
dataset_name = "cifar10"
model_name = "resnet9"
if dataset_name!="cifar10" or model_name!="resnet9":
    raise NotImplementedError("other datasets or models not implemented")
EXTENDED_REGISTRY = {
        "cifar10-resnet9-logits": lambda forget_id, n, epoch, mode: f"oracle_models/CIFAR10/forget_set_{forget_id}/{n}__{mode}_logits__{epoch}.pt",
}
def get_url(forget_id, n, epoch, mode):
    return EXTENDED_REGISTRY["cifar10-resnet9-logits"](forget_id, n, epoch, mode)

def get_tensor(url, timeout=30, device="cpu"):
    full_url = BASE_HF_REQ_URL + url
    out = requests.get(full_url, timeout=timeout).content
    return torch.load(io.BytesIO(out), map_location=device)

tmp_dir = Path("./tmp2")
os.makedirs(tmp_dir, exist_ok=True)
all_margins_path = tmp_dir / "all_margins.pt"
all_logits_path = tmp_dir / "all_logits.pt"
if not os.path.exists(all_margins_path):
    if not os.path.exists(all_logits_path):
        er = {i: {e: [] for e in EARLY_EPOCHS} for i in forget_ids}
        for forget_id in tqdm(forget_ids, desc="getting all logits"):
            logits_path = tmp_dir / f"all_logits_f{forget_id}.pt"
            if os.path.exists(logits_path):
                print(f"logits for forget set {forget_id} exist, loading")
                er[forget_id] = torch.load(logits_path)
            else:
                print(f"logits for forget set {forget_id} don't exist, computing")
                for ep in EARLY_EPOCHS:
                    for k in tqdm(range(N), desc=f"getting logits for forget_id {forget_id}, epoch {ep}"):
                        train_url = get_url(forget_id, k, ep, "train")
                        val_url = get_url(forget_id, k, ep, "val")
                        train_logits = get_tensor(train_url)
                        val_logits = get_tensor(val_url)
                        all_logits = torch.cat([train_logits, val_logits], dim=0).unsqueeze(0)
                        try:
                            assert all_logits.shape == (1, 60_000, 10)
                        except:
                            import pdb; pdb.set_trace()
                        if len(er[forget_id][ep]) == 0:
                            er[forget_id][ep] = all_logits
                        else:
                            er[forget_id][ep] = torch.cat([er[forget_id][ep], all_logits], dim=0)
                torch.save(er[forget_id], logits_path)
        torch.save(er, all_logits_path)
    all_logits = torch.load(all_logits_path)
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True
    )
    val_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True
    )
    dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    assert labels.shape[0] == 60_000 and len(labels.shape) == 1
    all_margins = {i: {e: [] for e in EARLY_EPOCHS} for i in forget_ids}
    for forget_id in tqdm(forget_ids, desc="getting all margins"):
        for ep in EARLY_EPOCHS:
            margins = get_margins_from_multimodel_logits(all_logits[forget_id][ep], labels)
            all_margins[forget_id][ep] = margins
    torch.save(all_margins, all_margins_path)
all_margins = torch.load(all_margins_path)
all_oracle_margins = {}
for i in tqdm(range(1, 10), desc="loading oracle margins"):
    all_oracle_margins[i] = load_oracle_margins({"forget_id": i, "dataset": dataset_name, "model": model_name, "N": N})

for i in tqdm(range(1, 10), desc="computing klom"):
    for ep in EARLY_EPOCHS:
        kl_path = tmp_dir / f"kl__f{i}__{ep}.pt"
        kl_res = kl_from_margins(all_margins[i][ep], all_oracle_margins[i])
        torch.save(kl_res, kl_path)
