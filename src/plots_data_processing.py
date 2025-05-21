from paths import EVAL_DIR, FORGET_INDICES_DIR
from ast import literal_eval
from datasets import DATASETS
import torch
import numpy as np
from pathlib import Path
import pandas as pd

selected_datasets = ["living17", "cifar10"]
model_dicts = {
    "cifar10": "resnet9",
    "living17": "resnet18"
}
forgetset_dicts = {
    "cifar10": [1,2,3,4,5,6,7,8,9],
    "living17": [1,2,3,4]
}
total_epochs_dicts = {
    "cifar10": 24,
    "living17": 25
}

def get_klom_stats(klom, key):
    return {f"{key}_klom_95": np.percentile(klom, 95), f"{key}_klom_99": np.percentile(klom, 99), f"{key}_klom_mean": np.mean(klom), f"{key}_klom_std": np.std(klom)}

def parse_method_fname(fname):
    if "do_nothing" in fname:
        method = "do_nothing"
        fset = int(fname.split("__")[1].split("_")[0][1])
        lr = None
        ep = None
        bs = None
        n = None
        ascent_epochs = None
    elif "ascent_forget" in fname:
        method, lr, ep, fset, bs, lat = fname.split("__")
        lr = float(lr.split("_")[1])
        ep = literal_eval(ep.split("_")[1])
        fset = int(fset[1])
        bs = int(bs.split("bs")[1])
        n = int(lat.split("_")[1].split(".")[0])
        ascent_epochs = None
    elif "scrub" in fname:
        method, lr, ep, fset, bs, ascent_epochs, lat = fname.split("__")
        lr = float(lr.split("_")[1])
        ep = literal_eval(ep.split("_")[1])
        fset = int(fset[1])
        bs = int(bs.split("bs")[1])
        n = int(lat.split("_")[1].split(".")[0])
        ascent_epochs = int(ascent_epochs.split("ascent_epochs")[1])
    return method, lr, ep, fset, bs, n, ascent_epochs

def parse_klom_fname(fname, KLOM_PATH, forget_indices, retain_indices, val_indices, TOTAL_EPOCHS, dataset_name):
    method, lr, ep, fset, bs, n, ascent_epochs = parse_method_fname(fname)
    res = torch.load(KLOM_PATH / fname)
    rows = []
    for e_id, klom in res.items():
        total_cost = (len(forget_indices[fset]) + len(retain_indices[fset]) + len(val_indices)) * TOTAL_EPOCHS
        if method == "ascent_forget":
            relative_cost = len(forget_indices[fset]) * e_id / total_cost
        elif method == "do_nothing":
            relative_cost = 0
        elif method == "scrub" or method == "scrubnew":
            relative_cost = (len(forget_indices[fset]) * ascent_epochs + len(retain_indices[fset]) * e_id) / total_cost
        else:
            raise NotImplementedError(f"Cost for {method} not implemented")
        assert method == "do_nothing" or e_id in ep, f"e_id {e_id} not in ep {ep}"
        try:
            retain_klom = get_klom_stats(klom[retain_indices[fset]], "retain")
            val_klom = get_klom_stats(klom[val_indices], "val")
            forget_klom = get_klom_stats(klom[forget_indices[fset]], "forget")
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
        rows.append(
            {
                "dataset": dataset_name,
                "method": method,
                "lr": lr,
                "epoch": e_id,
                "forget_id": fset,
                "batch_size": bs,
                "N": n,
                "ascent_epochs": ascent_epochs,
                "relative_cost": relative_cost,
                **retain_klom,
                **val_klom,
                **forget_klom,
            }
        )
    return rows

for dataset_name in selected_datasets:
    print("Loading data for", dataset_name)
    model_name = model_dicts[dataset_name]
    KLOM_PATH = EVAL_DIR / dataset_name / model_name
    FDIR = FORGET_INDICES_DIR / dataset_name
    forget_sets = forgetset_dicts[dataset_name]
    TOTAL_EPOCHS = total_epochs_dicts[dataset_name]
    forget_indices = {i: torch.load(FDIR / f"forget_indices_{i}.pt") for i in forget_sets}
    train_size = DATASETS[dataset_name]["train_size"]
    val_size = DATASETS[dataset_name]["val_size"]
    retain_indices = {i: [k for k in range(train_size) if k not in forget_indices[i]] for i in forget_sets}
    val_indices = [k for k in range(train_size, train_size+val_size)]
    contents = []
    for fname in KLOM_PATH.iterdir():
        contents.extend(parse_klom_fname(fname.name, KLOM_PATH, forget_indices, retain_indices, val_indices, TOTAL_EPOCHS, dataset_name))

tmp_dir = Path("./tmp2")
rows = []
for fname in tmp_dir.iterdir():
    if "kl" in fname.name:
        _, forget, epoch = fname.name.split("__")
        forget_id = int(forget[1])
        epoch = int(epoch.split(".")[0])
        kl = torch.load(fname)
        if forget_id not in forget_indices:
            continue
        rows.append({
            "dataset": dataset_name,
            "method": "retrain",
            "forget_id": forget_id,
            "epoch": epoch,
            "lr": None,
            "batch_size": None,
            "ascent_epochs": None,
            "N": None,
            "relative_cost": epoch / TOTAL_EPOCHS,
            **get_klom_stats(kl[retain_indices[forget_id]], "retain"),
            **get_klom_stats(kl[val_indices], "val"),
            **get_klom_stats(kl[forget_indices[forget_id]], "forget"),
        })
contents.extend(rows)
df = pd.DataFrame(contents)
df.to_csv("klom_results_supplementary.csv", index=False)