# project deps
from unlearning import UNLEARNING_METHODS
from datasets import DATASETS
from models import MODELS
from models import load_from_url_hf, get_urls_hf
from eval import kl_from_margins, get_margins, to_np_cpu
from paths import FORGET_INDICES_DIR

# third party deps
import numpy as np
import torch
from tqdm import tqdm

N = 2
forget_id = 4
assert forget_id in [4], f"forget idx: {forget_id} not supported"
ul_method = "ascent_forget"
ul_method_kwargs = {
    "optimizer_cls": torch.optim.SGD,
    "optimizer_kwargs": {"lr": 1e-3},
    "n_iters": 3,
}
assert ul_method in UNLEARNING_METHODS, f"method: {method} not found in {UNLEARNING_METHODS}"
dataset = "cifar10"
assert dataset in DATASETS, f"dataset: {dataset} not found in {DATASETS}"
model = "resnet9"
assert model in MODELS, f"model: {model} not found in {MODELS}"
# reminder that pipeline not ready or tested
if model!="resnet9" or dataset!="cifar10":
    raise NotImplementedError("Current pipeline just supports resnet9 cifar10 since it relies on hf checkpoints")


if ul_method == "do_nothing":
    pretrain_margin_urls = [
        "https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/full_models/CIFAR10/train_margins_all.pt", "https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/full_models/CIFAR10/val_margins_all.pt"
    ]
    print("Loading pretraining margins")
    all_unlearn_margins = np.concatenate([load_from_url_hf(url=m) for m in pretrain_margin_urls], axis=1)
    all_unlearn_margins = all_unlearn_margins[:N, :]
else:
    print(FORGET_INDICES_DIR / dataset / f"forget_set_{forget_id}.npy")
    forget_indices = np.load(FORGET_INDICES_DIR / dataset / f"forget_set_{forget_id}.npy")
    forget_loader = DATASETS[dataset]["loader"](indices=forget_indices)
    retain_indices = [i for i in range(DATASETS[dataset]["train_size"]) if i not in forget_indices]
    retain_loader = DATASETS[dataset]["loader"](indices=retain_indices)
    # this could be more efficient if indexing the datasets before init dataloaders but its not bottl
    all_dataloader = DATASETS[dataset]["loader"](split="all")
    pretrain_model_urls = get_urls_hf(N=N, mode="models")
    all_pretrain_models_gen = (load_from_url_hf(url=m, mode="models") for m in pretrain_model_urls)
    all_margins = []
    for pretrain_model in tqdm(all_pretrain_models_gen, total=N, desc="unlearning"):
        unlearned_model = UNLEARNING_METHODS[ul_method](pretrain_model, forget_loader, retain_loader, **ul_method_kwargs)
        margins = get_margins(unlearned_model, all_dataloader)
        assert margins.shape[0] == DATASETS[dataset]["train_size"] + DATASETS[dataset]["val_size"]
        all_margins.append(to_np_cpu(margins))
    all_unlearn_margins = np.stack(all_margins)
try:
    assert all_unlearn_margins.shape[0] == N and all_unlearn_margins.shape[1] == 60_000
except:
    import pdb; pdb.set_trace()

# TODO get oracle margins
oracle_margin_urls = get_urls_hf(N, directory=f"oracles/CIFAR10/forget_set_{forget_id}")
# for now test on validation since the train split margins are not there yet --------
oracle_margin_urls = [oo for oo in oracle_margin_urls if "val" in oo]
all_oracle_margins = np.stack([load_from_url_hf(url=m) for m in oracle_margin_urls])
all_unlearn_margins = all_unlearn_margins[:, 50_000:]
# ------------
res = kl_from_margins(all_unlearn_margins, all_oracle_margins)
import pdb; pdb.set_trace()
