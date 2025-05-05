# project deps
from unlearning import UNLEARNING_METHODS
from datasets import DATASETS
from models import MODELS
from models import load_from_url_hf
from eval import kl_from_margins
from paths import FORGET_INDICES_DIR

# third party deps
import numpy as np

N = 2
forget_idx = 5
assert forget_idx in [5], f"forget idx: {forget_idx} not supported"
ul_method = "do_nothing"
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
    all_unlearn_margins = np.stack([load_from_url_hf(url=m) for m in pretrain_margin_urls])
else:
    raise NotImplementedError("Method not implemented yet")
    # forget_indices = np.load(FORGET_INDICES_DIR / dataset / f"forget_set_{forget_idx}.npy")
    # pretrain_model_urls = get_urls_hf(mode="models")
    # all_pretrain_models_gen = (load_from_url_hf(url=m, mode="models") for m in pretrain_model_urls)
    # TODO: load dataloaders, run unlearning, get margins into all_unlearn_margins

assert all_unlearn_margins.shape[0] == N and all_unlearn_margins.shape[1] == 60_000

# TODO get oracle margins
oracle_margin_urls = 0 # TODO
all_oracle_margins = np.stack([load_from_url_hf(url=m) for m in oracle_margin_urls])
res = kl_from_margins(all_unlearn_margins, all_oracle_margins)
