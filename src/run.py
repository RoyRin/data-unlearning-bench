# project deps
from unlearning import UNLEARNING_METHODS
from datasets import DATASETS
from models import MODELS
from models import load_from_url_hf, get_urls_hf
from eval import kl_from_margins
from paths import FORGET_INDICES_DIR

# third party deps
import numpy as np

N = 2
forget_id = 4
assert forget_id in [4], f"forget idx: {forget_id} not supported"
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
    print("Loading pretraining margins")
    all_unlearn_margins = np.concatenate([load_from_url_hf(url=m) for m in pretrain_margin_urls], axis=1)
    all_unlearn_margins = all_unlearn_margins[:N, :]
else:
    raise NotImplementedError("Method not implemented yet")
    # forget_indices = np.load(FORGET_INDICES_DIR / dataset / f"forget_set_{forget_id}.npy")
    # pretrain_model_urls = get_urls_hf(mode="models")
    # all_pretrain_models_gen = (load_from_url_hf(url=m, mode="models") for m in pretrain_model_urls)
    # TODO: load dataloaders, run unlearning, get margins into all_unlearn_margins
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
