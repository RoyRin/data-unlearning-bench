# project deps
from unlearning import UNLEARNING_METHODS
from datasets import DATASETS
from models import MODELS
from models import get_urls_hf, load_from_url_hf
from eval import kl_from_margins

# third party deps
import numpy as np

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
    pretrain_margin_urls = get_urls_hf()
    all_unlearn_margins = np.stack([load_from_url_hf(url=m) for m in pretrain_margin_urls])
else:
    pretrain_model_urls = get_urls_hf(mode="models")
    all_pretrain_models_gen = (load_from_url_hf(url=m, mode="models") for m in pretrain_model_urls)
    # TODO: load dataloaders, run unlearning, get margins
    for model in all_pretrain_models_gen:

oracle_margin_urls = 0 # TODO: load once updated
all_oracle_margins = np.stack([load_from_url_hf(url=m) for m in oracle_margin_urls])
res = kl_from_margins(all_unlearn_margins, all_oracle_margins)
import pdb; pdb.set_trace()
