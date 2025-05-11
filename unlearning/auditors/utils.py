
from pathlib import Path
import numpy as np
import torch as ch
import yaml
import torchvision

from unlearning.models.resnet9 import ResNet9, WrappedModel
from unlearning.datasets.cifar10 import get_cifar_dataloader
from unlearning.training.train import train_cifar10
from unlearning.unlearning_algos.utils import get_margin
from unlearning.datasets import DATASET_TO_NUM_EPOCHS

from unlearning import BASE_DIR, LOG_DIR


class LMWrapper(ch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, index=None):
        input_ids, token_type_ids, attention_mask = x
        logits = self.model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask).logits
        return logits


class WrappedDataLoader:

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            # Assuming batch is a tuple (x, y, z, w)
            if len(batch) == 5:
                x, y, z, w, ind = batch
                yield (x, y, z), w, ind
            else:
                x, y, z, w = batch
                yield (x, y, z), w

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, attr):
        # Delegate attribute access to the original loader
        return getattr(self.loader, attr)


def load_model_from_dict(model, ckpt_path):
    sample_key = list(ch.load(ckpt_path).keys())[0]
    if sample_key[:5] == 'model':
        model.load_state_dict(ch.load(ckpt_path))
    else:
        model.model.load_state_dict(ch.load(ckpt_path))


def read_dict_as_str(d):
    s = ""
    for k, v in d.items():
        s += f"{k}={v}__"
    return s


def hash_a_str(string_):
    import hashlib
    return hashlib.md5(string_.encode()).hexdigest()


def make_results_dir(config):
    """
    takes in a config dict
    creates a name for the experiment
    creates a directory for the experiment with that name
    returns the directory path
    """
    RESULTS_BASE_DIR = Path(config["results_dir"])
    experiment_name = config["dataset"]
    forget_set_id = config.get("forget_set_id", '')
    experiment_name += f"__{forget_set_id}"
    experiment_name += f"__{config['unlearning_algo']}"

    dict_as_string = read_dict_as_str(config)
    experiment_name += f"__{hash_a_str(dict_as_string)}"

    RESULTS_DIR = RESULTS_BASE_DIR / experiment_name
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    # save config to results dir
    with open(RESULTS_DIR / "config.yaml", "w") as f:
        yaml.dump(config, f)
    return RESULTS_DIR


def model_factory(dataset, wrapped=True):
    """
    for now, let's tie the model to the dataset, so we have fewer moving pieces
    """
    if dataset.lower() == "cifar10":
        from unlearning.models.resnet9 import ResNet9

        return ResNet9(num_classes=10, wrapped=wrapped).cuda().eval()

    elif dataset.lower() == "cifar100":
        from unlearning.models.resnet9 import ResNet9

        return ResNet9(num_classes=100).cuda().eval()
    elif dataset.lower() == "living17":
        model = torchvision.models.resnet18()
        model.fc = ch.nn.Linear(512, 17)
        return WrappedModel(model.cuda().eval())
    elif dataset == 'QNLI':
        from unlearning.training.train_qnli import construct_model
        return LMWrapper(construct_model())
    else:
        raise NotImplementedError


def loader_factory(dataset,
                   split="train",
                   indices=None,
                   batch_size=256,
                   shuffle=False,
                   augment=False,
                   indexed=True,
                   drop_last=False):
    if dataset.lower() == "cifar10":
        from unlearning.datasets.cifar10 import get_cifar_dataloader
        return get_cifar_dataloader(split=split,
                                    num_workers=2,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    augment=augment,
                                    indices=indices,
                                    indexed=indexed,
                                    drop_last=drop_last)
    elif dataset.lower() == "living17":
        from unlearning.datasets.living17 import get_living17_dataloader
        return get_living17_dataloader(split=split,
                                       num_workers=2,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       indices=indices,
                                       indexed=indexed,
                                       drop_last=drop_last)
    elif dataset == 'QNLI':
        from unlearning.datasets.qnli import get_qnli_dataloader
        raw_loader = get_qnli_dataloader(split=split,
                                         num_workers=2,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         indices=indices,
                                         indexed=indexed,
                                         drop_last=drop_last)
        return WrappedDataLoader(raw_loader)
    else:
        raise NotImplementedError


def load_forget_set_indices(dataset,
                            forget_set_id,
                            DATA_DIR=None,
                            verbose=True):
    if DATA_DIR is None:
        DATA_DIR = BASE_DIR / "forget_set_inds"

    forget_set_path = DATA_DIR / dataset / f"forget_set_{forget_set_id}.npy"
    if verbose:
        print(f"forget_set_path - {forget_set_path}")

    forget_set_indices = np.load(forget_set_path)
    return forget_set_indices


import re


def sort_key(path):
    # Extract numerical parts and convert to integers
    numbers = re.findall(r'\d+', path.name)
    return [int(num) for num in numbers]


def get_full_model_paths(dataset, splits=["train", "val"]):
    DATA_DIR = BASE_DIR / "full_models" / dataset
    ep = DATASET_TO_NUM_EPOCHS[dataset.upper()]
    full_model_ckpt_paths = sorted(list(
        DATA_DIR.glob(f"sd_*_epoch_{ep-1}.pt")))

    # Sort paths using the custom key
    full_model_ckpt_paths = sorted(full_model_ckpt_paths, key=sort_key)

    # train and val
    full_model_logit_paths = [
        DATA_DIR / f"{split}_logits_all.pt" for split in splits
    ]
    full_model_margins_paths = [
        DATA_DIR / f"{split}_margins_all.pt" for split in splits
    ]
    return full_model_ckpt_paths, full_model_logit_paths, full_model_margins_paths


def get_oracle_paths(dataset, forget_set_id, splits=["train", "val"]):

    DATA_DIR = BASE_DIR / "oracles" / dataset / f"forget_set_{forget_set_id}"
    ep = DATASET_TO_NUM_EPOCHS[dataset.upper()]
    oracle_ckpt_0_path = DATA_DIR / f"sd_0____epoch_{ep-1}.pt"

    # train and val
    oracle_logit_paths = [
        DATA_DIR / f"{split}_logits_all.pt" for split in splits
    ]
    oracle_margins_paths = [
        DATA_DIR / f"{split}_margins_all.pt" for split in splits
    ]
    return oracle_ckpt_0_path, oracle_logit_paths, oracle_margins_paths



def train_one_cifar10_resnet9(
    retain_inds_path,
    checkpoints_dir,
    model_save_suffix,
    model_id=0,
    overwrite=False,
    eval_inputs_path="",
    eval_targets_path="",
    checkpoint_epochs=[23],
    lr=0.4,
    epochs=24,
    momentum=0.9,
    weight_decay=5e-4,
    lr_peak_epoch=5,
    label_smoothing=0.0,
):
    model = ResNet9(num_classes=10).cuda()
    if retain_inds_path != "":
        indices = np.load(retain_inds_path)
    else:
        indices = None

    loader = get_cifar_dataloader(indices=indices, shuffle=True)

    print("Train set size:", len(loader.dataset))

    print("checkpoint folder:", checkpoints_dir)

    model = train_cifar10(
        model=model,
        loader=loader,
        checkpoint_epochs=checkpoint_epochs,
        checkpoints_dir=checkpoints_dir,
        overwrite=overwrite,
        model_id=model_id,
        model_save_suffix=model_save_suffix,
        lr=lr,
        epochs=epochs,
        momentum=momentum,
        weight_decay=weight_decay,
        lr_peak_epoch=lr_peak_epoch,
        label_smoothing=label_smoothing,
    )

    if eval_inputs_path == "":
        print("Returning model")
        return model
    else:
        print("Returning margins")
        eval_inputs = ch.load(eval_inputs_path).cuda()
        eval_targets = ch.load(eval_targets_path).cuda()
        with ch.no_grad():
            margins = get_margin(model.eval(), eval_inputs, eval_targets).cpu()

        return margins
