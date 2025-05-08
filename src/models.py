# stdlib deps
from pathlib import Path
import io
import os

# external deps
import torch
from huggingface_hub import HfApi, hf_hub_url
import requests
import numpy as np


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU(inplace=True),
    )


def ResNet9(num_classes=10, channels_last=True):
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2),
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


# full_models
#   ├── CIFAR10
#   ├── CIFAR10_augmented
#   └── LIVING17
# oracles
# └── CIFAR10
#     ├── forget_set_1
#     ├── forget_set_2
#     ├── forget_set_3
#     ├── forget_set_4
#     ├── forget_set_5
#     ├── forget_set_6
#     ├── forget_set_7
#     ├── forget_set_8
#     ├── forget_set_9
#     └── forget_set_10
# Each folder has
# train_logits_##.pt - logits at the end of training for model ## for validation points
# val_logits_##.pt - logits at the end of training for model ## for train points
# ##__val_margins_#.npy - margins of model ## at epoch # (this is derived from logits)
# ##__train_margins_#.npy - margins of model ## at epoch # (this is derived from logits)
# sd_##____epoch_#.pt - model ## checkpoint at epoch #


def hf_files_hardcoded(
    N: int,
    directory: str = "full_models/CIFAR10",
    mode: str = "margins",
    epoch: int = 23,
):
    assert directory == "full_models/CIFAR10" or directory.startswith(
        "oracles/CIFAR10"
    ), f"directory: {directory} not supported"
    assert mode in ["margins", "models"], f"Mode {mode} not supported"
    assert epoch == 23, f"epoch: {epoch} not supported"
    if mode == "margins":
        return [
            os.path.join(
                directory,
                f"{i}__{s}_margins_{epoch}.{'pt' if 'full_models' in directory else 'npy'}",
            )
            for s in ["train", "val"]
            for i in range(N)
        ]
    return [os.path.join(directory, f"sd_{i}____epoch_{epoch}.pt") for i in range(N)]


def get_urls_hf(
    N: int,
    source: str = "hf",
    directory: str = "full_models/CIFAR10",  # oracles/CIFAR10/forget_set_1
    repo_id: str = "royrin/KLOM-models",
    mode: str = "margins",
    rev: str = "main",
):
    assert source in ["hf"], f"Source {source} not supported"
    assert mode in ["margins", "models"], f"Mode {mode} not supported"
    mode_ids = {"margins": "margins", "models": "sd"}
    # initially we were doing this but after the latest updte its too slow so hardcoding for now
    # api = HfApi()
    # all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=rev)
    all_files = hf_files_hardcoded(N, directory, mode)
    # filtered_files = [
    #   f for f in all_files if f.startswith(f"{directory}/") and mode_ids[mode] in f ]
    return [
        hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
        for filename in all_files 
    ]


def load_from_url_hf(url: str, device: str = "cpu", mode: str = "margins"):
    assert mode in ["margins", "models"], f"Mode {mode} not supported"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    contents_buf = io.BytesIO(resp.content)
    if url.endswith(".pt"):
        tensor_contents = torch.load(contents_buf, map_location=device)
    else:
        assert url.endswith(
            ".npy"
        ), f"only .pt and .npy files supported but found {url.split('.')[-1]}"
        return np.load(contents_buf)
    if mode == "margins":
        return tensor_contents.cpu().numpy()
    model = ResNet9().to(device)
    model.load_state_dict(
        {
            k.removeprefix("model.").removeprefix("module."): v
            for k, v in tensor_contents.items()
        },
        strict=True,
    )
    return model

MODELS = {
    "resnet9": ResNet9,
}

if __name__ == "__main__":
    pass
