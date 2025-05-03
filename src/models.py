# stdlib deps
from pathlib import Path
import io

# external deps
import torch
from huggingface_hub import HfApi, hf_hub_url
import requests

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

def get_urls_hf(
    source: str = "hf",
    directory: str = "full_models/CIFAR10",
    repo_id: str = "royrin/KLOM-models",
    mode: str = "margins",
    rev: str = "main",
):
    assert source in ["hf"], f"Source {source} not supported"
    assert mode in ["margins", "models"], f"Mode {mode} not supported"
    mode_ids = {"margins": "margins", "models": "sd"}
    api = HfApi()
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=rev)
    filtered_files = sorted(f for f in all_files if f.startswith(f"{directory}/") and f.endswith(".pt") and mode_ids[mode] in f)
    return [hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset") for filename in filtered_files]

def load_from_url_hf(
    url: str,
    device: str = "cpu",
    mode: str = "margins"
):
    assert mode in ["margins", "models"], f"Mode {mode} not supported"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    tensor_contents = torch.load(io.BytesIO(resp.content), map_location=device)
    if mode == "margins":
        return tensor_contents.cpu().numpy()
    model = ResNet9().to(device)
    model.load_state_dict(tensor_contents)
    return model

MODELS = {
    "resnet9": ResNet9,
}

if __name__=="__main__":
    print("Testing Loading")
    model_paths = get_urls_hf(mode="models")
    margin_paths = get_urls_hf()
    m1 = load_from_url_hf(model_paths[0], mode="models")
    m2 = load_from_url_hf(margin_paths[0])
    import pdb; pdb.set_trace()
