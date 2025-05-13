# stdlib deps
from pathlib import Path

# project deps
from paths import DATA_DIR

# external deps
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_cifar_dataloader(
    indices=None, split="train", shuffle=False, num_workers=8, batch_size: int = 256
):
    assert indices is None or not isinstance(
        indices, set
    ), "indices must be a sequence (list/tuple), not a set"
    assert split in [
        "train",
        "val",
        "all",
    ], "split must be one of ['train', 'val', 'all']"
    assert indices is None or split == "train", "indices must be None for split different than 'train'"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)),
        ]
    )
    if split == "all":
        train_dataset = datasets.CIFAR10(
            root=DATA_DIR, train=True, download=True, transform=transform
        )
        val_dataset = datasets.CIFAR10(
            root=DATA_DIR, train=False, download=True, transform=transform
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    else:
        dataset = datasets.CIFAR10(
            root=DATA_DIR, train=(split == "train"), download=True, transform=transform
        )
    if indices is not None:
        dataset = Subset(dataset, indices)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader

def get_living17_dataloader(
    indices=None, split="train", shuffle=False, num_workers=8, batch_size: int = 256
):
    # TODO: ADAPT THIS
    assert split in ["train", "val", "all"], "split must be one of ['train', 'val', 'all']"
    assert indices is None or not isinstance(
        indices, set
    ), "indices must be a sequence (list/tuple), not a set"
    assert indices is None or split != "train", "indices must be None for split='train'"
    train_tensors_filename = "raw_tensors_tr_new.pt"
    val_tensors_filename = "raw_tensors_val_new.pt"
    raw_tensors = torch.load(DATA_DIR / "living17" / (train_tensors_filename if split == "train" else val_tensors_filename))
    dataset = torch.utils.data.TensorDataset(*raw_tensors)
    if split == "train" and indices is not None:
        dataset = Subset(dataset, indices)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader

def get_living17_dataloader(
    indices=None, split="train", shuffle: bool = False, num_workers: int = 8, batch_size: int = 256
):
    assert split in {"train", "val", "all"}
    assert indices is None or not isinstance(indices, set)
    assert indices is None or split != "train"

    root = Path(DATA_DIR) / "living17"
    tr_file = root / "raw_tensors_tr_new.pt"
    va_file = root / "raw_tensors_val_new.pt"

    def load_tensor_dataset(path):
        tensors = torch.load(path)  # returns (images_uint8, labels_int64)
        return torch.utils.data.TensorDataset(*tensors)

    if split == "train":
        dataset = load_tensor_dataset(tr_file)
    elif split == "val":
        dataset = load_tensor_dataset(va_file)
    else:  # "all"
        dataset = torch.utils.data.ConcatDataset([load_tensor_dataset(tr_file), load_tensor_dataset(va_file)])

    if indices is not None:
        dataset = Subset(dataset, indices)

    # ImageNet normalisation that ResNet18 expects
    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    class ApplyNorm(torch.utils.data.Dataset):
        def __init__(self, base_ds):
            self.base = base_ds
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]          # x is uint8
            if x.ndim == 3 and x.shape[0] == 3:        # already C×H×W
                x = x.float().div_(255)
            else:                                       # H×W×C → C×H×W
                x = x.permute(2, 0, 1).float().div_(255)
            return imagenet_norm(x), y
    dataset = ApplyNorm(dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

DATASETS = {
    "cifar10": {
        "loader": get_cifar_dataloader,
        "train_size": 50_000,
        "val_size": 10_000,
    },
    "living17": {
        "loader": get_living17_dataloader,
        "train_size": 44_200,
        "val_size": 3400,
    }
}
