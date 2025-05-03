from typing import Tuple
import torch
import torchvision


class IndexedDataset(torch.utils.data.Dataset):

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        data, target = self.original_dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.original_dataset)


def get_cifar_dataset(split="train",
                      augment=False,
                      indices=None,
                      raw_imgs=False,
                      indexed=False):
    if augment:
        print("!" * 20)
        print("Augmentating!!!")
        print("!" * 20)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAffine(0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.201)),
        ])
    elif not raw_imgs:
        print("No augmentation")
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.201)),
        ])
    else:
        print("!" * 20)
        print("Raw images!!!")
        print("!" * 20)
        transforms = torchvision.transforms.Compose([])

    is_train = split == "train"
    dataset = torchvision.datasets.CIFAR10(
        root="/tmp/cifar_for_unlearning/",
        download=True,
        train=is_train,
        transform=transforms,
    )
    if indexed:
        dataset = IndexedDataset(dataset)
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)
    return dataset


def get_dataloader(dataset,
                   batch_size=256,
                   num_workers=8,
                   indices=None,
                   shuffle=False,
                   indexed=False):
    if indexed:
        dataset = IndexedDataset(dataset)
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(dataset=dataset,
                                       shuffle=shuffle,
                                       batch_size=batch_size,
                                       num_workers=num_workers)


def get_cifar_dataloader(batch_size=256,
                         num_workers=8,
                         split="train",
                         shuffle=False,
                         augment=False,
                         indices=None,
                         indexed=False,
                         drop_last=False):
    if split == "train_and_val":
        print("Making train-and-val dataset")
        dataset_1 = get_cifar_dataset(split="train",
                                      augment=augment,
                                      indexed=indexed)
        dataset_2 = get_cifar_dataset(split="val",
                                      augment=augment,
                                      indexed=indexed)
        dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2])
        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)
    else:
        dataset = get_cifar_dataset(split=split,
                                    augment=augment,
                                    indices=indices,
                                    indexed=indexed)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=shuffle,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         drop_last=drop_last)

    return loader


def get_cifar_forget_retain_loaders(
    original_dataset: torch.utils.data.Dataset,
    forget_set_type: str,  # "random" or "PC_{number}"
    forget_set_size: int = 100,
    shuffle: bool = True,
    num_workers: int = 8,
    batch_size: int = 256,
    seed: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    if forget_set_type == "random":
        forget_set, retain_set = torch.utils.data.random_split(
            original_dataset,
            [forget_set_size,
             len(original_dataset) - forget_set_size],
            generator=gen,
        )
        print("Forget set indices:", forget_set.indices)

    elif forget_set_type.startswith("PC_"):
        pc_num = int(forget_set_type.split("_")[1])
        pc_inds = torch.load(f"data/cifar10_PC_inds.pt")[pc_num]
        forget_set = torch.utils.data.Subset(
            original_dataset,
            pc_inds[:forget_set_size],
        )
        retain_set = torch.utils.data.Subset(
            original_dataset,
            pc_inds[forget_set_size:],
        )

    else:
        raise ValueError(f"Unknown forget_set_type: {forget_set_type}")

    # get dataloader
    forget_loader = torch.utils.data.DataLoader(
        dataset=forget_set,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    retain_loader = torch.utils.data.DataLoader(
        dataset=retain_set,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return forget_loader, retain_loader
