# Standard library imports
import os
import json
import copy
import time
import importlib
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple, Union
from uuid import uuid4
from contextlib import nullcontext

# Third-party imports
import numpy as np
import pandas as pd
import torch as ch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision import models
from torchvision.transforms import Normalize
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Subset

# Local imports
from unlearning.datasets.cifar10 import IndexedDataset

# Configure PyTorch settings
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)


# Constants and configurations can be added here

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256
NUM_TRAIN = 88_400 // 2


if False:
    from unlearning import LIVING17_ROOT

    TRAIN_TENSORS_PATH = "raw_tensors_tr_new.pt"
    VAL_TENSORS_PATH = "raw_tensors_val_new.pt"
    # example to show how to load the raw tensors
    raw_tensors = ch.load(
        os.path.join(LIVING17_ROOT, TRAIN_TENSORS_PATH if split == "train" else VAL_TENSORS_PATH)
    )

class ConcatLoader:
    """A loader that concatenates multiple data loaders."""
    
    def __init__(self, *loaders):
        """Initialize with multiple data loaders."""
        self.loaders = loaders

    def __iter__(self):
        """Initialize iterators for all loaders."""
        self.iterators = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        """Get next batch from the first loader, then second, etc."""
        try:
            batch = next(self.iterators[0])
        except StopIteration:
            try:
                batch = next(self.iterators[1])
            except StopIteration:
                raise StopIteration
        return batch

    def __len__(self):
        """Return total length of all loaders."""
        return sum(len(loader) for loader in self.loaders)

def get_living17_dataloader(
    raw_tensors: Tuple[ch.Tensor, ...],
    split: str = "train",
    num_workers: int = 0,
    batch_size: int = 512,
    shuffle: bool = False,
    indices: Optional[np.ndarray] = None,
    indexed: bool = True,
    drop_last: bool = False,
) -> Union[DataLoader, ConcatLoader]:
    """
    Create a DataLoader for the Living17 dataset.
    
    Args:
        raw_tensors: Tuple of tensors containing the data
        split: Dataset split ('train', 'val', or 'train_and_val')
        num_workers: Number of worker processes for data loading
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        indices: Optional indices to select a subset of the data
        indexed: Whether to wrap the dataset in an IndexedDataset
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        DataLoader or ConcatLoader for the specified split
    """
    if split == "train_and_val":
        tr_loader = get_living17_dataloader(
            raw_tensors,
            split="train",
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            indices=np.arange(44_200),
            indexed=indexed,
            drop_last=drop_last,
        )
        val_loader = get_living17_dataloader(
            raw_tensors,
            split="val",
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            indices=None,
            indexed=indexed,
            drop_last=drop_last,
        )
        return ConcatLoader(tr_loader, val_loader)

    ds = TensorDataset(*raw_tensors)

    if indexed:
        ds = IndexedDataset(ds)

    if split == "train" and indices is None:
        indices = np.arange(NUM_TRAIN)

    ds = Subset(ds, indices) if indices is not None else ds
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )