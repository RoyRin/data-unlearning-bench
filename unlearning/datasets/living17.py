from collections import defaultdict
import shutil
import pandas as pd
import torch as ch
import torch.nn as nn
from threading import Lock
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from contextlib import nullcontext

# lock = Lock()

# from torchvision import models
import torchmetrics
import numpy as np
from tqdm import tqdm
from torchvision import models

import os
import time
import json
import copy
import importlib
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

try:
    from ffcv.pipeline.operation import Operation
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (
        ToTensor,
        ToDevice,
        Squeeze,
        NormalizeImage,
        RandomHorizontalFlip,
        ToTorchImage,
        Cutout,
        Convert,
    )
    from ffcv.fields.rgb_image import (
        CenterCropRGBImageDecoder,
        RandomResizedCropRGBImageDecoder,
        SimpleRGBImageDecoder,
    )
    from ffcv.fields.basics import IntDecoder
except:
    print("FFCV not installed")

#from ipdb import set_trace as bp
#from IPython import embed

from unlearning.datasets.cifar10 import IndexedDataset
from pathlib import Path 

BASE_DIR = Path("/mnt/xfs/datasets/")

if not  BASE_DIR.exists():
    BASE_DIR = Path("school2_path/unlearning/precomputed_models/LIVING17_dataset")
    
ROOT = BASE_DIR /"living17"
 
TRAIN_PATH = "living17_tr.beton"
VAL_PATH = "living17_val.beton"
TRAIN_TENSORS_PATH = "raw_tensors_tr_new.pt"
VAL_TENSORS_PATH = "raw_tensors_val_new.pt"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256
NUM_TRAIN = 88_400 // 2


def get_living17_dataloader(
    split="train",
    num_workers=0,
    batch_size=512,
    shuffle=False,
    indices=None,
    indexed=True,
    drop_last=False,
):
    num_workers = 0
    if split == "train_and_val":
        tr_loader = get_living17_dataloader(
            split="train",
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            indices=np.arange(44_200),
            indexed=indexed,
            drop_last=drop_last,
        )
        val_loader = get_living17_dataloader(
            split="val",
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            indices=None,
            indexed=indexed,
            drop_last=drop_last,
        )
        return ConcatLoader(tr_loader, val_loader)

    raw_tensors = ch.load(
        os.path.join(ROOT, TRAIN_TENSORS_PATH if split == "train" else VAL_TENSORS_PATH)
    )
    ds = ch.utils.data.TensorDataset(*raw_tensors)

    if indexed:
        ds = IndexedDataset(ds)

    if split == "train":
        if indices is None:
            indices = np.arange(NUM_TRAIN)

    ds = ch.utils.data.Subset(ds, indices) if indices is not None else ds
    loader = ch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return loader


class ConcatLoader:
    def __init__(self, *loaders):
        self.loaders = loaders

    def __iter__(self):
        self.iterators = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        try:
            batch = next(self.iterators[0])
        except StopIteration:
            try:
                batch = next(self.iterators[1])
            except StopIteration:
                raise StopIteration

        return batch

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)
