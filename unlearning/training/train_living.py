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


from unlearning.auditors.utils import (
    load_forget_set_indices,
)

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    Squeeze,
    NormalizeImage,
    RandomHorizontalFlip,
    ToTorchImage,
)
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.fields.basics import IntDecoder
try:
    print("FFCV installed")
except Exception as e:
    print(e)
    print("FFCV not installed")



import fastargs

from unlearning import LIVING17_ROOT
print(fastargs.__file__)

Section("model", "model details").params(
    # arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
    arch=Param(And(str, OneOf(["resnet18"])), default="resnet18"),
    pretrained=Param(int, "is pretrained? (1/0)", default=0),
)

Section("data", "data related stuff").params(
    root=Param(
        str, "root directory of datasets", default=str(LIVING17_ROOT)
    ),
    train_dataset=Param(str, "file to use for training", default="living17_tr.beton"),
    val_dataset=Param(str, "file to use for validation", default="living17_val.beton"),
    num_workers=Param(int, "The number of workers", default=1),
    in_memory=Param(int, "does the dataset fit in memory? (1/0)", default=1),
    augment=Param(int, "augment vs not", default=0),
    rrc_scale_low=Param(float, "RRC scale low", default=0.08),
    rrc_scale_high=Param(float, "RRC scale high", default=1.0),
    # ood_dir=Param(str, 'where to load validation loaders', required=True),
    # ood_data=Param(str, 'which val datasets to evaluate', required=True),
)

Section("resolution", "resolution scheduling").params(
    min_res=Param(int, "the minimum (starting) resolution", default=160),
    max_res=Param(int, "the maximum (starting) resolution", default=160),
    end_ramp=Param(int, "when to stop interpolating resolution", default=0),
    start_ramp=Param(int, "when to start interpolating resolution", default=0),
)

Section("lr", "lr scheduling").params(
    step_ratio=Param(float, "learning rate step ratio", default=0.1),
    step_length=Param(int, "learning rate step length", default=30),
    lr_schedule_type=Param(OneOf(["step", "cyclic"]), default="cyclic"),
    lr=Param(float, "learning rate", default=0.6),
    lr_peak_epoch=Param(int, "Epoch at which LR peaks", default=12),
)

Section("logging", "how to log stuff").params(
    folder=Param(str, "log location", default="/tmp/"),
    log_level=Param(int, "0 if only at end 1 otherwise", default=1),
    save_model=Param(bool, "save percenter state_dict?", default=False),
    save_log=Param(bool, "log experiment data", default=0),
)

Section("validation", "Validation parameters stuff").params(
    batch_size=Param(int, "The batch size for validation", default=512),
    resolution=Param(int, "final resized validation image size", default=224),
    lr_tta=Param(int, "should do lr flipping/avging at test time", default=1),
)

Section("training", "training hyper param stuff").params(
    supercloud=Param(bool, "using supercloud?", default=True),
    eval_only=Param(int, "eval only?", default=0),
    batch_size=Param(int, "The batch size", default=1024),
    optimizer=Param(And(str, OneOf(["sgd"])), "The optimizer", default="sgd"),
    momentum=Param(float, "SGD momentum", default=0.9),
    weight_decay=Param(float, "weight decay", default=5e-4),
    epochs=Param(int, "number of epochs", default=25),
    label_smoothing=Param(float, "label smoothing parameter", default=0.1),
    use_blurpool=Param(int, "use blurpool?", default=0),
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256
NUM_TRAIN = 88_400 // 2


class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state("sum", default=ch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=ch.tensor(0), dist_reduce_fx="sum")

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count


@param("lr.lr")
@param("lr.step_ratio")
@param("lr.step_length")
@param("training.epochs")
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr


@param("lr.lr")
@param("training.epochs")
@param("lr.lr_peak_epoch")
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [0, lr, 0]
    return np.interp([epoch + 1], xs, ys)[0]


class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


class Trainer:
    def __init__(self, ckpt_dir, model_id, checkpoint_epochs, indices):
        self.all_params = get_current_config()
        self.device = "cuda"
        self.train_indices = indices

        self.train_loader = self.create_train_loader()
        self.train_loader_det = self.create_train_loader(shuffle=False)
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.ckpt_dir = ckpt_dir
        self.model_id = model_id
        self.checkpoint_epochs = checkpoint_epochs

        self.create_optimizer()
        self.initialize_logger()

    def model_weights(self):
        model = self.model
        buffs = parameters_to_vector(model.buffers()).cpu().numpy()
        params = parameters_to_vector(model.parameters()).detach().cpu().numpy()
        return np.concatenate([params, buffs]).astype("float16")

    @param("lr.lr_schedule_type")
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {"cyclic": get_cyclic_lr, "step": get_step_lr}

        return lr_schedules[lr_schedule_type](epoch)

    @param("training.momentum")
    @param("training.optimizer")
    @param("training.weight_decay")
    @param("training.label_smoothing")
    def create_optimizer(self, momentum, optimizer, weight_decay, label_smoothing):
        assert optimizer == "sgd"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ("bn" in k)]
        other_params = [v for k, v in all_params if not ("bn" in k)]
        param_groups = [
            {"params": bn_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": weight_decay},
        ]

        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param("data.root")
    @param("data.train_dataset")
    @param("data.num_workers")
    @param("training.batch_size")
    @param("data.augment")
    @param("data.rrc_scale_low")
    @param("data.rrc_scale_high")
    @param("data.in_memory")
    def create_train_loader(
        self,
        root,
        train_dataset,
        num_workers,
        batch_size,
        augment,
        rrc_scale_low,
        rrc_scale_high,
        in_memory,
        indices=None,
        shuffle=True,
    ):
        this_device = f"cuda"
        train_path = Path(os.path.join(root, train_dataset))
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        if augment:
            print("> RRC SCALE:", rrc_scale_low, flush=True)
            self.decoder = RandomResizedCropRGBImageDecoder(
                (res, res), scale=(rrc_scale_low, rrc_scale_high)
            )
            image_pipeline: List[Operation] = [
                self.decoder,
                RandomHorizontalFlip(),
                ToTensor(),
                ToDevice(ch.device(this_device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            ]
        else:
            self.decoder = CenterCropRGBImageDecoder(
                (res, res), ratio=DEFAULT_CROP_RATIO
            )
            image_pipeline: List[Operation] = [
                self.decoder,
                ToTensor(),
                ToDevice(ch.device(this_device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True),
        ]

        # if indices is None:
        #    all_indices = np.arange(88_400)
        #    all_indices = np.random.choice(all_indices, size=#(88_400 // 2,), replace=False)
        # else:
        #    all_indices = indices
        # indices = np.random.choice(all_indices, size=(m,), replace=False)

        order = OrderOption.QUASI_RANDOM if shuffle else OrderOption.SEQUENTIAL

        loader = Loader(
            train_path,
            batch_size=batch_size,
            indices=self.train_indices,  #if shuffle else half_indices,
            num_workers=num_workers,
            order=order,
            os_cache=in_memory,
            drop_last=shuffle,
            pipelines={
                "image": image_pipeline,
                "label": label_pipeline,
                "orig_label": label_pipeline,
            },
        )
        return loader

    @param("data.root")
    @param("data.val_dataset")
    @param("data.num_workers")
    @param("validation.resolution")
    @param("validation.batch_size")
    @param("data.in_memory")
    def create_val_loader(
        self, root, val_dataset, num_workers, resolution, batch_size, in_memory
    ):
        this_device = f"cuda"
        val_path = Path(os.path.join(root, val_dataset))
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True),
        ]

        loader = Loader(
            val_path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={
                "image": image_pipeline,
                "label": label_pipeline,
                "orig_label": label_pipeline,
            },
        )
        return loader

    @param("resolution.min_res")
    @param("resolution.max_res")
    @param("resolution.end_ramp")
    @param("resolution.start_ramp")
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param("training.epochs")
    @param("logging.log_level")
    @param("logging.save_model")
    def train(self, epochs, log_level, save_model):
        for epoch in range(epochs):
            # res = self.get_resolution(epoch)
            # self.decoder.output_size = (res, res)
            train_loss = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {"train_loss": train_loss, "epoch": epoch}

                self.eval_and_log(extra_dict)

            if epoch in self.checkpoint_epochs:
                print(f"saving model at epoch {epoch}")
                ch.save(
                    self.model.state_dict(),
                    Path(self.ckpt_dir) / f"sd_{self.model_id}____epoch_{epoch}.pt",
                )

        self.eval_and_log({"epoch": epoch})
        if save_model:
            ch.save(self.model.state_dict(), self.log_folder / "final_weights.pt")

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        self.log(
            dict(
                {
                    "current_lr": self.optimizer.param_groups[0]["lr"],
                    "top_1": stats["top_1"],
                    "top_5": stats["top_5"],
                    "val_time": val_time,
                },
                **extra_dict,
            )
        )

        return stats

    def model_weights(self):
        model = self.model
        buffs = parameters_to_vector(model.buffers()).cpu().numpy()
        params = parameters_to_vector(model.parameters()).detach().cpu().numpy()
        return np.concatenate([params, buffs]).astype("float16")

    @param("model.arch")
    def create_model_and_scaler(self, arch):
        scaler = GradScaler()
        model = getattr(models, arch)()
        model.fc = nn.Linear(512, 17)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.device)
        return model, scaler

    @param("logging.log_level")
    def train_loop(self, epoch, log_level):
        model = self.model
        model.train()
        losses = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target, _, _) in enumerate(iterator):
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = self.model(images)
                loss_train = self.loss(output, target)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end

            losses.append(loss_train.detach().item())
            ### Logging start
            if log_level > 0:

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ["ep", "iter", "shape", "lrs"]
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ["loss"]
                    values += [f"{loss_train.item():.3f}"]

                msg = ", ".join(f"{n}={v}" for n, v in zip(names, values))
                iterator.set_description(msg)
            ### Logging end

        return np.mean(losses)

    @param("validation.lr_tta")
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target, _, _ in tqdm(self.val_loader):
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ["top_1", "top_5"]:
                        self.val_meters[k](output, target)

                    loss_val = self.loss(output, target)
                    self.val_meters["loss"](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param("logging.folder")
    @param("logging.save_log")
    def initialize_logger(self, folder, save_log):
        self.val_meters = {
            "top_1": torchmetrics.Accuracy(task="multiclass", num_classes=17).to(
                self.device
            ),
            "top_5": torchmetrics.Accuracy(
                task="multiclass", top_k=5, num_classes=17
            ).to(self.device),
            "loss": MeanScalarMetric().to(self.device),
        }

        if save_log:
            uid = str(self.uid)
            assert uid != ""
            folder = (Path(folder) / uid).absolute()
            if folder.exists():
                shutil.rmtree(folder)

            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f"=> Logging in {self.log_folder}")
            params = {
                ".".join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / "params.json", "w+") as handle:
                json.dump(params, handle)

    @param("logging.save_log")
    def log(self, content, save_log):
        print(f"=> Log: {content}")
        cur_time = time.time()
        if save_log:
            with open(self.log_folder / "log", "a+") as fd:
                fd.write(
                    json.dumps(
                        {
                            "timestamp": cur_time,
                            "relative_time": cur_time - self.start_time,
                            **content,
                        }
                    )
                    + "\n"
                )
                fd.flush()

    def eval_and_save_logits(self, split):
        self.model.eval()
        loader = self.val_loader if split == "val" else self.train_loader_det
        logits = get_logits(self.model, loader)
        logits_path = self.ckpt_dir / f"{split}_logits_{self.model_id}.pt"
        ch.save(logits, logits_path)
        print(f"saved logits to {logits_path}")


def get_logits(model, loader):
    training = model.training
    model.eval()
    all_logits = []
    with ch.no_grad():
        with autocast():
            for i, (x, y, _, _) in enumerate(loader):
                x, y = x.cuda(), y.cuda()
                with ch.no_grad():
                    logits = model(x)
                all_logits.append(logits.cpu())

    model.train(training)
    return ch.cat(all_logits, 0)


def wrapper_for_train_living17_on_subset_submitit(
    forget_set_id: int,
    idx_start: int,
    n_models: int,
    ckpt_dir: str,
    should_save_train_logits: bool = False,
    should_save_val_logits: bool = False,
    model_id_offset: int = 0,
    checkpoint_epochs=[-1],
):
    """
    - masks_path gives the path to a np array of shape [K, train_set_size],
    where each row gives us a boolean mask of the samples that we want to train
    on (True for the included samples and False for the excluded samples)

    - idx_start and n_models tell us which rows of the masks to use for training;
    in particular, we will use the masks from masks_path[idx_start:idx_start +
    n_models] for training

    - model_id_offset is the offset to add to the model_id when saving the
    model, here only for a hacky use, feel free to ignore
    """
    
    
    train_indices = np.arange(44_200)
    print(f"original  indices - {len(train_indices)}")
    if forget_set_id is not None:
        forget_set_indices = load_forget_set_indices("LIVING17",
                                                 forget_set_id)
        
        
        indices = np.setdiff1d(train_indices, forget_set_indices)
        
    else:
        indices = train_indices

    print(f"len indices - {len(indices)}")
    print(f"computing from {idx_start} to {idx_start + n_models - 1}")
    for model_id in range(idx_start, idx_start + n_models):
        print(f"training - {model_id}")
        print(f"ckpt_dir-{ckpt_dir}")
        if (ckpt_dir / f"val_logits_{model_id + model_id_offset}.pt").exists():
            print(f"skipping model {model_id} because logits already exist")
            continue
        
        assert len(indices) > 0, "mask is empty"

        trainer = Trainer(
            ckpt_dir=ckpt_dir,
            model_id=model_id + model_id_offset,
            checkpoint_epochs=checkpoint_epochs,
            indices=indices,
        )
        t0 = time.time()
        trainer.train()
        t1 = time.time()
        print(f"Time elapsed to train: {t1 - t0:.5f}s")

        if should_save_train_logits:
            trainer.eval_and_save_logits("train")

        if should_save_val_logits:
            trainer.eval_and_save_logits("val")

    return 0  # return 0 if everything went well


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description="Fast cifar training")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate()  # Initialize config with defaults
    forget_set_id = config["unlearning.forget_set_id"]
    print(f"forget_set_id - {forget_set_id}")
    if forget_set_id is not None:
        SAVE_DIR = LIVING17_ORACLE_MODELS / f"forget_set_{forget_set_id}"
    else:
        SAVE_DIR = LIVING17_FULL_MODELS
    
    wrapper_for_train_living17_on_subset_submitit(
        forget_set_id=forget_set_id,
        idx_start=0,
        n_models=1,
        ckpt_dir=SAVE_DIR,
        should_save_train_logits=True,
        should_save_val_logits=True,
        model_id_offset=0,
    )
