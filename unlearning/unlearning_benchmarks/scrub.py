import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from itertools import cycle
import os
import time
import math
import pandas as pd
#import wandb
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import copy
import torch.nn as nn
from torch.autograd import Variable
from typing import List
import itertools
from tqdm.autonotebook import tqdm
#from models import *
# from unlearning.unlearning_benchmarks.SCRUB import models, utils
#import models

#from logger import *
#import utils

from unlearning.unlearning_benchmarks.SCRUB.thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate

from unlearning.unlearning_benchmarks.SCRUB.thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
if False:
    from unlearning.unlearning_benchmarks.SCRUB.thirdparty.repdistiller.distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

from unlearning.unlearning_benchmarks.SCRUB.thirdparty.repdistiller.helper.loops import train_distill, train_distill_hide, train_distill_linear, train_vanilla, train_negrad, train_bcu, train_bcu_distill, validate

#from unlearning.unlearning_benchmarks.SCRUB.thirdparty.repdistiller.helper.pretrain import init
import numpy as np
import torch

from argparse import Namespace

args = Namespace()
import os
import shutil
import yaml
import numpy as np
import torch as ch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import logging
import pprint
from contextlib import redirect_stdout, redirect_stderr

from unlearning.auditors.utils import (
    model_factory,
    loader_factory,
    load_forget_set_indices,
    get_full_model_paths,
    get_oracle_paths,
)

#from unlearning.unlearning_algos.base_nn import NAME_TO_ALGO


def read_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(path, model_factory, ds_name):
    model = model_factory(ds_name)
    loaded_model = ch.load(path)
    first_key = list(loaded_model.keys())[0]
    if "model" in first_key:
        model.load_state_dict(loaded_model)

    else:
        # add ".model" to each key in k,vs
        loaded_model = {f"model.{k}": v for k, v in loaded_model.items()}
        model.load_state_dict(loaded_model)
    return model


def l2_difference(model1, model2):
    l2_diff = 0.0
    # Ensure both models are in the same state (e.g., both in eval mode)
    model1.eval()
    model2.eval()

    with torch.no_grad():
        for (param1, param2) in zip(model1.parameters(), model2.parameters()):
            # Check if both parameters are on the same device and are of the same shape
            if param1.device != param2.device or param1.shape != param2.shape:
                raise ValueError(
                    "Models have parameters on different devices or with different shapes"
                )

            # Compute the squared L2 norm of the difference between the parameters
            param_diff = param1 - param2
            l2_diff += torch.norm(param_diff, p=2).item()**2

    # Return the square root of the sum of squared differences
    return l2_diff**0.5


def scrub_loop(args,
               optimizer,
               model,
               criterion_cls,
               module_list,
               swa_model,
               criterion_list,
               retain_loader,
               forget_loader,
               callback=None,
               callback_epochs=None,
               verbose=False):

    acc_rs = []
    acc_fs = []
    report_every = 10
    maximize_loss = None

    print(f"total epochs : {args.sgda_epochs}")

    for epoch in range(1, args.sgda_epochs + 1):
        print(f"Epoch {epoch} ...")

        lr = sgda_adjust_learning_rate(epoch, args, optimizer)

        if verbose:
            print("==> scrub unlearning ...")
            print(f"validating - ")
        if epoch % report_every == 0:
            acc_r, acc5_r, loss_r = validate(retain_loader, model,
                                             criterion_cls, args, True)
            acc_f, acc5_f, loss_f = validate(forget_loader, model,
                                             criterion_cls, args, True)
            acc_rs.append(100 - acc_r.item())
            acc_fs.append(100 - acc_f.item())

        maximize_loss = 0

        if epoch <= args.msteps:
            if verbose:
                print(f"train distill 1")
            # maximize loss on the forget set
            maximize_loss = train_distill(epoch, forget_loader, module_list,
                                          swa_model, criterion_list, optimizer,
                                          args, "maximize")
            print(f"maximize loss (forget loss): {maximize_loss}")
        if verbose:
            print(f"train distill 2 :")
        # minimize loss on the retain set
        train_acc, train_loss = train_distill(epoch, retain_loader,
                                              module_list, swa_model,
                                              criterion_list, optimizer, args,
                                              "minimize")
        print(f"train loss : {train_loss}")
        if epoch >= args.sstart:
            print("update params")
            swa_model.update_parameters(model)

        if verbose:
            print(
                "maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}"
                .format(maximize_loss, train_loss, train_acc))
        student = module_list[0]
        teacher = module_list[-1]
        print(
            f"{epoch} -- difference between teacher and student - {l2_difference(student, teacher)}"
        )
        if callback is not None:
            if epoch in callback_epochs:
                callback(model, epoch)
    return model


def scrub_wrapper(
        model,
        train_dataloader,
        forget_indices,
        val_loader=None,
        num_epochs=5,
        learning_rate=0.01,
        device='cpu',
        forget_batch_size=16,  # taken from paper
        retain_batch_size=64,
        dataset=None,  # Name 
        gamma=0.99,
        alpha=0.1,
        beta=0.9999,
        drop_last=False,
        shuffle=True,
        maximization_epochs=5,
        callback=None,
        callback_epochs=[-1],
        **kwargs):
    print(f"scrub dataset - {dataset}")
    print(f"scurb drop last -  {drop_last}")
    args.optim = 'sgd'
    args.gamma = gamma  # .1, 2 # We find that SCRUB is not very sensitive to these hyper- parameters
    args.alpha = alpha  # 0.001  # .1, 2 # We find that SCRUB is not very sensitive to these hyper- parameters
    args.beta = beta  # 0.95, 0.999, 0.9999
    args.kd_T = 4  #
    # 3

    args.sgda_batch_size = retain_batch_size  # 64, 128
    args.del_batch_size = forget_batch_size  # 32, 64, 128
    args.msteps = maximization_epochs  # 3, 5
    args.sgda_epochs = num_epochs  # 5, 10
    args.sgda_learning_rate = learning_rate  # 0.1
    # 2 * 3 * 2 *2 = 24

    # 3 * 24 = 72 <<< SCRUB baselines.

    args.distill = 'kd'
    args.smoothing = 0.0
    args.clip = 0.2
    args.sstart = 10

    # learning_rate = 0.01
    args.lr_decay_epochs = [3, 5, 9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 5e-4
    args.sgda_momentum = 0.9
    print(f"learning_rate - {learning_rate}")

    original_model = copy.deepcopy(model)
    unlearned_model = copy.deepcopy(model)

    print(f"forget bs : {forget_batch_size}; retain bs : {retain_batch_size}")
    print(f"drop last - {drop_last}; shuffle - {shuffle}")
    forget_loader = loader_factory(dataset,
                                   indices=forget_indices,
                                   indexed=False,
                                   batch_size=forget_batch_size,
                                   shuffle=shuffle,
                                   drop_last=drop_last)
    N_points = len(train_dataloader.dataset)
    retain_inds = np.setdiff1d(np.arange(N_points), forget_indices)
    retain_loader = loader_factory(dataset,
                                   indices=retain_inds,
                                   indexed=False,
                                   batch_size=retain_batch_size,
                                   shuffle=shuffle,
                                   drop_last=drop_last)
    print(f"forgetting indices : {forget_indices} - {len(forget_indices)}")
    print(f"retain indices :  {len(retain_inds)}")

    module_list = nn.ModuleList([])
    module_list.append(unlearned_model)  # student.
    module_list.append(original_model)  # teacher.

    trainable_list = nn.ModuleList([])
    trainable_list.append(unlearned_model)
    """
    model_s = module_list[0]
    model_t = module_list[-1]
    """
    # optimizer
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(trainable_list.parameters(),
                                    lr=args.sgda_learning_rate,
                                    momentum=args.sgda_momentum,
                                    weight_decay=args.sgda_weight_decay)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(trainable_list.parameters(),
                                     lr=args.sgda_learning_rate,
                                     weight_decay=args.sgda_weight_decay)
    elif args.optim == "rmsp":
        optimizer = torch.optim.RMSprop(trainable_list.parameters(),
                                        lr=args.sgda_learning_rate,
                                        momentum=args.sgda_momentum,
                                        weight_decay=args.sgda_weight_decay)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(
        criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        return (1 - args.beta
                ) * averaged_model_parameter + args.beta * model_parameter

    swa_model = torch.optim.swa_utils.AveragedModel(unlearned_model,
                                                    avg_fn=avg_fn)

    # args

    unlearned_model = scrub_loop(args,
                                 optimizer,
                                 unlearned_model,
                                 criterion_cls,
                                 module_list,
                                 swa_model,
                                 criterion_list,
                                 retain_loader,
                                 forget_loader,
                                 callback=callback,
                                 callback_epochs=callback_epochs)

    return unlearned_model
