import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
import itertools
import math
import os
import time
from collections import OrderedDict
from itertools import cycle
from typing import List

import matplotlib.pyplot as plt
import models
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.ticker import FuncFormatter
from models import *
from sklearn.linear_model import LogisticRegression
from thirdparty.repdistiller.distiller_zoo import (FSP, KDSVD, PKT, ABLoss,
                                                   Attention, Correlation,
                                                   DistillKL, FactorTransfer,
                                                   HintLoss, NSTLoss, RKDLoss,
                                                   Similarity, VIDLoss)
from thirdparty.repdistiller.helper.loops import (
    train_bcu, train_bcu_distill, train_distill, train_distill_hide,
    train_distill_linear, train_negrad, train_vanilla, validate)
from thirdparty.repdistiller.helper.pretrain import init
from thirdparty.repdistiller.helper.util import \
    adjust_learning_rate as sgda_adjust_learning_rate
if False:
    from torch.autograd import Variable
    from tqdm.autonotebook import tqdm
    import variational
    import wandb
    from logger import *

if __name__ == "__main__":
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(
        criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    if args.optim == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=args.sgda_learning_rate,
                              momentum=args.sgda_momentum,
                              weight_decay=args.sgda_weight_decay)
    elif args.optim == "adam":
        optimizer = optim.Adam(trainable_list.parameters(),
                               lr=args.sgda_learning_rate,
                               weight_decay=args.sgda_weight_decay)
    elif args.optim == "rmsp":
        optimizer = optim.RMSprop(trainable_list.parameters(),
                                  lr=args.sgda_learning_rate,
                                  momentum=args.sgda_momentum,
                                  weight_decay=args.sgda_weight_decay)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        swa_model.cuda()

    acc_rs = []
    acc_fs = []
    acc_ts = []
    for epoch in range(1, args.sgda_epochs + 1):

        lr = sgda_adjust_learning_rate(epoch, args, optimizer)

        print("==> scrub unlearning ...")

        acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls,
                                         args, True)
        acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls,
                                         args, True)
        acc_rs.append(100 - acc_r.item())
        acc_fs.append(100 - acc_f.item())

        maximize_loss = 0
        if epoch <= args.msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list,
                                          swa_model, criterion_list, optimizer,
                                          args, "maximize")
        train_acc, train_loss = train_distill(
            epoch,
            retain_loader,
            module_list,
            swa_model,
            criterion_list,
            optimizer,
            args,
            "minimize",
        )
        #(epoch, train_loader, module_list, swa_model, criterion_list, optimizer, opt, split, quiet=False)
        if epoch >= args.sstart:
            swa_model.update_parameters(model_s)

        print("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".
              format(maximize_loss, train_loss, train_acc))
