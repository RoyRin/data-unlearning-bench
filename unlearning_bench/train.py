import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from torchvision import models as torchvision_models


from unlearning_bench.models import ResNet9
from unlearning_bench.datasets import get_cifar_dataloader


def get_logits(model, loader: torch.utils.data.DataLoader):
    training = model.training
    model.eval()
    all_logits = torch.zeros(len(loader.dataset), 10)
    batch_size = loader.batch_size
    with torch.no_grad():
        for i, (x, y, index) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                logits = model(x, index)
            all_logits[i * batch_size:(i + 1) * batch_size] = logits.cpu()

    model.train(training)
    return all_logits


def compute_margin(logit, true_label):
    logit_other = logit.clone()
    logit_other[true_label] = -np.inf

    return logit[true_label] - logit_other.logsumexp(dim=-1)


def get_margins(targets, logits):
    margins = [
        compute_margin(logit, target)
        for (logit, target) in zip(logits, targets)
    ]
    return np.array(margins)



def save_logits_and_margins(model, ckpt_dir, model_num, epoch):
    full_dataloader_unshuffled = get_cifar_dataloader(num_workers=2,
                                                      indexed=True)
    logits = get_logits(model, full_dataloader_unshuffled)
    targets = full_dataloader_unshuffled.dataset.original_dataset.targets
    margins = get_margins(targets, logits)

    logits_path = ckpt_dir / f"{model_num}__train_logits__{epoch}.pt"
    torch.save(logits, logits_path)
    margins_path = ckpt_dir / f"{model_num}__train_margins__{epoch}.npy"
    print(f"saved logits to {logits_path}")
    print(f"saved margins to {margins_path}")

    ####
    val_dataloader_unshuffled = get_cifar_dataloader(split="val",
                                                     num_workers=2,
                                                     indexed=True)
    val_targets = val_dataloader_unshuffled.dataset.original_dataset.targets
    val_logits = get_logits(model, val_dataloader_unshuffled)
    val_logits_path = ckpt_dir / f"{model_num}__val_logits__{epoch}.pt"
    torch.save(val_logits, val_logits_path)
    print(f"saved logits to {val_logits_path}")
    val_margins = get_margins(val_targets, val_logits)
    val_margins_path = ckpt_dir / f"{model_num}__val_margins__{epoch}.npy"
    np.save(val_margins_path, val_margins)
    print(f"saved margins to {val_margins_path}")


def train_cifar10(
    model,
    loader,
    checkpoint_epochs=[23],
    checkpoints_dir=Path("./data/cifar10_checkpoints"),
    overwrite=False,
    model_id=0,
    model_save_suffix="",
    lr=0.4,
    epochs=24,
    train_epochs=None,
    momentum=0.9,
    weight_decay=5e-4,
    lr_peak_epoch=5,
    label_smoothing=0.0,
    eval_loader=None,
    fixed_lr=False,
    final_lr_ratio=0.1,
    report_every=50,
    should_save_logits=False,
):
    print(f"params: ")
    for k, v in locals().items():
        if k not in ["model", "loader"]:
            print(f"{k} : {v}")
    # mkdir checkpoints_dir
    checkpoints_dir = Path(checkpoints_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # check if the last checkpoints file is there, if yes - skip
    if not overwrite:
        checkpoint_to_check = (
            checkpoints_dir /
            f"sd_{model_id}__{model_save_suffix}__epoch_{checkpoint_epochs[-1]}.pt"
        )
        # if file exists, ignore
        if Path(checkpoint_to_check).exists():
            print(f"checkpoint already exists: {checkpoint_to_check}")
            print("skipping and loading the model from the last checkpoint")
            model.load_state_dict(torch.load(checkpoint_to_check))
            return model

    opt = SGD(model.parameters(),
              lr=lr,
              momentum=momentum,
              weight_decay=weight_decay)
    iters_per_epoch = len(loader)
    # Cyclic LR with single triangle

    if fixed_lr:
        lr_schedule = np.ones((epochs + 1) * iters_per_epoch)
    else:

        # Generate updated learning rate schedule
        lr_schedule = np.interp(
            np.arange((epochs + 1) * iters_per_epoch),
            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
            [0, 1, 0],
        )

    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)

    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    if train_epochs is not None:
        epochs = train_epochs

    for ep in tqdm(range(epochs)):
        for ims, labs, index in loader:
            ims = ims.cuda()
            labs = labs.cuda()
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims, index)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

        if ep in checkpoint_epochs:
            print(f"saving model at epoch {ep}")
            torch.save(
                model.state_dict(),
                checkpoints_dir /
                f"sd_{model_id}__{model_save_suffix}__epoch_{ep}.pt",
            )
            if should_save_logits:
                save_logits_and_margins(model, checkpoints_dir, model_id, ep)

        if eval_loader is not None and ep % report_every == 0:
            test_acc = eval_cifar10(model, eval_loader, verbose=False)
            print(f"Epoch {ep} test acc: {test_acc * 100:.1f}%")
            print(f"learning rate : {scheduler.get_last_lr()[0]}")

    return model


def eval_cifar10(model, loader, verbose=True):
    is_training = model.training
    model.eval()

    with torch.no_grad():
        total_correct, total_num = 0.0, 0.0
        for ims, labs, index in loader:
            ims = ims.cuda()
            labs = labs.cuda()
            with autocast():
                out = model(ims, index)
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]

    accuracy = total_correct / total_num
    if verbose:
        print(f"Accuracy: {accuracy * 100:.1f}%")

    model.train(is_training)
    return accuracy

