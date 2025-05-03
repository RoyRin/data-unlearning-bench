import torch
from tqdm import tqdm
from torch.cuda.amp import autocast


def to_cuda(x):
    if isinstance(x, (list, tuple)):
        return [to_cuda(xi) for xi in x]
    return x.cuda()


def get_accuracy(model, loader):
    tr = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, index in tqdm(loader):
            x = to_cuda(x)
            y = y.cuda()
            with autocast():
                logits = model(x, index)
            preds = logits.argmax(dim=1)
            correct += (preds == y).float().sum().item()
            total += y.size(0)
    model.train(tr)
    return correct / total * 100


def eval_accuracy(model, train_loader, val_loader, forget_loader):
    train_acc = get_accuracy(model, train_loader)
    val_acc = get_accuracy(model, val_loader)
    forget_acc = get_accuracy(model, forget_loader)
    return train_acc, val_acc, forget_acc
