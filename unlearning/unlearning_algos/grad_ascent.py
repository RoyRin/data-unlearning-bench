from typing import List, Callable
from copy import deepcopy
from tqdm.auto import tqdm
import torch as ch
import numpy as np


def gradient_ascent(model: ch.nn.Module,
                    train_dataloader: ch.utils.data.DataLoader = None,
                    forget_indices: List[int] = None,
                    forget_dataloader: ch.utils.data.Subset = None,
                    lr: float = 0.1,
                    num_iters: int = 25,
                    verbose: bool = True,
                    **kwargs) -> ch.nn.Module:
    print(f"training for - {num_iters}")
    assert (train_dataloader is not None and forget_indices is not None) or (
        forget_dataloader is not None
    ), "Either pass in the train dataloader and the indices to forget, or the forget dataloader"

    # make a copy of the model and keep the original model unchanged
    unlearned_model = deepcopy(model)

    unlearned_model = unlearned_model.cuda()
    unlearned_model.train()

    if forget_dataloader is None:
        forget_ds = ch.utils.data.Subset(train_dataloader.dataset,
                                         forget_indices)
        forget_dataloader = ch.utils.data.DataLoader(
            forget_ds, batch_size=train_dataloader.batch_size)

    if verbose:
        iterator = tqdm(range(num_iters), desc="Gradient Ascent")
    else:
        iterator = range(num_iters)

    for _ in iterator:
        for inputs, labels in forget_dataloader:
            unlearned_model.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = unlearned_model(inputs)
            loss = ch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            with ch.no_grad():
                for param in unlearned_model.parameters():
                    param += lr * param.grad

    return unlearned_model


def gradient_ascent_grid_search(
    model: ch.nn.Module,
    target_retain_accuracy: float,
    target_forget_accuracy: float,
    evaluate_fn: Callable,
    lr_grid: List[float] = [0.1, 0.05, 0.02, 0.01, 0.008, 0.005, 0.001],
    num_iters_grid: List[int] = [1, 2, 3, 4, 5, 10, 20, 50],
    train_dataloader: ch.utils.data.DataLoader = None,
    forget_indices: List[int] = None,
    forget_dataloader: ch.utils.data.Subset = None,
) -> ch.nn.Module:
    # grid search over learning rate and num_iters
    # to match target_retain_accuracy and target_forget_accuracy

    assert train_dataloader is not None and forget_indices is not None
    if forget_dataloader is None:
        forget_ds = ch.utils.data.Subset(train_dataloader.dataset,
                                         forget_indices)
        forget_dataloader = ch.utils.data.DataLoader(
            forget_ds, batch_size=train_dataloader.batch_size)

    retain_ds = ch.utils.data.Subset(
        train_dataloader.dataset,
        [i for i in range(len(train_dataloader)) if i not in forget_indices],
    )
    retain_dataloader = ch.utils.data.DataLoader(
        retain_ds, batch_size=train_dataloader.batch_size)

    best_unlearned_model = None
    best_retain_diff = np.inf
    best_forget_diff = np.inf
    best_lr = None
    best_num_iters = None

    for lr in tqdm(lr_grid, desc="Grid Search..."):
        for num_iters in num_iters_grid:
            unlearned_model = gradient_ascent(
                model,
                train_dataloader,
                forget_indices,
                forget_dataloader,
                lr=lr,
                num_iters=num_iters,
                verbose=False,
            )
            retain_accuracy = evaluate_fn(unlearned_model,
                                          retain_dataloader,
                                          verbose=False)
            forget_accuracy = evaluate_fn(unlearned_model,
                                          forget_dataloader,
                                          verbose=False)

            retain_diff = np.abs(retain_accuracy - target_retain_accuracy)
            forget_diff = np.abs(target_forget_accuracy - forget_accuracy)

            if retain_diff + forget_diff < best_retain_diff + best_forget_diff:
                best_unlearned_model = unlearned_model
                best_retain_diff = retain_diff
                best_forget_diff = forget_diff
                best_lr = lr
                best_num_iters = num_iters

    print(f"Best Retain Diff: {best_retain_diff}")
    print(f"Best Forget Diff: {best_forget_diff}")

    print(f"Best Learning Rate: {best_lr}")
    print(f"Best Num Iters: {best_num_iters}")
    print(f"")
    return best_unlearned_model
