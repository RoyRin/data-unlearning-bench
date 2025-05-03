# Utils
import torch
import numpy as np

# Unlearning Methods

from unlearning.unlearning_benchmarks.clip_and_noise import CaN
from unlearning.unlearning_benchmarks.gradient_ascent import GA
from unlearning.unlearning_benchmarks.gradient_descent import GD
"""
    unlearned_model = unlearn_fn(
        model=model,
        train_dataloader=train_loader,
        forget_dataloader=None,
        forget_indices=forget_set_indices,
        **unlearning_kwargs,
    )
"""

from unlearning.auditors.utils import (
    model_factory,
    loader_factory,
    load_forget_set_indices,
    get_full_model_paths,
    get_oracle_paths,
)

from torch.utils.data import DataLoader, Subset


def retain_and_forget(train_dataloader, forget_indices, batch_size=None):
    """
    Splits the data loader into two: one containing the data points specified by forget_indices,
    and another containing the rest of the data.

    Parameters:
    - train_dataloader (DataLoader): The original DataLoader.
    - forget_indices (list of int): Indices of the data points to be separated.

    Returns:
    - DataLoader: Containing the data points specified by forget_indices.
    - DataLoader: Containing the rest of the data points.
    """
    # Access the dataset from the original DataLoader
    dataset = train_dataloader.dataset

    # Calculate the complement of the forget_indices
    total_indices = set(range(len(dataset)))
    forget_indices_set = set(forget_indices)
    retain_indices = list(total_indices - forget_indices_set)
    forget_indices = list(forget_indices_set)

    # Create subsets for the dataset
    forget_subset = Subset(dataset, forget_indices)
    retain_subset = Subset(dataset, retain_indices)
    if batch_size is None:
        batch_size = train_dataloader.batch_size
    # Create new DataLoaders from the subsets
    forget_dataloader = DataLoader(forget_subset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=train_dataloader.num_workers)
    retain_dataloader = DataLoader(retain_subset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=train_dataloader.num_workers)

    return retain_dataloader, forget_dataloader


def gradient_descent_wrapper(model,
                             train_dataloader,
                             forget_indices,
                             val_loader=None,
                             num_epochs=5,
                             learning_rate=0.001,
                             weight_decay=5e-4,
                             device='cpu',
                             batch_size=64,
                             dataset=None,
                             forget_batch_size=32,
                             retain_batch_size=64,
                             callback=None,
                             callback_epochs=None,
                             acc_save_path=None,
                             **kwargs):
    # get retain and forget from (train_loader + forget_indices)
    #retain_loader, forget_loader = retain_and_forget(train_dataloader,
    #                                                 forget_indices, #batch_size = batch_size)
    print(f"val_loader - {val_loader}!!!\n" * 4)
    N_points = len(train_dataloader.dataset)
    retain_inds = np.setdiff1d(np.arange(N_points), forget_indices)
    retain_loader = loader_factory(dataset,
                                   indices=retain_inds,
                                   indexed=True,
                                   batch_size=retain_batch_size,
                                   shuffle=True,
                                   drop_last=False)
    forget_loader = loader_factory(dataset,
                                   indices=forget_indices,
                                   indexed=True,
                                   batch_size=forget_batch_size,
                                   shuffle=True,
                                   drop_last=False)

    print(f"using parameters : {learning_rate}, {batch_size}, {num_epochs}")

    param_dict_gd = dict()
    param_dict_gd['lr'] = learning_rate
    param_dict_gd['epochs'] = num_epochs
    param_dict_gd['lr_scheduler'] = None
    param_dict_gd['momentum'] = 0.9
    #param_dict_gd['weight_decay'] = 5e-4
    param_dict_gd['noise_var'] = 0
    param_dict_gd['device'] = device
    param_dict_gd['attack_eval'] = False
    unlearner = GD(
        model,
        #config,
        lr=param_dict_gd['lr'],
        epochs=param_dict_gd['epochs'],
        lr_scheduler=param_dict_gd['lr_scheduler'],
        momentum=param_dict_gd['momentum'],
        weight_decay=weight_decay,  # param_dict_gd['weight_decay'],
        noise_var=param_dict_gd['noise_var'],
        device=param_dict_gd['device'],
        attack_eval=param_dict_gd['attack_eval'],
        callback=callback,
        callback_epochs=callback_epochs)
    eval_epochs = kwargs.get('eval_epochs', None)

    updated_model_ = unlearner.get_updated_model(
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        validation_loader=val_loader,
        #
        callback=callback,
        callback_epochs=callback_epochs,
        eval_epochs=eval_epochs,
    )

    updated_model, wall_clock_time, accs = updated_model_

    if acc_save_path is not None:
        print(f"saving to {acc_save_path}")
        np.save(acc_save_path, accs)
        # save to yaml

    print(f"Wall clock time: {wall_clock_time}")
    print(f"Accuracies: {accs}")
    return updated_model


def gradient_ascent_wrapper(
        model,
        train_dataloader,
        forget_indices,
        val_loader=None,
        learning_rate=1e-6,
        num_epochs=5,
        #epochs=10,
        device='cpu',
        batch_size=64,
        dataset=None,
        forget_batch_size=32,
        retain_batch_size=64,
        callback=None,
        callback_epochs=None,
        **kwargs):
    print(f"using parameters : {learning_rate}, {batch_size}, {num_epochs}")
    # get retain and forget from (train_loader + forget_indices)
    #retain_loader, forget_loader = retain_and_forget(train_dataloader,
    #                                                 forget_indices,
    #                                                 batch_size=batch_size)

    forget_loader = loader_factory(dataset,
                                   indices=forget_indices,
                                   indexed=True,
                                   batch_size=forget_batch_size,
                                   shuffle=True,
                                   drop_last=False)
    N_points = len(train_dataloader.dataset)
    retain_inds = np.setdiff1d(np.arange(N_points), forget_indices)
    retain_loader = loader_factory(dataset,
                                   indices=retain_inds,
                                   indexed=True,
                                   batch_size=retain_batch_size,
                                   shuffle=True,
                                   drop_last=False)

    param_dict_ga = dict()
    param_dict_ga['lr'] = learning_rate
    param_dict_ga['epochs'] = num_epochs
    param_dict_ga['lr_scheduler'] = None
    param_dict_ga['momentum'] = 0.9
    param_dict_ga['weight_decay'] = 5e-4
    param_dict_ga['device'] = device
    param_dict_ga['attack_eval'] = False

    unlearner = GA(
        model=model,
        #config=config,
        lr=param_dict_ga['lr'],
        #epochs=param_dict_ga['epochs'],
        lr_scheduler=param_dict_ga['lr_scheduler'],
        momentum=param_dict_ga['momentum'],
        weight_decay=param_dict_ga['weight_decay'],
        #attack_eval=param_dict_gd['attack_eval'], # NOTE: not sure what this entails
        epochs=num_epochs,
        device=param_dict_ga['device'],
        callback=callback,
        callback_epochs=callback_epochs)
    updated_model_ = unlearner.get_updated_model(
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        validation_loader=val_loader,
        callback=callback,
        callback_epochs=callback_epochs)

    updated_model, wall_clock_time, accs = updated_model_
    print(f"Wall clock time: {wall_clock_time}")
    print(f"Accuracies: {accs}")
    return updated_model


def clip_and_noise_wrapper(model,
                           retain_loader,
                           forget_loader,
                           val_loader,
                           param_dict_can=None,
                           device='cpu',
                           **kwargs):

    if param_dict_can is None:
        print(f'Using default parameters ...')
        param_dict_can = dict()
        param_dict_can['lr'] = 5e-5
        param_dict_can['epochs'] = 1
        param_dict_can['lr_scheduler'] = None
        param_dict_can['momentum'] = 0.9
        param_dict_can['weight_decay'] = 5e-4
        param_dict_can['loss_sign'] = 1
        param_dict_can['tau'] = 0.001
        param_dict_can['C'] = torch.inf()
        param_dict_can['device'] = device

    unlearner = CaN(model,
                    lr=param_dict_can['lr'],
                    epochs=param_dict_can['epochs'],
                    lr_scheduler=param_dict_can['lr_scheduler'],
                    momentum=param_dict_can['momentum'],
                    weight_decay=param_dict_can['weight_decay'],
                    device=param_dict_can['device'])
    updated_model = unlearner.get_updated_model(retain_loader=retain_loader,
                                                forget_loader=forget_loader,
                                                validation_loader=val_loader)

    return updated_model


def Unlearner(
        method: str,
        model,
        retain_loader,
        forget_loader,
        val_loader,
        #mia_forget_loader,
        #mia_test_loader,
        # config,
        param_dict_gd=None,
        param_dict_ga=None,
        param_dict_can=None,
        param_dict_lca=None,
        param_dict_knnSurrogate=None,
        param_dict_random=None,
        param_dict_lime=None,
        device='cpu'):

    if method == "GD":
        epochs = 10

        if param_dict_gd is None:
            print('Using default parameters ...')
            param_dict_gd = dict()
            param_dict_gd['lr'] = 5e-5
            param_dict_gd['epochs'] = 1
            param_dict_gd['lr_scheduler'] = None
            param_dict_gd['momentum'] = 0.9
            param_dict_gd['weight_decay'] = 5e-4
            param_dict_gd['noise_var'] = 0
            param_dict_gd['device'] = device
            param_dict_gd['attack_eval'] = False
        unlearner = GD(
            model,
            # config,
            lr=param_dict_gd['lr'],
            epochs=epochs,
            #epochs=param_dict_gd['epochs'],
            lr_scheduler=param_dict_gd['lr_scheduler'],
            momentum=param_dict_gd['momentum'],
            weight_decay=param_dict_gd['weight_decay'],
            noise_var=param_dict_gd['noise_var'],
            device=param_dict_gd['device'],
            attack_eval=param_dict_gd['attack_eval'])
        updated_model = unlearner.get_updated_model(
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            validation_loader=
            val_loader,  #mia_forget_loader=mia_forget_loader, mia_test_loader=mia_test_loader
        )

    elif method == "GA":
        epochs = 10
        if param_dict_ga is None:
            print(f'Using default parameters ...')
            param_dict_ga = dict()
            param_dict_ga['lr'] = 5e-5
            param_dict_ga['epochs'] = 1
            param_dict_ga['lr_scheduler'] = None
            param_dict_ga['momentum'] = 0.9
            param_dict_ga['weight_decay'] = 5e-4
            param_dict_ga['device'] = device
            param_dict_ga['attack_eval'] = False
        unlearner = GA(
            model=model,
            # config=config,
            lr=param_dict_ga['lr'],
            epochs=epochs,
            #epochs=param_dict_ga['epochs'],
            lr_scheduler=param_dict_ga['lr_scheduler'],
            momentum=param_dict_ga['momentum'],
            weight_decay=param_dict_ga['weight_decay'],
            attack_eval=param_dict_gd[
                'attack_eval'],  # NOTE: not sure what this entails
            device=param_dict_ga['device'])
        updated_model = unlearner.get_updated_model(
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            validation_loader=val_loader)

    elif method == "CaN":
        if param_dict_can is None:
            print(f'Using default parameters ...')
            param_dict_can = dict()
            param_dict_can['lr'] = 5e-5
            param_dict_can['epochs'] = 1
            param_dict_can['lr_scheduler'] = None
            param_dict_can['momentum'] = 0.9
            param_dict_can['weight_decay'] = 5e-4
            param_dict_can['loss_sign'] = 1
            param_dict_can['tau'] = 0.001
            param_dict_can['C'] = torch.inf()
            param_dict_can['device'] = device

        unlearner = CaN(model,
                        lr=param_dict_can['lr'],
                        epochs=param_dict_can['epochs'],
                        lr_scheduler=param_dict_can['lr_scheduler'],
                        momentum=param_dict_can['momentum'],
                        weight_decay=param_dict_can['weight_decay'],
                        device=param_dict_can['device'])
        updated_model = unlearner.get_updated_model(
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            validation_loader=val_loader)

    else:
        raise NotImplementedError("This method has not been implemented, yet.")

    return updated_model
