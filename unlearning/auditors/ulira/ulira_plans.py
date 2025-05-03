import numpy as np
from pathlib import Path

#import torch
from unlearning.auditors.utils import (
    loader_factory, )
from unlearning.auditors.utils import (
    model_factory,
    loader_factory,
    load_forget_set_indices,
    get_full_model_paths,
    get_oracle_paths,
    make_results_dir,
)

from unlearning import BASE_DIR, LOG_DIR, ULIRA_BASE_DIR


def get_ulira_training_masks(dataset_name):
    masks_path = ULIRA_BASE_DIR / "training_masks.npy"
    masks = np.load(masks_path)
    return masks


ds_name = "CIFAR10"
training_masks = get_ulira_training_masks(ds_name)


def generate_one_ulira_forget_mask(train_targets,
                                   training_mask,
                                   SEED=42,
                                   unlearning_forget_set_count=40,
                                   unlearning_forget_set_size=200,
                                   only_class_5=True,
                                   forget_set=None,
                                   class_5_range=1_000):
    """
    class_5_range is the number of points from class 5 we want to consider forgetting
    """
    print("fhi !")
    np.random.seed(SEED)
    N = len(train_targets)
    if forget_set is None:
        if only_class_5:
            class_5_mask = (train_targets == 5)
        else:
            print(f"sample from all classes!")
            class_5_mask = np.ones(N, dtype=bool)
        all_class_5_indices = class_5_mask.nonzero()[0]

        class_5_indices = np.random.choice(all_class_5_indices,
                                           class_5_range,
                                           replace=False)
        class_5_mask = np.zeros(N)
        class_5_mask[class_5_indices] = 1
        ####
    else:
        forget_set_inds = load_forget_set_indices("CIFAR10", forget_set)
        class_5_mask = np.zeros(N, dtype=bool)  # GET INDICES FROM FORGET SET
        class_5_mask[forget_set_inds] = 1
    print(f"here ")
    class_5_indices = class_5_mask.nonzero()[0]
    # pick 1000 points from class 5

    ulira_forget_mask = []

    indiv_training_mask_ = training_mask

    for _ in range(unlearning_forget_set_count):

        class_5_trained_on_mask = np.array(indiv_training_mask_ * class_5_mask,
                                           dtype=bool)
        class_5_trained_on_mask = np.array(class_5_trained_on_mask)

        class_5_trained_on_indices = class_5_trained_on_mask.nonzero()[0]
        if unlearning_forget_set_size < len(class_5_trained_on_indices):
            print(f"unlearning_forget_set_size - {unlearning_forget_set_size}")
            print(
                f" len(class_5_trained_on_indices) - {len(class_5_trained_on_indices)}"
            )
            indices = np.random.choice(class_5_trained_on_indices,
                                       unlearning_forget_set_size,
                                       replace=False)
        else:
            indices = class_5_trained_on_indices
        mask = np.zeros(N)
        print(f"indices {indices[:10]}")
        mask[indices] = 1
        ulira_forget_mask.append(mask)

    return np.array(ulira_forget_mask)


def get_ulira_forget_masks(dataset_name,
                           original_model_count=256,
                           class_5_range=1000,
                           unlearnings_per_model=40,
                           unlearning_forget_set_size=50,
                           overwrite=False,
                           only_class_5=True,
                           forget_set=None,
                           save=True):
    training_masks = get_ulira_training_masks(dataset_name)
    dirname = f"forget__{unlearning_forget_set_size}"
    if forget_set is not None:
        dirname = f"{dirname}__forget_set_{forget_set}"

    ulira_forget_mask_dir = ULIRA_BASE_DIR / dirname
    print(f"we would save to {ulira_forget_mask_dir}")
    # make dir
    ulira_forget_mask_dir.mkdir(exist_ok=True)

    train_loader = loader_factory(dataset_name, indexed=True)
    train_targets = np.array(train_loader.dataset.original_dataset.targets)

    ulira_forget_masks = []
    for model_i in range(original_model_count):
        path = ulira_forget_mask_dir / f"forget_masks__{model_i}.npy"
        if path.exists() and not overwrite:

            # load it
            print(f"loading {path}")
            ulira_forget_mask = np.load(path)
            print(f"ulira_forget_mask.shape {ulira_forget_mask.shape}")
            ulira_forget_masks.append(ulira_forget_mask)

        else:
            training_mask = training_masks[model_i]
            print(f"generating! {model_i} - {training_mask[:10]}")

            ulira_forget_mask = generate_one_ulira_forget_mask(
                train_targets,
                training_mask,
                class_5_range=class_5_range,
                unlearning_forget_set_count=unlearnings_per_model,
                unlearning_forget_set_size=unlearning_forget_set_size,
                only_class_5=only_class_5,
                forget_set=forget_set)
            # sums
            print(f"ulira_forget_mask- {sum(ulira_forget_mask[0])}")
            print(f"ulira_forget_mask- {sum(ulira_forget_mask[1])}")
            print(f"ulira_forget_mask- {sum(ulira_forget_mask[2])}")
            #raise ValueError("stop here")

            # save it
            if save:
                print(f"Saving to {path}")
                np.save(path, ulira_forget_mask)
            ulira_forget_masks.append(ulira_forget_mask)

        if model_i % 50 == 0:
            print(f"{model_i} / {original_model_count}")
            #priunt shape
            print(ulira_forget_mask.shape)
    return ulira_forget_masks


def load_ulira_forget_masks(original_model_count=256,
                            unlearning_forget_set_size=50,
                            forget_set=None):
    dirname = f"forget__{unlearning_forget_set_size}"
    if forget_set is not None:
        dirname = f"{dirname}__forget_set_{forget_set}"

    ulira_forget_mask_dir = ULIRA_BASE_DIR / dirname

    # make dir
    ulira_forget_mask_dir.mkdir(exist_ok=True)

    ulira_forget_masks = []
    for model_i in range(original_model_count):

        path = ulira_forget_mask_dir / f"forget_masks__{model_i}.npy"
        if not path.exists():
            # load it
            #print(f"loading {path}")
            print(f"path.name - {path}")
            raise Exception("path missing")
        ulira_forget_mask = np.load(path)
        #print(f"ulira_forget_mask.shape {ulira_forget_mask.shape}")
        ulira_forget_masks.append(ulira_forget_mask)

        if model_i % 100 == 0:
            print(f"{model_i} / {original_model_count}")
            #priunt shape
            print(ulira_forget_mask.shape)
    return ulira_forget_masks


def load_all_ulira_forget_masks(unlearning_forget_set_size=50,
                                forget_set_id=None):
    ulira_forget_mask_dir = ULIRA_BASE_DIR
    # make dir

    fn = f"all_forget_masks__{unlearning_forget_set_size}.npy"

    if forget_set_id is not None:
        fn = f"all_forget_masks__{unlearning_forget_set_size}__forget_set_{forget_set_id}.npy"

    path = ulira_forget_mask_dir / fn

    if not path.exists():
        # load it
        #print(f"loading {path}")
        raise Exception(f"path missing - {path}")
    return np.load(path)
