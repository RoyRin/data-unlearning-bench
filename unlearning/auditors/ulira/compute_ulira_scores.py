import numpy as np
import torch as ch
from pathlib import Path
from unlearning.auditors.utils import (
    loader_factory, )
import yaml
import numpy as np
import torch as ch
import matplotlib.pyplot as plt
from pathlib import Path

from unlearning.auditors import ulira
from unlearning.auditors.eval_suite import load_model, read_yaml
from contextlib import redirect_stdout, redirect_stderr

from unlearning import BASE_DIR, LOG_DIR, ULIRA_BASE_DIR

import sys
try:
    index = int(sys.argv[1])
    print(f"index - {index}")
except:
    print(f"setting index to 0")
    index = 0
    print(f"index - {index}")

results_path = ULIRA_BASE_DIR

# running unlearnings vs oracles
SAVE_DIR = ULIRA_BASE_DIR / "ulira_compiled_results__drop_last"
SAVE_DIR.mkdir(exist_ok=True, parents=True)

dirs = list(results_path.glob("results__*"))

sub_dirs = []
# for each dir in dirs, get the subdirs
for dir_ in dirs:
    sub_dirs += list(dir_.glob("CIFAR*"))

for i, dir_ in enumerate(sub_dirs):
    print(f"{i} - {dir_.parent.name} / {dir_.name}")


def extract_details_from_dir(dir_name):
    # results__1000__resnet9_25_epochs__forget_set_3
    splits = dir_name.name.split("__")
    print(f"dirname  = {dir_name}")
    print(splits)
    unlearning_forget_set_size = int(splits[1])
    resnet9 = "resnet9" in splits[2]
    epochs_of_training = int(splits[2].split("_")[-2])
    forget_set_id = None
    if len(splits) > 3:
        forget_set_split = splits[4]
        print(f"splits - {splits}")
        if "forget_set" in forget_set_split:
            forget_set_id = int(forget_set_split.split("_")[-1])
    return unlearning_forget_set_size, resnet9, epochs_of_training, forget_set_id


def get_details_from_subdir(name):
    print(f"name-{name}")
    method_name = None

    if "GA_" in name:
        method_name = "ga"
    elif "GD_" in name:
        method_name = "gd"
    elif "dm_matching" in name:
        method_name = "dm_matching"
    elif "dm_direct" in name:
        method_name = "dm_direct"
    elif "scrub" in name:
        method_name = "scrub"
    else:
        raise ValueError("Unknown method")
    return method_name


##
sub_dir = sub_dirs[index]
##

unlearnings_per_model = 40
print(f"running with {sub_dir}")
tup = extract_details_from_dir(sub_dir.parent)
unlearning_forget_set_size, resnet9_, epochs_of_training, forget_set_id = tup
method_name = get_details_from_subdir(sub_dir.name)

print(f"sub_dir.parent - {sub_dir.parent}")
print(tup)

results_config = {
    "method": method_name,
    "unlearning_forget_set_size": unlearning_forget_set_size,
    "forget_set_id": forget_set_id,
    "resnet9": resnet9_,
    "epochs_of_training": epochs_of_training
}

import datetime

now = datetime.datetime.now()
#date_str = now.strftime("%Y-%m-%d__%H")
resnet18 = not resnet9_
use_school1_results = False

resnet9_str = "resnet9" if resnet9_ else "resnet18"

sub_dir_name = sub_dir.name

save_path = SAVE_DIR / f"ulira_results__{unlearning_forget_set_size}__{unlearning_forget_set_size}__{method_name}__{resnet9_str}__{epochs_of_training}__fs_{forget_set_id}__{sub_dir_name}.yaml"

if save_path.exists():
    raise ValueError(f"path exists - {save_path}")
print(f"we will save to : {save_path}")

########## above, setting up which ULIRA to call
########## below, ULIRA code
##########

#Load Oracles

# THIS IS WHERE WE SPECIFIY WHERE THE ORACLES ARE
dataset_name = "CIFAR10"

if resnet9_ and epochs_of_training == 25:
    oracle_model_dir_name = "resnet9_25_epochs"
elif resnet18 and epochs_of_training == 100:
    oracle_model_dir_name = "final_models"
else:
    raise ValueError("Unknown model")

###
## Load an Oracle
###

print(f"loading from {oracle_model_dir_name}")

if not use_school1_results:
    #results_dir = ULIRA_BASE_DIR / results_dirname
    train_f = ULIRA_BASE_DIR / oracle_model_dir_name  # / "train_margins_all.pt"
    val_f = ULIRA_BASE_DIR / oracle_model_dir_name  #/ "val_margins_all.pt"

else:
    #results_dir = ULIRA_BASE_DIR / "MIT_RESULTS" / results_dirname
    train_f = ULIRA_BASE_DIR / "MIT_RESULTS" / oracle_model_dir_name  # / "train_margins_all.pt"
    val_f = ULIRA_BASE_DIR / "MIT_RESULTS" / oracle_model_dir_name  #/ "val_margins_all.pt"

ds_name = "CIFAR10"
#forget_masks = ulira.get_ulira_forget_mask(ds_name, class_5_range=1000, overwrite=False )print(f"original training_masks- shape {training_masks.shape}")
# which points to forget
ulira_train_all_margins, ulira_val_all_margins = ulira.load_all_ulira_margins(
    ds_name, dir_name=oracle_model_dir_name)

all_oracle_margins_ulira = ch.cat(
    [ch.tensor(ulira_train_all_margins.T),
     ch.tensor(ulira_val_all_margins.T)]).T

all_oracle_margins_ulira.shape
ds_name = "CIFAR10"
drop_last_hack = (method_name == "scrub")
real_indices = np.arange(60_000)
training_indices = np.arange(50_000)
real_indices[:10]
training_indices.shape

###
## load forget masks
###
from unlearning.auditors.ulira_plans import get_ulira_forget_masks, load_ulira_forget_masks, load_all_ulira_forget_masks

max_unlearnings = 2000

forget_masks = load_all_ulira_forget_masks(
    unlearning_forget_set_size=unlearning_forget_set_size,
    forget_set_id=forget_set_id)
training_masks = ulira.get_ulira_training_masks()

forget_masks = forget_masks[:max_unlearnings]

# shapes
print(f"forget_masks- shape {forget_masks.shape}")
print(f"original training_masks- shape {training_masks.shape}")

max_forget_count = forget_masks.shape[0]
print(sum(forget_masks[0]))

###
## Training mask retiling
###

dont_repeat_them = False
print(f"do tiling for masks and for oracle margins")
if dont_repeat_them:
    repeated_training_masks = np.repeat(training_masks,
                                        unlearnings_per_model,
                                        axis=0)

    repeated_all_oracle_margins_ulira = all_oracle_margins_ulira.clone()
    repeated_all_oracle_margins_ulira = repeated_all_oracle_margins_ulira.repeat_interleave(
        unlearnings_per_model, dim=0)
    #
else:

    non_repeated_training_masks = training_masks.copy()
    non_repeated_all_oracle_margins_ulira = all_oracle_margins_ulira.clone()

    training_masks = np.repeat(training_masks, unlearnings_per_model,
                               axis=0)[:max_forget_count]
    all_oracle_margins_ulira = all_oracle_margins_ulira.repeat_interleave(
        unlearnings_per_model, dim=0)[:max_forget_count]
    #repeated_all_oracle_margins_ulira = all_oracle_margins_ulira.clone()

print(f"training_masks - shape {training_masks.shape}")

print(f"all_oracle_margins_ulira - shape {all_oracle_margins_ulira.shape}")

# confirm that we have the right number of forgotten points per unlearning
print(f"forget_set_id - {forget_set_id}")

if forget_set_id == None and sum(
        forget_masks[0]) != unlearning_forget_set_size:
    raise ValueError("forget set size is not correct")

#forget_masks_40 = load_ulira_forget_masks(original_model_count = 256)
#forget_masks = np.concatenate(forget_masks_40)
# save the forget masks
print(f"loadded - {forget_masks.shape}")


def get_forgettable_indices(forget_masks):
    """ extract out the forgettable indices from the forget_masks"""
    forgettable_indices = set([])
    #for forget_group in forget_masks:
    for row in forget_masks:
        l = set(row.nonzero()[0])
        forgettable_indices = forgettable_indices.union(l)
    # this value should be 1000
    #if len(forgettable_indices) != 1000:
    #    raise ValueError("forget_points should have 1000 points")
    # for each of these points, compute a ulira score using the margins
    forgettable_indices = list(forgettable_indices)
    return forgettable_indices


forgettable_indices = get_forgettable_indices(forget_masks)
print(len(forgettable_indices))

###
##  Extract Unlearning Margins
###


def sort_by_last_num(s):
    prefix = str(s).split(".")[0]

    last = str(prefix).split("__")[-1]
    return last
    #return int(last[:-len(".pt")])


#method_names = ["scrub", "dm_matching", "dm_direct", "GA_", "GD_"]
print(f"method_name {method_name}")
pts = sorted(list((sub_dir / "ulira").glob("*.pt")), key=sort_by_last_num)
for pt in pts:
    print(pt)
from yaml.loader import SafeLoader


class SafeLoaderWithFallback(SafeLoader):

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            try:
                key = self.construct_object(key_node, deep=deep)
                value = self.construct_object(value_node, deep=deep)
                mapping[key] = value
            except yaml.YAMLError as e:
                print(f"Skipping key-value pair {key_node.value}: {e}")
        return mapping


def load_yaml_skip_kv_errors(file_path):
    with open(file_path, 'r') as file:
        try:
            return yaml.load(file, Loader=SafeLoaderWithFallback)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None


config = load_yaml_skip_kv_errors(sub_dir / "config.yaml")
print(f"config is {config}")
accuracies_f = sorted(list((sub_dir / "ulira").glob("*.yaml")),
                      key=sort_by_last_num)


def read_yaml_files(file_paths):
    all_forget = []
    all_train = []
    all_val = []

    for file_path in file_paths:
        with open(file_path, 'r') as file_:
            data = yaml.safe_load(file_)
            for entry in data:
                all_forget.append(entry['forget'])
                all_train.append(entry['train'])
                all_val.append(entry['val'])

    forget_acc = np.mean(all_forget)
    train_acc = np.mean(all_train)
    val_acc = np.mean(all_val)

    return forget_acc, train_acc, val_acc


forget_acc, train_acc, val_acc = read_yaml_files(accuracies_f)

print(f"forget_acc - {forget_acc}")
print(f"train_acc - {train_acc}")
print(f"val_acc - {val_acc}")

try:

    ######
    for name, pts in [("dm_matching_pts", dm_matching_pts),
                      ("dm_direct_pts", dm_direct_pts), ("Scrub", scrub_pts)]:
        print(name)
        nums = []
        for p in pts:
            num = int(p.name.split("__")[-1][:-len(".pt")])
            nums.append(num)
        expected = list(range(0, 2000, 40))
        missing = set(expected) - set(nums)
        print(sorted(list(missing)))
        indices = sorted(np.array(list(missing)) / 20)
        print(f"indices {indices}")
    # scrub - need to do : 0 (0)
    # GD - 350 (7)

except Exception as e:
    print(e)


def get_num(p):
    name = p.name.split(".")[0]
    num = int(name.split("__")[-1])
    return num


def find_num(num, uliras_pts):
    for pt in uliras_pts:
        if num == get_num(pt):
            return pt
    return None


def load_and_merge_with_missing(uliras_pts,
                                incrementer=20,
                                num_unlearnings=2000):
    all_margins = []
    expected = list(range(0, num_unlearnings, incrementer))
    actually_trained = []
    for expected_num in expected:
        pt = find_num(expected_num, uliras_pts)
        if pt is not None:
            for i in range(incrementer):
                actually_trained.append(i + expected_num)
            margin = ch.load(pt)
            all_margins.append(margin)
    # repeat the marings by the number of unlearnings
    ret = ch.cat(all_margins, dim=0)
    actually_trained = np.array(actually_trained)

    return ret, actually_trained


def load_and_merge(uliras_pts):
    all_margins = []
    for pt in uliras_pts:
        margin = ch.load(pt)
        all_margins.append(margin)
    # repeat the marings by the number of unlearnings

    ret = ch.cat(all_margins, dim=0)
    #ret = ret.repeat_interleave(unlearnings_per_model, dim=0)
    return ret


print(f" doing {method_name}")
pts_of_interest = pts
print(f"len(pts) - {len(pts)}")
if len(pts) < 65:
    print(f"len(pts) - {len(pts)}")
    raise Exception("we need at least 65 of the ulira scores computed")

all_unlearned_margins_ulira, actually_trained = load_and_merge_with_missing(
    pts_of_interest, incrementer=20, num_unlearnings=2000)
print(
    f"all_unlearned_margins_ulira,shape - {all_unlearned_margins_ulira.shape}")

print(f"loading")
try:
    print(f"training_masks_copy.shape- {training_masks_copy.shape}")
    training_masks = training_masks_copy.copy()
    forget_masks = forget_masks_copy.copy()
except:
    print(
        f"you didn't copy the masks!! don't forget to restart the notebook from the top"
    )

training_masks = training_masks[actually_trained]
forget_masks = forget_masks[actually_trained]
#print(forget_masks.shape)
#print(training_masks.shape)

all_unlearned_margins_ulira.shape
all_oracle_margins_ulira.shape
#all_oracle_margins_ulira = all_oracle_margins_ulira[actually_trained]

all_oracle_margins_ulira = all_oracle_margins_ulira[actually_trained]
all_oracle_margins_ulira = all_oracle_margins_ulira.T[real_indices].T

all_oracle_margins_ulira.shape
print(
    f"all_unlearned_margins_ulira - shape {all_unlearned_margins_ulira.shape}")
print(f"all_oracle_margins_ulira - shape {all_oracle_margins_ulira.shape}")

training_masks = training_masks.T[training_indices].T
forget_masks = forget_masks.T[training_indices].T
print(f"forget_masks.shape {forget_masks.shape}")
print(f"training_masks.shape {training_masks.shape}")

if (forget_set_id == None) and sum(
        forget_masks[0]) != unlearning_forget_set_size:
    raise ValueError("forget set size is not correct")

results_config["config"] = config

###
## Actually running ULIRA
###


# get 200 points from class 5 that are not from the forgettable_points
#forgettable_indices
def get_test_points_for_ulira(forget_indices,
                              forgettable_indices,
                              num_test_points=200,
                              seed=42):
    """
    get 200 points forgettable points from class 5 that are not in the forget_indices
    """
    np.random.seed(seed)
    # copy forgettable_indices
    class_5 = forgettable_indices.copy()
    class_5 = np.array(list(set(class_5) - set(forget_indices)))
    np.random.shuffle(class_5)
    return list(class_5)[:num_test_points]


def gaussian_pdf(x, mean, std):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 *
                                                   ((x - mean) / std)**2)


def single_ulira(
        all_unlearned_margins: ch.Tensor,  # T x n
        all_test_margins: ch.Tensor,  # T x n
        unlearned_model_margin: float,
        sample_ind,
        threshold=0.5,
        plot=False,
        verbose=False,
        drop_outliers_num=0):
    """
    Compute the ULIRA (Unlearning Likelihood Ratio) results for each sample. for a single unlearned model.

    Note: 
        1: (x,y) is likely a member of training (unlearned model)
        0: (x,y) is likely a member of the oracle (oracle model)

    Args:
        all_unlearned_margins (ch.Tensor): Tensor of shape T x n containing the margins of all unlearned models.
        all_oracle_margins (ch.Tensor): Tensor of shape T x n containing the margins of all oracle models.
        hold_out_model (int): Index of the hold-out model.
        threshold (float, optional): Threshold value for the likelihood ratio. Defaults to 0.5.

    Returns:
        np.ndarray: Array of ULIRA results for each sample, where 1 indicates the likelihood ratio is above the threshold, and 0 otherwise.
    """
    #
    test_margins = all_test_margins[:, sample_ind].cpu().numpy()

    unlearned_margins = all_unlearned_margins[:, sample_ind].cpu().numpy()

    #
    #unlearned_model_margin = unlearned_margins[hold_out_model]
    #other_unlearned_models = np.delete(unlearned_margins, hold_out_model)

    # fit gaussians
    #drop_outliers_num = 4
    if False:  #drop_outliers_num > 0:
        test_margins = np.sort(test_margins)
        test_margins = test_margins[drop_outliers_num:-drop_outliers_num]
        unlearned_margins = np.sort(unlearned_margins)
        unlearned_margins = unlearned_margins[
            drop_outliers_num:-drop_outliers_num]

    def fit_gaussian(margins):
        # remove the top and bottom 3%

        mean, std = np.mean(margins), np.std(margins)
        return mean, std

    test_mean, test_std = fit_gaussian(test_margins)
    unlearned_mean, unlearned_std = fit_gaussian(unlearned_margins)

    #print(f"aaa ({np.round(oracle_mean,2)}, {np.round(oracle_std,2)}) vs ({np.round(unlearned_mean,2)}, {np.round(unlearned_std,2)})")
    # compute LIRA
    unlearned_model_margin = float(unlearned_model_margin)

    oracle_prob = gaussian_pdf(unlearned_model_margin, test_mean, test_std)
    unlearned_prob = gaussian_pdf(unlearned_model_margin, unlearned_mean,
                                  unlearned_std)

    likelihood_ratio = unlearned_prob / (unlearned_prob + oracle_prob)

    # print(f"likelihood_ratio - {likelihood_ratio} : type - {type(likelihood_ratio)}")

    likelihood_ratio = float(likelihood_ratio)
    if verbose:
        print(f"unlearned_margins shape - {unlearned_margins.shape}")
        print(f"oracle_margins shape - {test_margins.shape}")
        # print(f"oracle_margins - {test_margins}")

    if plot:
        # plot gaussians
        #plt.xlim(-20, 20)
        #x = np.linspace(-20, 20, 1000)
        min_ = min(np.min(test_margins), np.min(unlearned_margins))
        max_ = max(np.max(test_margins), np.max(unlearned_margins))
        x = np.linspace(min_, max_, 1000)
        plt.xlim(min_, max_)

        plt.hist(test_margins, bins=50, alpha=0.5, label="test", density=True)
        plt.hist(unlearned_margins,
                 bins=50,
                 alpha=0.5,
                 label="unlearned",
                 density=True)
        oracle_pdf = gaussian_pdf(x, test_mean, test_std)
        unlearned_pdf = gaussian_pdf(x, unlearned_mean, unlearned_std)
        num_pts = len(unlearned_margins)
        num_unique_pts = len(np.unique(unlearned_margins))

        plt.title(
            f"sample index: {sample_ind} | likelihood ratio: {round(likelihood_ratio,2):.2f} = {round(unlearned_prob,2)} / ({round(unlearned_prob,2)} + {round(oracle_prob,2)})\n num pts: {num_pts} | num unique pts: {num_unique_pts}"
        )
        plt.plot(x, oracle_pdf, label="oracle")
        plt.plot(x, unlearned_pdf, label="unlearned")
        plt.axvline(unlearned_model_margin, color="black", linestyle="--")
        plt.legend()
        plt.show()
    # save result
    return float(likelihood_ratio > threshold)


#####
def get_training_models_with_and_without(training_masks, index):
    """
    return indices of models with and without the point
    """
    m = training_masks[:, index].T

    with_index, without_index = np.where(m == 1)[0], np.where(m == 0)[0]
    return with_index, without_index


def get_models_with_pt_forgotten(masks, forget_mask, sample_index):
    """
    find the indices where the point is in the plan and in the forget mask
    """

    models_with = []
    models_inds_with, _ = get_training_models_with_and_without(
        masks, sample_index)

    for model_ind in models_inds_with:
        if forget_mask[model_ind, sample_index] == 1:
            models_with.append(model_ind)
    return np.array(models_with)


def get_models_with_pt_retained(masks, forget_mask, sample_index):
    """
    find the indices where the point is in the plan and in the forget mask
    """

    models_with = []
    models_inds_with, _ = get_training_models_with_and_without(
        masks, sample_index)

    for model_ind in models_inds_with:
        if forget_mask[model_ind, sample_index] == 0:
            models_with.append(model_ind)
    return np.array(models_with)


#####


def do_ulira_on_one_model(indices,
                          correct_ulira_label,
                          target_margins,
                          shadow_unlearning_margins,
                          shadow_oracle_margins,
                          forget_masks,
                          training_masks,
                          plot_index=-1,
                          verbose=False):
    """

    """
    correct_ulira = 0
    ulira_labels = []
    for i, sample_index in enumerate(indices):
        if i % 25 == 0:
            print(
                f"i - {i}/ {len(indices)} = {round(correct_ulira *100. / (i+1),2)}"
            )
        # get models with x forgotten and models that never saw x
        models_with_x_forgotten = get_models_with_pt_forgotten(
            training_masks, forget_masks, sample_index)
        _, models_without_x = get_training_models_with_and_without(
            training_masks, sample_index)

        num_models_with_x_forgotten = len(models_with_x_forgotten)
        num_models_without_x = len(models_without_x)
        if verbose:
            print(
                f"num_models_with_x_forgotten - {num_models_with_x_forgotten}")
            print(f"num_models_without_x - {num_models_without_x}")

        max_models = 300

        min_models = min(num_models_with_x_forgotten, num_models_without_x)
        min_models = min(max_models, min_models)
        #
        models_with_x_forgotten = models_with_x_forgotten[:min_models]
        models_without_x = models_without_x[:min_models]

        # the margin of the target model on sample index
        target_margin = target_margins[sample_index]

        # we want to do a lira test between unlearned models that saw x
        # and unlearned models that did not see x
        ulira_label = single_ulira(
            shadow_unlearning_margins[models_with_x_forgotten],
            shadow_oracle_margins[models_without_x],
            #shadow_unlearning_margins[models_without_x],
            target_margin,
            sample_index,
            plot=(i <= plot_index),
            verbose=verbose)

        ulira_labels.append(ulira_label)
        if ulira_label == correct_ulira_label:
            correct_ulira += 1
        plot_it_all = False  # correct_ulira_label==0 #True
        if plot_it_all:
            if ulira_label == correct_ulira_label:
                print(
                    f"plotting good boy - expecting {correct_ulira_label}, where 1 = forget, 0 = retain"
                )

            else:
                # plot it :

                print(
                    f"plotting wrong boy - expecting {correct_ulira_label}, where 1 = forget, 0 = retain"
                )

            ulira_label = single_ulira(
                shadow_unlearning_margins[models_with_x_forgotten],
                shadow_unlearning_margins[models_without_x],
                target_margin,
                sample_index,
                plot=True,
                verbose=verbose)

    return correct_ulira


###
##  ULIRA main function
###


def ulira_paper(all_unlearned_margins_ulira: ch.Tensor,
                all_oracle_margins_ulira: ch.Tensor,
                forget_masks,
                training_masks,
                plot_index=-1,
                verbose=False,
                max_models=None):
    """
    forgettable_points_ind, # which of the forgettable points we are looking at 

    """
    #sample_ind = forgettable_points[forgettable_points_ind]
    #oracle_count, N = training_masks.shape
    unlearning_count = all_unlearned_margins_ulira.shape[0]  # 1000
    target_count = unlearning_count // 2  # 500
    target_count = 50
    #shadow_count = int(0.1* unlearning_count)  # 500
    original_working = False  # True
    if original_working:
        # target models are first 500 unlearned models
        target_unlearning_margins = all_unlearned_margins_ulira[:target_count]
        target_oracle_margins = all_oracle_margins_ulira[:target_count]
        target_forget_masks = forget_masks[:target_count]
        target_training_masks = training_masks[:target_count]
        ###

        shadow_unlearning_margins = all_unlearned_margins_ulira[
            target_count:unlearning_count]
        shadow_oracle_margins = all_oracle_margins_ulira[
            target_count:unlearning_count]
        shadow_forget_masks = forget_masks[target_count:unlearning_count]
        shadow_training_masks = training_masks[target_count:unlearning_count]

    else:
        #choose a random set of target models
        target_indices = np.random.choice(unlearning_count,
                                          target_count,
                                          replace=False)
        target_unlearning_margins = all_unlearned_margins_ulira[target_indices]
        target_oracle_margins = all_oracle_margins_ulira[target_indices]
        target_forget_masks = forget_masks[target_indices]
        target_training_masks = training_masks[target_indices]

        # shadow models are the rest
        shadow_indices = np.array(
            list(set(range(unlearning_count)) - set(target_indices)))
        shadow_unlearning_margins = all_unlearned_margins_ulira[shadow_indices]
        shadow_oracle_margins = all_oracle_margins_ulira[shadow_indices]
        shadow_forget_masks = forget_masks[shadow_indices]
        shadow_training_masks = training_masks[shadow_indices]

    # get the set of indices that models consider forgetting
    forgettable_indices = list(sorted(get_forgettable_indices(forget_masks)))

    model_scores = []
    model_count = len(target_unlearning_margins)
    if max_models is not None:
        model_count = max_models

    for model_i in range(model_count):
        print(f"model {model_i}")
        target_margins = target_unlearning_margins[model_i]

        # indices that model_i forgot
        #forget_indices = target_forget_masks[model_i].nonzero()[0]
        target_forget_set_mask = target_forget_masks[model_i]
        target_training_set_mask = target_training_masks[model_i]

        forget_mask_ = np.array(target_forget_set_mask *
                                target_training_set_mask,
                                dtype=bool)
        forget_indices = forget_mask_.nonzero()[0]
        forget_pt_count = len(forget_indices)

        # test indices =
        test_indices = get_test_points_for_ulira(
            forget_indices,
            forgettable_indices,
            num_test_points=forget_pt_count,
            seed=42)

        # take 200 points that the model never saw
        #target_training_set_indices = target_training_set_mask.nonzero()[0]
        #print(target_training_set_mask)
        #target_unseen_training_mask = 1- target_training_set_mask
        seen_training_indices = target_training_set_mask.nonzero()[0]
        #unseen_training_indices = target_unseen_training_mask.nonzero()[0]
        # test indiices is forgettable indices that have have never been seen by the model
        test_indices = list(
            set(forgettable_indices) - set(seen_training_indices))
        test_indices = sorted(list(test_indices))[:forget_pt_count]
        if verbose:
            print(f"we want a 0")

        if model_i < 2:
            plot_index = plot_index
        else:
            plot_index = -1

        # NOTE -  we don't want oracles. we want shadow models that never saw the point
        #
        test_ulira_correct = do_ulira_on_one_model(test_indices,
                                                   0,
                                                   target_margins,
                                                   shadow_unlearning_margins,
                                                   shadow_oracle_margins,
                                                   shadow_forget_masks,
                                                   shadow_training_masks,
                                                   plot_index=plot_index,
                                                   verbose=verbose)
        if verbose:
            print(f"we want a 1")
        # forget indices
        forget_ulira_correct = do_ulira_on_one_model(forget_indices,
                                                     1,
                                                     target_margins,
                                                     shadow_unlearning_margins,
                                                     shadow_oracle_margins,
                                                     shadow_forget_masks,
                                                     shadow_training_masks,
                                                     plot_index=plot_index)

        model_score = (test_ulira_correct + forget_ulira_correct) / (
            len(test_indices) + len(forget_indices))
        model_scores.append(model_score)
        if verbose:
            print(f"test_ulira_correct- {test_ulira_correct} ||")
        print(
            f"model - score: {model_score} - mean : {np.mean(model_scores)} - std : {np.std(model_scores)}"
        )
    #raise
    return (model_scores)


###
##  CALL ULIRA
###

# pprint config
print(f"config - {config}")
for k, v in config.items():
    print(f"{k} - {v}")

print(f"results_config- {results_config}")
for k, v in results_config.items():
    print(f"{k} - {v}")

noop = False  # False# False
if noop:
    #forget_masks = ulira.get_ulira_forget_mask(ds_name, class_5_range=1000, overwrite=False )
    #forget_masks_ = generate_ulira_forget_mask(dataset_name, class_5_range=5000, training_mask=training_masks)
    print(f"do nothing")

    m, n = non_repeated_training_masks.shape
    num_models = non_repeated_all_oracle_margins_ulira.shape[0]
    forgettable_masks = np.zeros((n))
    forgettable_masks.shape
    forgettable_masks[forgettable_indices] = 1

    fake_forget_masks = []
    print(f"creating a fake mask ")

    for training_mask in non_repeated_training_masks[:num_models]:
        # pick 200 forgettable indices

        forgettable_indices_200 = np.random.choice(forgettable_indices,
                                                   unlearning_forget_set_size,
                                                   replace=False)
        forgettable_mask = forgettable_masks * training_mask

        # if we want to only take 200 of these points
        if True:
            forgettable_mask_indices = forgettable_mask.nonzero()[0]
            forgettable_mask_indices = np.random.choice(
                forgettable_mask_indices, 200, replace=False)
            forgettable_mask = np.zeros_like(training_mask)
            forgettable_mask[forgettable_mask_indices] = 1

        fake_forget_masks.append(forgettable_mask)
    fake_forget_masks = np.array(fake_forget_masks)
    print(f"fake_forget_masks - {fake_forget_masks.shape}")

    model_scores = ulira_paper(non_repeated_all_oracle_margins_ulira,
                               non_repeated_all_oracle_margins_ulira,
                               fake_forget_masks,
                               non_repeated_training_masks,
                               max_models=50,
                               plot_index=1,
                               verbose=False)

else:
    print(
        f"all_unlearned_margins_ulira,shape - {all_unlearned_margins_ulira.shape}"
    )
    print(f"all_oracle_margins_ulira,shape - {all_oracle_margins_ulira.shape}")
    print(f"forget_masks.shape - {forget_masks.shape}")
    print(f"training_masks.shape - {training_masks.shape}")
    if False:
        model_scores = ulira_paper(all_unlearned_margins_ulira,
                                   all_oracle_margins_ulira,
                                   forget_masks,
                                   training_masks,
                                   plot_index=1,
                                   max_models=None,
                                   verbose=False)
    else:
        model_scores = ulira_paper(all_unlearned_margins_ulira,
                                   non_repeated_all_oracle_margins_ulira,
                                   forget_masks,
                                   non_repeated_training_masks,
                                   max_models=None,
                                   plot_index=1,
                                   verbose=False)
# 48883

results_config["model_scores"] = model_scores

results_config["accuracies"] = {
    "forget": float(forget_acc),
    "train": float(train_acc),
    "val": float(val_acc)
}

# save to yaml
with open(save_path, 'w') as file:
    yaml.dump(results_config, file)
print(f"saved to {save_path}")
