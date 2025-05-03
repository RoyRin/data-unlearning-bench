import numpy as np
from pathlib import Path
from memorization import utils

SEED = 42

from unlearning import ULIRA_BASE_DIR


def make_indices_plan(model_count, ratio=0.5, data_points=50_000, seed=SEED):
    """
    """
    # seed
    np.random.seed(seed)
    # get number of ones and zeros
    ones = int(ratio * data_points)
    zeros = data_points - ones
    # shuffle model count times
    plans = []
    for i in range(model_count):
        plan = np.concatenate([np.ones(ones), np.zeros(zeros)])
        # plan to int
        plan = plan.astype(int)
        np.random.shuffle(plan)
        # get nonzero indices
        plan_indices = np.nonzero(plan)[0]
        plans.append(plan_indices)
    return np.array(plans)


def make_plan(model_count, ratio=0.5, data_points=50_000, seed=SEED):
    """
    """
    # seed
    np.random.seed(seed)
    # get number of ones and zeros
    ones = int(ratio * data_points)
    zeros = data_points - ones
    # shuffle model count times
    plans = []
    for i in range(model_count):
        plan = np.concatenate([np.ones(ones), np.zeros(zeros)]).astype(int)
        # plan to int
        np.random.shuffle(plan)
        # get nonzero indices
        # plan_indices = np.nonzero(plan)[0]
        plans.append(plan)
    return np.array(plans)


def get_plan():
    """
    Load the plan from the data directory.

    Returns:
    - numpy.ndarray: An array of plans, where each plan is represented as an array of indices.
    """
    plan_save_path = ULIRA_BASE_DIR / "plans.npy"
    if plan_save_path.exists():
        plans = np.load(plan_save_path)
        return plans
    else:
        raise FileNotFoundError(f"Plan not found at {plan_save_path}.")


def make_forget_plan(model_count, ratio=0.5, data_points=50_000, seed=SEED):
    """
    """
    # seed
    np.random.seed(seed)
    # get number of ones and zeros
    ones = int(ratio * data_points)
    zeros = data_points - ones
    # shuffle model count times
    plans = []
    for i in range(model_count):
        plan = np.concatenate([np.ones(ones), np.zeros(zeros)]).astype(int)
        # plan to int
        np.random.shuffle(plan)
        # get nonzero indices
        # plan_indices = np.nonzero(plan)[0]
        plans.append(plan)
    return np.array(plans)


if __name__ == "__main__":
    ULIRA_BASE_DIR.mkdir(exist_ok=True)

    plan_save_path = ULIRA_BASE_DIR / "plans.npy"
    if plan_save_path.exists():
        plans = np.load(plan_save_path)
        print(plans)
        print(f"shape : {plans.shape}")
        exit(0)

    # make plan
    seed = 42
    ratio = 0.5
    data_points = 50_000
    model_count = 2_000
    plans = make_plan(model_count, ratio=ratio, data_points=data_points)
    # save plan
    print(f"shape : {plans.shape}")
    print(plans[:10])
    np.save(plan_save_path, plans)
    print(f"saving to {plan_save_path}")
