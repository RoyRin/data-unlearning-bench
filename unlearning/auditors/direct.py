"""
This is online lira
"""

import numpy as np
import torch as ch
from scipy import stats
from scipy import special
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from unlearning.unlearning_algos.utils import get_margins
from unlearning.auditors.utils import load_model_from_dict


def get_u_margins(model, ckpt_path, unlearn_fn, unlearning_kwargs,
                  train_loader, eval_loader, forget_set_indices):

    if len(forget_set_indices) < 5:
        raise Exception(
            f"You probably forgot to set the forget_set_indices. - {forget_set_indices}"
        )

    load_model_from_dict(model, ckpt_path)
    model = model.cuda().eval()
    print(f"unlearning_kwargs - {unlearning_kwargs}")
    unlearned_model = unlearn_fn(
        model=model,
        train_dataloader=train_loader,
        forget_dataloader=None,
        forget_indices=forget_set_indices,
        # val_loader = eval_loader,
        **unlearning_kwargs)
    return get_margins(unlearned_model, eval_loader), unlearned_model


def u_margin_job(
    full_ckpt_path,
    unlearn_fn,
    unlearning_kwargs,
    dataset,
    model_factory,
    loader_factory,
    forget_set_indices,
    eval_set_inds,
):
    model = model_factory(dataset)
    load_model_from_dict(model, full_ckpt_path)
    model.eval()

    train_loader = loader_factory(dataset, indexed=True)
    unlearned_models = unlearn_fn(model=model,
                                  train_dataloader=train_loader,
                                  forget_dataloader=None,
                                  forget_indices=forget_set_indices,
                                  **unlearning_kwargs)

    eval_loader = loader_factory(dataset,
                                 split="train_and_val",
                                 indices=eval_set_inds,
                                 indexed=True)
    unlearned_margins = get_margins(unlearned_models, eval_loader)

    return unlearned_margins


def u_full_eval_job(
    full_ckpt_path,
    unlearn_fn,
    unlearning_kwargs,
    dataset,
    model_factory,
    loader_factory,
    forget_set_indices,
    eval_set_inds,
):
    '''
        Keep track of the entire trajectory (one per epoch) and the resulting outputs.
        WARNING: only works with unlearn_fn=oracle_matching!
    '''
    model = model_factory(dataset)
    model.load_state_dict(ch.load(full_ckpt_path))
    model.eval()
    train_loader = loader_factory(dataset, indexed=True)
    eval_loader = loader_factory(dataset, split="train_and_val", indexed=True)
    unlearned_models, output_dict = unlearn_fn(
        model,
        train_loader,
        forget_set_indices,
        train_and_val_loader=eval_loader,
        **unlearning_kwargs)
    return output_dict


def renyi_divergence(p, q, alpha):
    if alpha == 1:
        # Use KL divergence as a special case when alpha = 1
        return stats.entropy(pk=p, qk=q)
    else:
        sum_p_alpha_q_1_minus_alpha = np.sum(p**alpha / q**(alpha - 1))
        return (1 / (alpha - 1)) * np.log(sum_p_alpha_q_1_minus_alpha)


def compute_binned_KL_div(p_arr,
                          q_arr,
                          bin_count=20,
                          eps=1e-5,
                          min_val=-100,
                          max_val=100):
    # we're measuring -E_{p} [log(q)]
    # i.e., wherever we put mass in p, we better have substanent mass in q
    p_arr = np.clip(p_arr, min_val, max_val)
    bins_start = min(p_arr.min(), q_arr.min())
    bins_end = max(p_arr.max(), q_arr.max())
    bins = np.linspace(bins_start, bins_end, bin_count)

    # compute probabilities for p & q for each bin
    p_binned_dist = np.digitize(p_arr, bins)
    p_bin_counts = np.array(
        [np.sum(p_binned_dist == i) for i in range(bin_count)])
    p_bin_probs = p_bin_counts / p_bin_counts.sum()

    q_binned_dist = np.digitize(q_arr, bins)
    q_bin_counts = np.array(
        [np.sum(q_binned_dist == i) for i in range(bin_count)])
    q_bin_probs = q_bin_counts / q_bin_counts.sum()
    # avoid NaN in the entropy
    q_bin_probs = np.clip(q_bin_probs, a_min=eps, a_max=None)
    # no need to worry about renorm, stats.entropy handles it
    KL_div = stats.entropy(pk=p_bin_probs, qk=q_bin_probs)

    RD_2 = renyi_divergence(p_bin_probs, q_bin_probs, alpha=2)
    RD_5 = renyi_divergence(p_bin_probs, q_bin_probs, alpha=5)
    RD_10 = renyi_divergence(p_bin_probs, q_bin_probs, alpha=10)

    entropy = stats.entropy(pk=p_bin_probs)
    cross_entropy = KL_div + entropy

    return KL_div, cross_entropy, RD_2, RD_5, RD_10


def direct_audit_precomputed(
    all_unlearned_margins: ch.Tensor,
    all_oracle_margins: ch.Tensor,
    clip_min: float = -100,
    clip_max: float = 100,
):
    """
    Returns list of tuples of (ksp_value, tp_value, cross_entropy)
        ksp_val = Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.
        tp_value = Performs a t-test for the null hypothesis that 2 independent samples have identical average (expected) values.
        cross_entropy = -E_{Pr[unlearned]} [log(Pr(oracle))]
            cross entropy between
                unlearned models eval on (x)
                and oracles eval on (x)

    """
    print("Starting T-tests...")
    results = []
    N = all_oracle_margins.shape[1]
    for sample in range(N):
        if sample % (N // 10) == 0:
            print(f"Sample {sample}/{N}")
        S1 = all_oracle_margins[:, sample]
        S2 = all_unlearned_margins[:, sample]
        try:
            oracle_arr, unlearned_arr = S1.cpu().numpy(), S2.cpu().numpy()
        except:
            oracle_arr, unlearned_arr = S1, S2
        t_stat, tp_value = stats.ttest_ind(oracle_arr,
                                           unlearned_arr,
                                           equal_var=False)
        # compare difference in CDFs
        stat, ksp_value = stats.ks_2samp(oracle_arr,
                                         unlearned_arr,
                                         alternative="two-sided")

        # measure -E_{Pr[unlearned]} [log(Pr(oracle))]
        # i.e., wherever we put mass in unlearned, we better have substanent mass in oracle
        KL_div, cross_entropy, RD_2, RD_5, RD_10 = compute_binned_KL_div(
            unlearned_arr, oracle_arr, min_val=clip_min, max_val=clip_max)

        #kl_div = compute_binned_kl_divergence(unlearned_arr, oracle_arr , eps = 1e-3)
        results.append(
            np.array([
                ksp_value, tp_value, KL_div, cross_entropy, RD_2, RD_5, RD_10
            ]))

    return np.stack(results)


def plot_margins_direct(all_unlearned_margins, all_oracle_margins,
                        inds_to_plot):
    """
    note: margins are computed for the eval set; currently the eval set
    is the entire train set; if we change the eval set, we need to change
    how we index into all_unlearned_margins and all_oracle_margins with
    the indices in forget_set_indices
    """

    n_cols = 5
    n_rows = len(inds_to_plot) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))

    plt.legend()
    for indE, E in enumerate(inds_to_plot):
        ax = axes[indE // n_cols, indE % n_cols]
        ax.set_xlabel("Margins")
        sns.histplot(
            all_unlearned_margins[:, E],
            ax=ax,
            color="g",
            alpha=0.5,
            stat="probability",
            bins=20,
        )
        sns.histplot(
            all_oracle_margins[:, E],
            ax=ax,
            color="k",
            alpha=0.5,
            stat="probability",
            bins=20,
        )
    # Manually create handles for the legend
    handle1 = mlines.Line2D([], [], color="k", label="oracle")
    handle2 = mlines.Line2D([], [], color="g", label="unlearned")

    # Creating one legend for the whole figure using the handles
    fig.legend(handles=[handle1, handle2], loc="upper center", ncol=2)
    fig.tight_layout()

    return fig
