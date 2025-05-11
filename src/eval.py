# stdlib deps
import os

# project deps
from paths import EVAL_DIR, DATA_DIR

# external deps
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats


def to_np_cpu(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Type for {x} should be torch or numpy ndarray")
        import pdb

        pdb.set_trace()

def get_margins_from_multimodel_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    logits, labels = logits.to(device), labels.to(device)
    assert logits.dim() == 3
    n_models, n_datapoints, n_classes = logits.shape
    one_hot = torch.nn.functional.one_hot(labels, num_classes=n_classes).bool()
    one_hot = one_hot.unsqueeze(0).expand(n_models, -1, -1)
    logits_correct = logits[one_hot].view(n_models, n_datapoints)
    lse_other = logits.masked_fill(one_hot, -torch.inf).logsumexp(dim=-1)
    return logits_correct - lse_other

def get_margin(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    For each image in images, compute the margin of the correct class.
    margin = logit_correct - log_sum_exp(logit_other)
    """
    logits = model(images)
    bindex = torch.arange(logits.shape[0]).to(logits.device, non_blocking=False)
    logits_correct = logits[bindex, labels]
    # Use clone to avoid modifying original logits if model is used elsewhere
    cloned_logits = logits.clone()
    cloned_logits[bindex, labels] = torch.tensor(
        -torch.inf, device=cloned_logits.device, dtype=cloned_logits.dtype
    )
    return logits_correct - cloned_logits.logsumexp(dim=-1)


def get_margins(
    model: torch.nn.Module, loader: DataLoader, device: str = "cuda"
) -> torch.Tensor:
    model = model.to(device).eval()
    all_margins = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda"):
                margins = get_margin(model, x, y)
            all_margins.append(margins.cpu())
    return torch.cat(all_margins)


def compute_binned_KL_div(
    p_arr: np.ndarray,
    q_arr: np.ndarray,
    bin_count=20,
    eps=1e-5,
    min_val=-100,
    max_val=100,
):
    """
    Computes KL divergence between two distributions represented by samples,
    using binning. Calculates D_KL(p || q).
    """
    # Clip arrays to avoid extreme values affecting bin ranges
    p_arr = np.clip(p_arr, min_val, max_val)
    q_arr = np.clip(q_arr, min_val, max_val)

    # Determine bins based on the combined range of both arrays
    bins_start = min(p_arr.min(), q_arr.min())
    bins_end = max(p_arr.max(), q_arr.max())
    if bins_start >= bins_end:  # Handle edge case where all values are the same
        bins_end = bins_start + 1
    bins = np.linspace(bins_start, bins_end, bin_count + 1)  # bin_count intervals

    # Digitize arrays: find which bin each sample falls into
    # np.digitize returns indices starting from 1
    p_binned_indices = np.digitize(p_arr, bins)
    q_binned_indices = np.digitize(q_arr, bins)

    # Count samples per bin (adjusting for 1-based indexing of digitize)
    p_bin_counts = np.array(
        [np.sum(p_binned_indices == i) for i in range(1, bin_count + 1)]
    )
    q_bin_counts = np.array(
        [np.sum(q_binned_indices == i) for i in range(1, bin_count + 1)]
    )

    # Convert counts to probabilities
    p_total = p_bin_counts.sum()
    q_total = q_bin_counts.sum()

    # Avoid division by zero if an array is empty
    p_bin_probs = (
        p_bin_counts / p_total
        if p_total > 0
        else np.zeros_like(p_bin_counts, dtype=float)
    )
    q_bin_probs = (
        q_bin_counts / q_total
        if q_total > 0
        else np.zeros_like(q_bin_counts, dtype=float)
    )

    # Avoid log(0) issues in KL divergence calculation. Add eps where p > 0.
    q_bin_probs_safe = np.where(
        p_bin_probs > 0, np.maximum(q_bin_probs, eps), q_bin_probs
    )
    # Renormalize q_safe slightly if needed? Scipy handles non-normalized qk ok.

    return stats.entropy(pk=p_bin_probs, qk=q_bin_probs_safe)


def kl_from_margins(
    all_unlearned_margins: torch.Tensor,
    all_oracle_margins: torch.Tensor,
    clip_min: float = -100,
    clip_max: float = 100,
):
    assert (
        all_oracle_margins.shape == all_unlearned_margins.shape
    ), "Margin tensors must have the same shape"
    print("Computing results...")
    results_list = []
    N = all_oracle_margins.shape[1]
    for sample in tqdm(range(N), desc="KL div"):
        oracle_arr = to_np_cpu(all_oracle_margins[:, sample])
        unlearned_arr = to_np_cpu(all_unlearned_margins[:, sample])
        KL_div = compute_binned_KL_div(
            unlearned_arr, oracle_arr, min_val=clip_min, max_val=clip_max
        )
        results_list.append(KL_div)
    results = np.stack(results_list)
    return results
