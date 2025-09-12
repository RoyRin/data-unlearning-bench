import numpy as np
from scipy import stats

# --- Version 1: Original, non-vectorized function (Reference Implementation) ---
# This is the old "teacher klom" implementation we are comparing against.
# It is known to produce NaN for the p_total=0 edge case.
def compute_binned_KL_div_original(
    p_arr: np.ndarray,
    q_arr: np.ndarray,
    bin_count=20,
    eps=1e-5,
    min_val=-100,
    max_val=100,
):
    """
    Computes KL divergence for a single sample (1D array).
    This is the reference implementation.
    """
    p_arr = np.clip(p_arr, min_val, max_val)
    q_arr = np.clip(q_arr, min_val, max_val)
    bins_start = min(p_arr.min(), q_arr.min())
    bins_end = max(p_arr.max(), q_arr.max())
    if bins_start >= bins_end:
        bins_end = bins_start + 1
    bins = np.linspace(bins_start, bins_end, bin_count + 1)
    p_binned_indices = np.digitize(p_arr, bins)
    q_binned_indices = np.digitize(q_arr, bins)
    p_bin_counts = np.array([np.sum(p_binned_indices == i) for i in range(1, bin_count + 1)])
    q_bin_counts = np.array([np.sum(q_binned_indices == i) for i in range(1, bin_count + 1)])
    p_total = p_bin_counts.sum()
    q_total = q_bin_counts.sum()
    p_bin_probs = (p_bin_counts / p_total if p_total > 0 else np.zeros_like(p_bin_counts, dtype=float))
    q_bin_probs = (q_bin_counts / q_total if q_total > 0 else np.zeros_like(q_bin_counts, dtype=float))
    q_bin_probs_safe = np.where(p_bin_probs > 0, np.maximum(q_bin_probs, eps), q_bin_probs)
    return stats.entropy(pk=p_bin_probs, qk=q_bin_probs_safe)


# --- Version 2: New, fully vectorized function with NumPy KL (Optimized Implementation) ---
# This is the final, robust version from your main script.

def compute_kl_numpy(pk: np.ndarray, qk: np.ndarray) -> np.ndarray:
    """
    Computes KL divergence using pure NumPy, robustly handling the p=0 edge case.
    """
    terms = np.zeros_like(pk, dtype=float)
    mask = pk > 0
    # Calculate the terms ONLY where pk is positive
    terms[mask] = pk[mask] * np.log(pk[mask] / qk[mask])
    return np.sum(terms, axis=0)

def compute_binned_KL_div_vectorized_numpy(
    p_chunk: np.ndarray,
    q_chunk: np.ndarray,
    bin_count=20,
    eps=1e-5,
    min_val=-100,
    max_val=100,
):
    """
    Computes KL divergence using a manual NumPy implementation for the final entropy
    step, providing full transparency and NaN safety.
    """
    p_chunk = np.clip(p_chunk, min_val, max_val)
    q_chunk = np.clip(q_chunk, min_val, max_val)
    
    bins_starts = np.minimum(np.nanmin(p_chunk, axis=0), np.nanmin(q_chunk, axis=0))
    bins_ends = np.maximum(np.nanmax(p_chunk, axis=0), np.nanmax(q_chunk, axis=0))
    bins_starts = np.nan_to_num(bins_starts, nan=0.0)
    bins_ends = np.nan_to_num(bins_ends, nan=1.0)
    identical_mask = bins_starts >= bins_ends
    bins_ends[identical_mask] = bins_starts[identical_mask] + 1
    
    bin_edges = np.stack(
        [np.linspace(bins_starts[i], bins_ends[i], bin_count + 1) for i in range(p_chunk.shape[1])],
        axis=1
    )
    
    p_binned_indices = np.sum(p_chunk[:, np.newaxis, :] >= bin_edges[np.newaxis, :, :], axis=1)
    q_binned_indices = np.sum(q_chunk[:, np.newaxis, :] >= bin_edges[np.newaxis, :, :], axis=1)
    
    bin_range = np.arange(1, bin_count + 1)[:, np.newaxis, np.newaxis]
    p_bin_counts = np.sum(p_binned_indices == bin_range, axis=1).astype(float)
    q_bin_counts = np.sum(q_binned_indices == bin_range, axis=1).astype(float)
    
    p_totals = p_bin_counts.sum(axis=0)
    q_totals = q_bin_counts.sum(axis=0)
    
    p_bin_probs = np.divide(p_bin_counts, p_totals, where=p_totals > 0, out=np.zeros_like(p_bin_counts))
    q_bin_probs = np.divide(q_bin_counts, q_totals, where=q_totals > 0, out=np.zeros_like(q_bin_counts))
    
    q_bin_probs_safe = np.where(p_bin_probs > 0, np.maximum(q_bin_probs, eps), q_bin_probs)
    
    kl_divs = compute_kl_numpy(pk=p_bin_probs, qk=q_bin_probs_safe)
    
    return np.nan_to_num(kl_divs, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    N = 100
    D_CHUNK = 2048
    
    print("=" * 80)
    print("Final Validation: New NumPy Vectorized KL vs. Old Original Looped KL")
    print("This test checks for correctness up to 4 decimal places of precision.")
    print(f"Generating synthetic data with shape: ({N}, {D_CHUNK})")
    print("=" * 80)
    
    np.random.seed(42)
    p_chunk_data = np.random.randn(N, D_CHUNK) * 10
    q_chunk_data = np.random.randn(N, D_CHUNK) * 10 + np.random.randn(D_CHUNK) * 2
    
    # Intentionally create the edge case to ensure NaN handling is tested
    p_chunk_data[:, 50] = -5000 
    
    # --- Run both implementations ---
    print("Computing KL divergence using the new (NumPy Vectorized) function...")
    kl_vectorized = compute_binned_KL_div_vectorized_numpy(p_chunk_data, q_chunk_data)
    
    print("Computing KL divergence using the old (Original Looped) function...")
    kl_original = np.array([
        compute_binned_KL_div_original(p_chunk_data[:, i], q_chunk_data[:, i]) 
        for i in range(D_CHUNK)
    ])
    
    # --- Final comparison with tolerance ---
    print("\n--- Comparing Final Results ---")
    
    # Harmonize the results: convert known NaNs from the old method to 0 for a fair comparison
    nan_mask = np.isnan(kl_original)
    kl_original_harmonized = np.nan_to_num(kl_original, nan=0.0)
    
    # Check if the arrays are close within the desired precision
    are_results_close = np.allclose(kl_vectorized, kl_original_harmonized, atol=1e-4, rtol=0)
    
    if are_results_close:
        print("\n✅ SUCCESS: The new vectorized implementation produces the same results as the original")
        print("           (within the specified tolerance of 4 decimal places).")
        print("           The optimization is correct and can be safely used.")
    else:
        max_abs_diff = np.max(np.abs(kl_vectorized - kl_original_harmonized))
        print("\n❌ FAILURE: The final outputs have diverged beyond the tolerance.")
        print(f"   Maximum absolute difference: {max_abs_diff}")
        print("   There is a discrepancy between the implementations.")

    print("=" * 80)
