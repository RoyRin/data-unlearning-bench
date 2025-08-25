import numpy as np
from scipy import stats

# --- Version 1: Original, non-vectorized function (Reference Implementation) ---
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


# --- Version 2: Corrected, fully vectorized function (Optimized Implementation) ---
def compute_binned_KL_div_vectorized(
    p_chunk: np.ndarray,
    q_chunk: np.ndarray,
    bin_count=20,
    eps=1e-5,
    min_val=-100,
    max_val=100,
):
    """
    Computes KL divergence for a chunk of samples in a fully vectorized manner.
    This version has been validated to be numerically equivalent to the original.
    """
    p_chunk = np.clip(p_chunk, min_val, max_val)
    q_chunk = np.clip(q_chunk, min_val, max_val)
    bins_starts = np.minimum(p_chunk.min(axis=0), q_chunk.min(axis=0))
    bins_ends = np.maximum(p_chunk.max(axis=0), q_chunk.max(axis=0))
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
    p_bin_probs = np.divide(p_bin_counts, p_totals, where=p_totals > 0)
    q_bin_probs = np.divide(q_bin_counts, q_totals, where=q_totals > 0)
    q_bin_probs_safe = np.where(p_bin_probs > 0, np.maximum(q_bin_probs, eps), q_bin_probs)
    return stats.entropy(pk=p_bin_probs, qk=q_bin_probs_safe, axis=0)


if __name__ == "__main__":
    N = 100
    D_CHUNK = 2048
    
    print("=" * 80)
    print("Running Final End-to-End Validation for KL Divergence Implementations")
    print(f"Generating synthetic data with shape: ({N}, {D_CHUNK})")
    print("=" * 80)
    
    # Use a fixed seed for reproducible test data
    np.random.seed(42)
    p_chunk_data = np.random.randn(N, D_CHUNK) * 10
    q_chunk_data = np.random.randn(N, D_CHUNK) * 10 + np.random.randn(D_CHUNK) * 2
    
    # Test a boundary condition
    max_vals = q_chunk_data.max(axis=0)
    q_chunk_data[0, :] = max_vals
    
    # --- Run both implementations ---
    print("Computing KL divergence using the vectorized (optimized) function...")
    kl_vectorized = compute_binned_KL_div_vectorized(p_chunk_data, q_chunk_data)
    
    print("Computing KL divergence using the original (looped) function...")
    kl_original = np.array([
        compute_binned_KL_div_original(p_chunk_data[:, i], q_chunk_data[:, i]) 
        for i in range(D_CHUNK)
    ])
    
    # --- Final comparison ---
    print("\n--- Comparing Final Results ---")
    
    if np.allclose(kl_vectorized, kl_original):
        print("\n✅ SUCCESS: The vectorized implementation produces the same results as the original.")
        print("The optimization is correct and can be safely used.")
    else:
        max_abs_diff = np.max(np.abs(kl_vectorized - kl_original))
        print("\n❌ FAILURE: The final outputs have diverged.")
        print(f"   Maximum absolute difference: {max_abs_diff}")
        print("   There is a discrepancy between the implementations.")

    print("=" * 80)
