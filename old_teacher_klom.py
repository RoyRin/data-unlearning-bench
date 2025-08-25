import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
import argparse
import sys
import json
import os
import glob
from pathlib import Path


def to_np_cpu(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Type for {x} should be torch or numpy ndarray")

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
    bins = np.linspace(bins_start, bins_end,
                       bin_count + 1)  # bin_count intervals

    # Digitize arrays: find which bin each sample falls into
    # np.digitize returns indices starting from 1
    p_binned_indices = np.digitize(p_arr, bins)
    q_binned_indices = np.digitize(q_arr, bins)

    # Count samples per bin (adjusting for 1-based indexing of digitize)
    p_bin_counts = np.array(
        [np.sum(p_binned_indices == i) for i in range(1, bin_count + 1)])
    q_bin_counts = np.array(
        [np.sum(q_binned_indices == i) for i in range(1, bin_count + 1)])

    # Convert counts to probabilities
    p_total = p_bin_counts.sum()
    q_total = q_bin_counts.sum()

    # Avoid division by zero if an array is empty
    p_bin_probs = (p_bin_counts / p_total if p_total > 0 else np.zeros_like(
        p_bin_counts, dtype=float))
    q_bin_probs = (q_bin_counts / q_total if q_total > 0 else np.zeros_like(
        q_bin_counts, dtype=float))

    # Avoid log(0) issues in KL divergence calculation. Add eps where p > 0.
    q_bin_probs_safe = np.where(p_bin_probs > 0, np.maximum(q_bin_probs, eps),
                                q_bin_probs)
    # Renormalize q_safe slightly if needed? Scipy handles non-normalized qk ok.

    return stats.entropy(pk=p_bin_probs, qk=q_bin_probs_safe)


def kl_from_margins(
    all_unlearned_margins: torch.Tensor,
    all_oracle_margins: torch.Tensor,
    clip_min: float = -100,
    clip_max: float = 100,
):
    assert (all_oracle_margins.shape == all_unlearned_margins.shape
            ), "Margin tensors must have the same shape"
    print("Computing KL divergence scores...")
    results_list = []
    N = all_oracle_margins.shape[1]
    for sample in tqdm(range(N), desc="KL div"):
        oracle_arr = to_np_cpu(all_oracle_margins[:, sample])
        unlearned_arr = to_np_cpu(all_unlearned_margins[:, sample])
        KL_div = compute_binned_KL_div(unlearned_arr,
                                       oracle_arr,
                                       min_val=clip_min,
                                       max_val=clip_max)
        results_list.append(KL_div)
    results = np.stack(results_list)
    return results


def discover_margin_files(directory_path, data_split=None):
    """Discover margin files in a directory, optionally filtered by data split"""
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Find all files in the directory (assuming all files are margin files)
    all_files = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            all_files.append(file_path)
    
    # Filter files based on data split if specified
    if data_split is not None:
        if data_split == "train":
            margin_files = [f for f in all_files if "_train_" in f.name]
            print(f"Filtering for training data files (containing '_train_')")
        elif data_split == "val":
            margin_files = [f for f in all_files if "_val_" in f.name]
            print(f"Filtering for validation data files (containing '_val_')")
        else:
            raise ValueError(f"Invalid data_split: {data_split}. Must be 'train' or 'val'")
    else:
        margin_files = all_files
        print(f"Using all margin files (no data split filter)")
    
    # Sort files for consistent ordering
    margin_files.sort()
    
    print(f"Discovered {len(margin_files)} margin files in {directory}")
    if len(all_files) != len(margin_files):
        print(f"  (filtered from {len(all_files)} total files)")
    for i, file_path in enumerate(margin_files):
        print(f"  {i+1:2d}. {file_path.name}")
    
    return margin_files


def validate_margin_dimensions(unlearned_files, oracle_files, subset_indices=None):
    """Validate that all margin files have compatible dimensions"""
    print("=" * 80)
    print("VALIDATING MARGIN DIMENSIONS")
    print("=" * 80)
    
    if len(unlearned_files) != len(oracle_files):
        raise ValueError(f"Number of unlearned files ({len(unlearned_files)}) != number of oracle files ({len(oracle_files)})")
    
    print(f"Checking dimensions for {len(unlearned_files)} margin file pairs...")
    
    # Check first file to get expected dimensions
    print(f"\nChecking first file pair for reference dimensions...")
    unlearned_data = torch.load(unlearned_files[0], map_location='cpu')
    oracle_data = torch.load(oracle_files[0], map_location='cpu')
    
    unlearned_margins = unlearned_data['margins']
    oracle_margins = oracle_data['margins']
    
    # Apply subset extraction if needed for dimension checking
    if subset_indices is not None:
        print(f"Applying subset extraction for dimension validation...")
        unlearned_margins = extract_margin_subset(unlearned_margins, subset_indices)
        oracle_margins = extract_margin_subset(oracle_margins, subset_indices)
    
    expected_shape = unlearned_margins.shape
    print(f"  Unlearned file: {unlearned_files[0].name} -> {unlearned_margins.shape}")
    print(f"  Oracle file: {oracle_files[0].name} -> {oracle_margins.shape}")
    
    if unlearned_margins.shape != oracle_margins.shape:
        raise ValueError(f"Shape mismatch in first file pair: unlearned {unlearned_margins.shape} vs oracle {oracle_margins.shape}")
    
    print(f"✓ Reference dimensions: {expected_shape}")
    print(f"  Total margin values per file: {expected_shape[0]:,}")
    
    # Check all remaining files
    print(f"\nValidating dimensions for remaining {len(unlearned_files)-1} file pairs...")
    
    for i, (unlearned_file, oracle_file) in enumerate(zip(unlearned_files[1:], oracle_files[1:]), 1):
        try:
            unlearned_data = torch.load(unlearned_file, map_location='cpu')
            oracle_data = torch.load(oracle_file, map_location='cpu')
            
            unlearned_margins = unlearned_data['margins']
            oracle_margins = oracle_data['margins']
            
            # Apply subset extraction if needed
            if subset_indices is not None:
                unlearned_margins = extract_margin_subset(unlearned_margins, subset_indices)
                oracle_margins = extract_margin_subset(oracle_margins, subset_indices)
            
            if unlearned_margins.shape != expected_shape:
                raise ValueError(f"Unlearned file {unlearned_file.name} has shape {unlearned_margins.shape}, expected {expected_shape}")
            
            if oracle_margins.shape != expected_shape:
                raise ValueError(f"Oracle file {oracle_file.name} has shape {oracle_margins.shape}, expected {expected_shape}")
            
            if unlearned_margins.shape != oracle_margins.shape:
                raise ValueError(f"Shape mismatch in file pair {i+1}: unlearned {unlearned_margins.shape} vs oracle {oracle_margins.shape}")
            
            # Print progress every 10 files or for small numbers of files
            if i % 10 == 0 or len(unlearned_files) <= 10:
                print(f"  ✓ File pair {i+1:2d}/{len(unlearned_files)}: {unlearned_file.name} & {oracle_file.name}")
                
        except Exception as e:
            print(f"  ✗ Error validating file pair {i+1}: {unlearned_file.name} & {oracle_file.name}")
            raise e
    
    print(f"\n✓ All {len(unlearned_files)} file pairs have compatible dimensions: {expected_shape}")
    print("=" * 80)
    
    return expected_shape


def load_margins_from_paths(margin_paths, subset_indices=None):
    """Load margins from multiple paths and combine into a single tensor, optionally extracting a subset"""
    all_margins = []
    
    print(f"Loading margins from {len(margin_paths)} files...")
    for i, path in enumerate(tqdm(margin_paths, desc="Loading margin files")):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Margin file not found: {path}")
        
        margin_data = torch.load(path, map_location='cpu')
        margins = margin_data['margins']  # Shape: (num_tokens,)
        
        # Extract subset if indices are provided
        if subset_indices is not None:
            margins = extract_margin_subset(margins, subset_indices)
        
        all_margins.append(margins)
        
        if i == 0:
            print(f"First file contains {len(margins):,} margin values")
    
    # Verify all files have the same number of margin values
    margin_counts = [len(m) for m in all_margins]
    if not all(count == margin_counts[0] for count in margin_counts):
        raise ValueError(f"Margin files have different margin counts: {margin_counts}")
    
    print(f"All {len(margin_paths)} files contain {margin_counts[0]:,} margin values each")
    
    # Stack along a new dimension (first dimension will be ensemble members)
    # Shape: (ensemble_size, num_margins)
    combined_margins = torch.stack(all_margins, dim=0)
    return combined_margins


def load_batch_indices(indices_file):
    """Load batch indices from JSON file"""
    with open(indices_file, 'r') as f:
        data = json.load(f)
    return data['batch_indices'], data['count']


def get_margin_indices_from_batch_indices(batch_indices, batch_size=8*64*1024, micro_batch_size=64*1024):
    """Convert batch indices to margin indices (token positions)"""
    micro_batches_per_step = batch_size // micro_batch_size
    margin_indices = []
    
    print(f"Converting {len(batch_indices)} batch indices to margin indices...")
    print(f"Batch size: {batch_size:,}, Micro batch size: {micro_batch_size:,}")
    print(f"Micro batches per step: {micro_batches_per_step}")
    
    for step in batch_indices:
        for micro_idx in range(micro_batches_per_step):
            micro_batch_idx = step * micro_batches_per_step + micro_idx
            start_pos = micro_batch_idx * micro_batch_size
            end_pos = start_pos + micro_batch_size  # Note: no +1 here since margins are computed on targets
            
            # Add all token positions in this micro batch
            margin_indices.extend(range(start_pos, end_pos))
    
    print(f"Generated {len(margin_indices):,} margin indices")
    return margin_indices


def extract_margin_subset(all_margins, batch_indices, batch_size=8*64*1024, micro_batch_size=64*1024):
    """Extract margin subset based on batch indices"""
    margin_indices = get_margin_indices_from_batch_indices(batch_indices, batch_size, micro_batch_size)
    
    # Convert to tensor for efficient indexing
    margin_indices_tensor = torch.tensor(margin_indices, dtype=torch.long)
    
    # Filter out indices that exceed the margin tensor length
    valid_mask = margin_indices_tensor < len(all_margins)
    if not valid_mask.all():
        num_invalid = (~valid_mask).sum().item()
        print(f"WARNING: {num_invalid} margin indices exceed margin tensor length ({len(all_margins)})")
        margin_indices_tensor = margin_indices_tensor[valid_mask]
    
    # Extract subset
    subset_margins = all_margins[margin_indices_tensor]
    print(f"Extracted {len(subset_margins):,} margins from subset")
    
    return subset_margins


def save_kl_results(results, output_path, unlearned_paths, oracle_paths, stats, subset_info=None):
    """Save KL divergence results with metadata"""
    output_data = {
        'kl_scores': results,
        'unlearned_margin_paths': [str(p) for p in unlearned_paths],
        'oracle_margin_paths': [str(p) for p in oracle_paths],
        'num_samples': len(results),
        'stats': stats
    }
    
    if subset_info:
        output_data['subset_info'] = subset_info
    
    torch.save(output_data, output_path)
    print(f"Saved KL divergence results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute KL divergence scores from margin files")
    parser.add_argument("--unlearned-dir", required=True,
                       help="Path to directory containing unlearned margin files")
    parser.add_argument("--oracle-dir", required=True,
                       help="Path to directory containing oracle margin files")
    parser.add_argument("--output", required=True,
                       help="Path to save KL divergence results")
    parser.add_argument("--clip-min", type=float, default=-100,
                       help="Minimum value for clipping margins")
    parser.add_argument("--clip-max", type=float, default=100,
                       help="Maximum value for clipping margins")
    
    # Data split filtering
    parser.add_argument("--data-split", type=str, choices=["train", "val"],
                       help="Filter margin files by data split: 'train' for training data, 'val' for validation data")
    
    # Margin subset extraction arguments  
    parser.add_argument("--subset-indices", type=str,
                       help="Path to JSON file containing batch indices for margin subset extraction")
    parser.add_argument("--use-subset", action='store_true',
                       help="Extract margin subset based on batch indices for evaluation")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KL DIVERGENCE COMPUTATION FROM MARGIN DIRECTORIES")
    print("=" * 80)
    
    # Validate directory paths
    unlearned_dir = Path(args.unlearned_dir)
    oracle_dir = Path(args.oracle_dir)
    output_path = Path(args.output)
    
    if not unlearned_dir.exists():
        print(f"Error: Unlearned margins directory does not exist: {unlearned_dir}")
        sys.exit(1)
    
    if not oracle_dir.exists():
        print(f"Error: Oracle margins directory does not exist: {oracle_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Discover margin files in both directories
    print(f"\nDiscovering margin files...")
    if args.data_split:
        print(f"Data split filter: {args.data_split}")
    print(f"Unlearned margins directory: {unlearned_dir}")
    unlearned_files = discover_margin_files(unlearned_dir, args.data_split)
    
    print(f"\nOracle margins directory: {oracle_dir}")
    oracle_files = discover_margin_files(oracle_dir, args.data_split)
    
    # Limit the number of files to the minimum of the two directories - ROY- HACK
    
    max_num_files = min(len(unlearned_files), len(oracle_files))
    unlearned_files = unlearned_files[:max_num_files]
    oracle_files = oracle_files[:max_num_files]
    
    
    if len(unlearned_files) == 0:
        print(f"Error: No margin files found in unlearned directory: {unlearned_dir}")
        sys.exit(1)
    
    if len(oracle_files) == 0:
        print(f"Error: No margin files found in oracle directory: {oracle_dir}")
        sys.exit(1)
    
    if len(unlearned_files) != len(oracle_files):
        print(f"Error: Number of files mismatch - unlearned: {len(unlearned_files)}, oracle: {len(oracle_files)}")
        sys.exit(1)
    
    print(f"\n✓ File discovery completed: {len(unlearned_files)} margin files in each directory")
    
    # Handle subset extraction if requested
    subset_info = None
    subset_indices = None
    if args.use_subset:
        if not args.subset_indices:
            print("Error: --use-subset requires --subset-indices")
            sys.exit(1)
        
        indices_file_path = Path(args.subset_indices)
        
        if not indices_file_path.exists():
            print(f"Error: Subset indices file does not exist: {indices_file_path}")
            sys.exit(1)
        
        print("=" * 80)
        print("LOADING SUBSET INDICES FOR MARGIN EXTRACTION")
        print("=" * 80)
        
        # Load batch indices
        batch_indices, count = load_batch_indices(indices_file_path)
        subset_indices = batch_indices
        
        # Store subset information for metadata
        subset_info = {
            'indices_file_path': str(indices_file_path),
            'batch_indices_count': count,
            'batch_indices': batch_indices,
            'subset_extraction_enabled': True
        }
        
        print(f"✓ Loaded {len(batch_indices)} batch indices (count: {count})")
        print("These indices will be used to extract margin subsets for KL divergence computation")
        print("=" * 80)
    
    # Validate margin dimensions before proceeding
    # HACK
    
    if True:
        expected_shape = validate_margin_dimensions(unlearned_files, oracle_files, subset_indices)
    
    # Load margins
    print("Loading unlearned margins...")
    all_unlearned_margins = load_margins_from_paths(unlearned_files, subset_indices)
    
    print("Loading oracle margins...")
    all_oracle_margins = load_margins_from_paths(oracle_files, subset_indices)
    
    # Verify shapes match (should be guaranteed by validation, but double-check)
    if all_unlearned_margins.shape != all_oracle_margins.shape:
        print(f"Error: Shape mismatch between unlearned {all_unlearned_margins.shape} and oracle {all_oracle_margins.shape} margins")
        sys.exit(1)
    
    print(f"\n✓ Margin loading completed!")
    print(f"Final margin tensor shape: {all_unlearned_margins.shape}")
    print(f"Ensemble size: {all_unlearned_margins.shape[0]}")
    print(f"Number of samples: {all_unlearned_margins.shape[1]}")
    
    # Compute KL scores
    print("=" * 80)
    results = kl_from_margins(
        all_unlearned_margins,
        all_oracle_margins,
        clip_min=args.clip_min,
        clip_max=args.clip_max
    )
    
    # Compute statistics
    stats = {
        'mean': float(np.mean(results)),
        'std': float(np.std(results)),
        'min': float(np.min(results)),
        'max': float(np.max(results)),
        'median': float(np.median(results))
    }
    
    print(f"\n" + "=" * 80)
    print(f"KL DIVERGENCE COMPUTATION RESULTS")
    print(f"=" * 80)
    print(f"KL divergence statistics:")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std Dev: {stats['std']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  Min: {stats['min']:.6f}")
    print(f"  Max: {stats['max']:.6f}")
    print(f"  Total samples: {len(results):,}")
    
    # Save results
    save_kl_results(results, output_path, unlearned_files, oracle_files, stats, subset_info)
    
    print(f"\n" + "=" * 80)
    print(f"KL DIVERGENCE COMPUTATION COMPLETED SUCCESSFULLY!")
    print(f"=" * 80)
    print(f"Processed {len(unlearned_files)} unlearned and {len(oracle_files)} oracle margin files")
    if subset_info:
        print(f"Used margin subset from {subset_info['batch_indices_count']} batches ({subset_info['indices_file_path']})")
    print(f"Final margin tensor shape: {all_unlearned_margins.shape}")
    print(f"Results saved to: {output_path}")
    print(f"=" * 80)


if __name__ == "__main__":
    main()