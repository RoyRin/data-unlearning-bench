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
from typing import List, Union, Generator
import math

import numpy as np
from scipy import stats
import time # For profiling
import sys  # For exiting after profiling

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

def kl_from_margin_generators_vectorized(
    unlearned_margins_generator: Generator[np.ndarray, None, None],
    oracle_margins_generator: Generator[np.ndarray, None, None],
    total_chunks: int,
    clip_min: float = -100,
    clip_max: float = 100,
) -> np.ndarray:
    """
    Computes KL divergence scores by processing whole chunks from margin generators
    in a vectorized fashion.
    """
    print("Computing KL divergence scores from streamed margin data (vectorized)...")
    all_results_chunks = []

    # Use the `total` argument in tqdm for a proper progress bar
    progress_bar = tqdm(
        zip(unlearned_margins_generator, oracle_margins_generator),
        total=total_chunks,
        desc="Processing chunks"
    )

    for unlearned_chunk, oracle_chunk in progress_bar:
        assert (oracle_chunk.shape == unlearned_chunk.shape), \
            f"Margin chunk shapes must match. Got {oracle_chunk.shape} and {unlearned_chunk.shape}"

        kl_divs_chunk = compute_binned_KL_div_vectorized(
            p_chunk=unlearned_chunk,
            q_chunk=oracle_chunk,
            min_val=clip_min,
            max_val=clip_max
        )
        all_results_chunks.append(kl_divs_chunk)
            
    if not all_results_chunks:
        return np.array([])
    return np.concatenate(all_results_chunks)

def discover_margin_files(directory_path, data_split=None):
    """Discover margin files in a directory, optionally filtered by data split"""
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Find all supported files in the directory (np, npy, npz)
    all_files = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix in {".np", ".npy", ".npz"}:
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

def load_batch_indices(indices_file):
    """Load batch indices from JSON file"""
    with open(indices_file, 'r') as f:
        data = json.load(f)
    return data['batch_indices'], data['count']

def get_d_slice_efficient(start_idx: int, end_idx: int, numpy_files: List[str]) -> np.ndarray:
    # Efficiently loads a single contiguous slice from all N memory-mapped files.
    all_slices = [np.load(f, mmap_mode='r')[start_idx:end_idx] for f in numpy_files]
    return np.stack(all_slices)

def get_margins_subset_from_batch_indices(
    batch_indices: Union[List[int], np.ndarray],
    numpy_files: List[str],
    batch_size: int = 8 * 64 * 1024
) -> np.ndarray:
    # Fetches margin data for given batch_indices without creating an intermediate index list.
    all_batch_data = [
        get_d_slice_efficient(step * batch_size, (step + 1) * batch_size, numpy_files)
        for step in batch_indices
    ]
    if not all_batch_data:
        # Return an array with the correct first dimension (N) but empty second dimension.
        return np.empty((len(numpy_files), 0))
    return np.concatenate(all_batch_data, axis=1)

def load_margins_in_batched_ensembles(
    margin_paths: List[str],
    batch_size: int = 8 * 64 * 1024,
    subset_indices: Union[List[int], np.ndarray] = None,
    ensemble_size: int = 64
) -> Generator[np.ndarray, None, None]:
    # Creates a generator to load ensembles of margin batches (N, D_chunk) on the fly.
    
    if subset_indices is not None:
        target_batch_indices = subset_indices
    else:
        # If no subset is specified, determine total batches from the first file's shape.
        d_dim = np.load(margin_paths[0], mmap_mode='r').shape[0]
        total_batches = math.ceil(d_dim / batch_size)
        target_batch_indices = range(total_batches)

    # Iterate through the target indices in chunks of 'ensemble_size'.
    for i in range(0, len(target_batch_indices), ensemble_size):
        chunk_indices = target_batch_indices[i : i + ensemble_size]
        
        if len(chunk_indices) > 0:
            yield get_margins_subset_from_batch_indices(
                batch_indices=chunk_indices,
                numpy_files=margin_paths,
                batch_size=batch_size
            )

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
    parser.add_argument("--ensemble-size", type=float, default=1,
                       help="Number of batches per ensemble")
    
    # Skip validation checks
    parser.add_argument("--skip-checks", action='store_true', default=True,
                       help="Skip dimension validation checks before computing KL scores (default: True)")
    
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
    
    # Validate margin dimensions before proceeding (optional)
    if not args.skip_checks:
        print("=" * 80)
        print("PERFORMING DIMENSION VALIDATION CHECKS")
        print("=" * 80)
        expected_shape = validate_margin_dimensions(unlearned_files, oracle_files, subset_indices)
    else:
        print("=" * 80)
        print("SKIPPING DIMENSION VALIDATION CHECKS")
        print("=" * 80)

    # --- START: Progress Bar Calculation ---
    # Calculate the total number of chunks for the progress bar BEFORE creating the generators.
    if subset_indices is not None:
        num_target_batches = len(subset_indices)
    else:
        # If processing all data, determine total batches from the first file's shape.
        # This requires one small, fast read to get the dimension.
        # NOTE: Assumes the default batch_size=8*64*1024 is used.
        d_dim = np.load(unlearned_files[0], mmap_mode='r').shape[0]
        batch_size = 8 * 64 * 1024 
        num_target_batches = math.ceil(d_dim / batch_size)

    total_chunks = math.ceil(num_target_batches / args.ensemble_size)
    # --- END: Progress Bar Calculation ---
    
    # Load margins
    print("Loading unlearned margins...")
    unlearned_margins_generator = load_margins_in_batched_ensembles(margin_paths=unlearned_files, subset_indices=subset_indices, ensemble_size=args.ensemble_size)
    
    print("Loading oracle margins...")
    oracle_margins_generator = load_margins_in_batched_ensembles(margin_paths=oracle_files, subset_indices=subset_indices, ensemble_size=args.ensemble_size)
    
    print(f"\n✓ Margin loading completed!")
    
    # Compute KL scores
    print("=" * 80)
    results = kl_from_margin_generators_vectorized(
        unlearned_margins_generator,
        oracle_margins_generator,
        total_chunks=total_chunks,
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
    print(f"Results saved to: {output_path}")
    print(f"=" * 80)


if __name__ == "__main__":
    main()
