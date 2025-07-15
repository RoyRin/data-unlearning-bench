import torch
import json
import os
import sys
from pathlib import Path

# Add current directory to path to import lite_gpt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lite_gpt import _load_data_shard


def load_batch_indices(indices_file):
    """Load batch indices from JSON file"""
    with open(indices_file, 'r') as f:
        data = json.load(f)
    return data['batch_indices'], data['count']


def extract_tokens_from_indices(all_tokens, batch_indices, batch_size, micro_batch_size):
    """Extract tokens for specific batch indices using the same logic as exploring.py"""
    micro_batches_per_step = batch_size // micro_batch_size
    selected_tokens_list = []
    
    print(f"Extracting tokens for {len(batch_indices)} batches...")
    print(f"Batch size: {batch_size:,}, Micro batch size: {micro_batch_size:,}")
    print(f"Micro batches per step: {micro_batches_per_step}")
    
    for step in batch_indices:
        for micro_idx in range(micro_batches_per_step):
            micro_batch_idx = step * micro_batches_per_step + micro_idx
            start_pos = micro_batch_idx * micro_batch_size
            end_pos = start_pos + micro_batch_size + 1  # +1 for target tokens
            
            if end_pos > len(all_tokens):
                print(f"WARNING: Batch {step}, micro batch {micro_idx} exceeds token limit")
                print(f"  Required end_pos: {end_pos:,}, Available tokens: {len(all_tokens):,}")
                break
                
            micro_batch_tokens = all_tokens[start_pos:end_pos]
            selected_tokens_list.append(micro_batch_tokens)
    
    # Concatenate all selected tokens
    if selected_tokens_list:
        selected_tokens = torch.cat(selected_tokens_list, dim=0)
    else:
        selected_tokens = torch.empty(0, dtype=all_tokens.dtype)
    
    return selected_tokens


def verify_indices_match_dataset(indices_file, full_dataset_file, expected_subset_file):
    """Verify that extracting with indices produces the expected subset"""
    print(f"\n{'='*80}")
    print(f"VERIFYING: {Path(indices_file).name}")
    print(f"{'='*80}")
    
    # Load batch indices
    print(f"Loading indices from: {indices_file}")
    batch_indices, count = load_batch_indices(indices_file)
    print(f"Loaded {len(batch_indices)} batch indices: {batch_indices[:10]}{'...' if len(batch_indices) > 10 else ''}")
    print(f"Count from JSON: {count}")
    
    # Load full dataset
    print(f"\nLoading full dataset from: {full_dataset_file}")
    all_tokens = _load_data_shard(full_dataset_file)
    print(f"Loaded {len(all_tokens):,} tokens from full dataset")
    
    # Extract tokens using indices (same parameters as exploring.py)
    batch_size = 8 * 64 * 1024  # 524,288 tokens per step
    micro_batch_size = 64 * 1024  # 65,536 tokens per micro batch
    
    print(f"\nExtracting tokens using batch indices...")
    extracted_tokens = extract_tokens_from_indices(all_tokens, batch_indices, batch_size, micro_batch_size)
    print(f"Extracted {len(extracted_tokens):,} tokens")
    
    # Load expected subset
    print(f"\nLoading expected subset from: {expected_subset_file}")
    expected_tokens = _load_data_shard(expected_subset_file)
    print(f"Expected subset contains {len(expected_tokens):,} tokens")
    
    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Extracted tokens: {len(extracted_tokens):,}")
    print(f"Expected tokens:  {len(expected_tokens):,}")
    print(f"Length match: {'✓' if len(extracted_tokens) == len(expected_tokens) else '✗'}")
    
    if len(extracted_tokens) == len(expected_tokens):
        # Compare token values
        tokens_match = torch.equal(extracted_tokens, expected_tokens)
        print(f"Token values match: {'✓' if tokens_match else '✗'}")
        
        if tokens_match:
            print(f"🎉 VERIFICATION PASSED: Indices correctly extract the expected subset!")
            return True
        else:
            # Show some differences for debugging
            diff_mask = extracted_tokens != expected_tokens
            num_diffs = diff_mask.sum().item()
            print(f"❌ Token values differ at {num_diffs:,} positions")
            
            if num_diffs > 0:
                first_diff_idx = diff_mask.nonzero()[0].item()
                print(f"First difference at position {first_diff_idx}:")
                print(f"  Extracted: {extracted_tokens[first_diff_idx].item()}")
                print(f"  Expected:  {expected_tokens[first_diff_idx].item()}")
                
                # Show context around first difference
                start_ctx = max(0, first_diff_idx - 5)
                end_ctx = min(len(extracted_tokens), first_diff_idx + 6)
                print(f"\nContext around first difference (positions {start_ctx}-{end_ctx-1}):")
                print(f"Extracted: {extracted_tokens[start_ctx:end_ctx].tolist()}")
                print(f"Expected:  {expected_tokens[start_ctx:end_ctx].tolist()}")
            
            return False
    else:
        print(f"❌ VERIFICATION FAILED: Length mismatch!")
        return False


def test_1pct_forget_random():
    """Test the 1% forget random indices"""
    return verify_indices_match_dataset(
        indices_file='data/ngpt-set-indices/1pct-forget-random-indices.json',
        full_dataset_file='data/fineweb10B/fineweb_train_subset.bin',
        expected_subset_file='data/fineweb10B/1pct-forget-random.bin'
    )


def test_all_subsets():
    """Test all four subset combinations"""
    test_cases = [
        ('1pct-forget-random', '1pct-forget-random'),
        ('1pct-forget-loss', '1pct-forget-loss'),
        ('5pct-forget-random', '5pct-forget-random'),
        ('5pct-forget-loss', '5pct-forget-loss'),
    ]
    
    results = {}
    for indices_name, subset_name in test_cases:
        print(f"\n{'='*100}")
        print(f"TESTING: {indices_name}")
        print(f"{'='*100}")
        
        success = verify_indices_match_dataset(
            indices_file=f'data/ngpt-set-indices/{indices_name}-indices.json',
            full_dataset_file='data/fineweb10B/fineweb_train_subset.bin',
            expected_subset_file=f'data/fineweb10B/{subset_name}.bin'
        )
        results[indices_name] = success
    
    # Summary
    print(f"\n{'='*100}")
    print("FINAL SUMMARY")
    print(f"{'='*100}")
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:25}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall result: {'🎉 ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return results


if __name__ == "__main__":
    # Start with just the 1% random test as requested
    print("BATCH INDICES VERIFICATION TEST")
    print("Testing 1% forget random indices first...")
    
    success = test_1pct_forget_random()
    
    if success:
        print(f"\n✓ 1% test passed! Running all tests...")
        test_all_subsets()
    else:
        print(f"\n❌ 1% test failed. Check the implementation.")
