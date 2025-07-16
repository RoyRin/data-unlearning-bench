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


def calculate_total_batches(dataset_file, batch_size=8*64*1024):
    """Calculate total number of batches in the dataset"""
    all_tokens = _load_data_shard(str(dataset_file))
    total_tokens = len(all_tokens)
    total_batches = total_tokens // batch_size
    return total_batches


def verify_complementary_indices(forget_indices_file, retain_indices_file, dataset_file):
    """Verify that forget and retain indices are proper complements"""
    print(f"\n{'='*80}")
    print(f"VERIFYING COMPLEMENTARY INDICES")
    print(f"{'='*80}")
    print(f"Forget file: {Path(forget_indices_file).name}")
    print(f"Retain file: {Path(retain_indices_file).name}")
    
    # Load forget indices
    print(f"\nLoading forget indices...")
    forget_indices, forget_count = load_batch_indices(forget_indices_file)
    forget_set = set(forget_indices)
    print(f"Forget set: {len(forget_indices)} indices (count: {forget_count})")
    
    # Load retain indices
    print(f"Loading retain indices...")
    retain_indices, retain_count = load_batch_indices(retain_indices_file)
    retain_set = set(retain_indices)
    print(f"Retain set: {len(retain_indices)} indices (count: {retain_count})")
    
    # Calculate total batches in dataset
    print(f"Calculating total batches in dataset...")
    total_batches = calculate_total_batches(dataset_file)
    print(f"Total batches in dataset: {total_batches}")
    
    # Test 1: Check for overlap
    print(f"\n{'='*60}")
    print("TEST 1: CHECKING FOR OVERLAP")
    print(f"{'='*60}")
    overlap = forget_set & retain_set
    if overlap:
        print(f"❌ OVERLAP DETECTED: {len(overlap)} indices appear in both sets!")
        print(f"Overlapping indices: {sorted(list(overlap))[:10]}{'...' if len(overlap) > 10 else ''}")
        return False
    else:
        print(f"✓ NO OVERLAP: Forget and retain sets are disjoint")
    
    # Test 2: Check union equals total set
    print(f"\n{'='*60}")
    print("TEST 2: CHECKING UNION COMPLETENESS")
    print(f"{'='*60}")
    union_set = forget_set | retain_set
    expected_set = set(range(total_batches))
    
    print(f"Forget set size: {len(forget_set)}")
    print(f"Retain set size: {len(retain_set)}")
    print(f"Union set size: {len(union_set)}")
    print(f"Expected total: {total_batches}")
    
    if union_set == expected_set:
        print(f"✓ UNION COMPLETE: Forget ∪ Retain = Total dataset")
    else:
        print(f"❌ UNION INCOMPLETE: Missing or extra indices detected")
        
        missing = expected_set - union_set
        extra = union_set - expected_set
        
        if missing:
            print(f"Missing indices: {len(missing)} (e.g., {sorted(list(missing))[:10]})")
        if extra:
            print(f"Extra indices: {len(extra)} (e.g., {sorted(list(extra))[:10]})")
        
        return False
    
    # Test 3: Verify sizes add up correctly
    print(f"\n{'='*60}")
    print("TEST 3: CHECKING SIZE ARITHMETIC")
    print(f"{'='*60}")
    size_sum = len(forget_set) + len(retain_set)
    if size_sum == total_batches:
        print(f"✓ SIZE ARITHMETIC: {len(forget_set)} + {len(retain_set)} = {total_batches}")
    else:
        print(f"❌ SIZE MISMATCH: {len(forget_set)} + {len(retain_set)} = {size_sum} ≠ {total_batches}")
        return False
    
    print(f"\n🎉 ALL TESTS PASSED: Forget and retain indices are proper complements!")
    return True


def test_all_complementary_pairs():
    """Test all forget/retain index pairs"""
    dataset_file = 'data/fineweb10B/fineweb_train_subset.bin'
    
    test_pairs = [
        ('1pct-forget-random-indices.json', '1pct-retain-random-indices.json'),
        ('1pct-forget-loss-indices.json', '1pct-retain-loss-indices.json'),
        ('5pct-forget-random-indices.json', '5pct-retain-random-indices.json'),
        ('5pct-forget-loss-indices.json', '5pct-retain-loss-indices.json'),
    ]
    
    results = {}
    for forget_file, retain_file in test_pairs:
        print(f"\n{'='*100}")
        print(f"TESTING COMPLEMENTARY PAIR: {forget_file} & {retain_file}")
        print(f"{'='*100}")
        
        forget_path = f'data/ngpt-set-indices/{forget_file}'
        retain_path = f'data/ngpt-set-indices/{retain_file}'
        
        if not Path(forget_path).exists():
            print(f"❌ Forget file not found: {forget_path}")
            results[forget_file] = False
            continue
            
        if not Path(retain_path).exists():
            print(f"❌ Retain file not found: {retain_path}")
            results[forget_file] = False
            continue
        
        success = verify_complementary_indices(forget_path, retain_path, dataset_file)
        results[forget_file] = success
    
    # Summary
    print(f"\n{'='*100}")
    print("COMPLEMENTARY INDICES VERIFICATION SUMMARY")
    print(f"{'='*100}")
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:40}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall result: {'🎉 ALL COMPLEMENTARY TESTS PASSED' if all_passed else '❌ SOME COMPLEMENTARY TESTS FAILED'}")
    return results


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
    import argparse
    
    parser = argparse.ArgumentParser(description="Test batch indices verification")
    parser.add_argument("--test-complements", action='store_true',
                       help="Test that forget/retain indices are proper complements")
    parser.add_argument("--test-subsets", action='store_true',
                       help="Test that indices extract correct subsets")
    parser.add_argument("--test-all", action='store_true',
                       help="Run all tests")
    
    args = parser.parse_args()
    
    if args.test_complements or args.test_all:
        print("TESTING COMPLEMENTARY INDICES...")
        test_all_complementary_pairs()
    
    if args.test_subsets or args.test_all:
        print("\n" + "="*100)
        print("TESTING SUBSET EXTRACTION...")
        print("="*100)
        print("Testing 1% forget random indices first...")
        
        success = test_1pct_forget_random()
        
        if success:
            print(f"\n✓ 1% test passed! Running all subset tests...")
            test_all_subsets()
        else:
            print(f"\n❌ 1% test failed. Check the implementation.")
    
    if not any([args.test_complements, args.test_subsets, args.test_all]):
        # Default behavior - run both tests
        print("BATCH INDICES VERIFICATION TEST")
        print("Running all tests by default...")
        
        print("\n" + "="*100)
        print("TESTING COMPLEMENTARY INDICES...")
        print("="*100)
        test_all_complementary_pairs()
        
        print("\n" + "="*100)
        print("TESTING SUBSET EXTRACTION...")
        print("="*100)
        print("Testing 1% forget random indices first...")
        
        success = test_1pct_forget_random()
        
        if success:
            print(f"\n✓ 1% test passed! Running all subset tests...")
            test_all_subsets()
        else:
            print(f"\n❌ 1% test failed. Check the implementation.")
