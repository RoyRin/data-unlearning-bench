import torch
import json
import numpy as np
from pathlib import Path
import sys
import os

# Add current directory to path to import from teacher_klom
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from teacher_klom import load_batch_indices, extract_margin_subset, get_margin_indices_from_batch_indices

def create_dummy_margin_file(filepath, num_tokens=1000000):
    """Create a dummy margin file for testing"""
    margins = torch.randn(num_tokens)  # Random margins
    margin_data = {
        'margins': margins,
        'model_path': 'dummy_model.pt',
        'data_path': 'dummy_data.bin',
        'num_tokens': num_tokens,
        'margin_stats': {
            'mean': margins.mean().item(),
            'std': margins.std().item(),
            'min': margins.min().item(),
            'max': margins.max().item()
        }
    }
    torch.save(margin_data, filepath)
    print(f"Created dummy margin file: {filepath} with {num_tokens:,} margins")

def test_margin_subset_extraction():
    """Test margin subset extraction with actual indices"""
    print("=" * 80)
    print("TESTING MARGIN SUBSET EXTRACTION")
    print("=" * 80)
    
    # Create a temporary margin file
    temp_margin_file = "temp_margins.pt"
    create_dummy_margin_file(temp_margin_file, num_tokens=728760320)  # Same size as training data
    
    try:
        # Load actual indices
        indices_file = "data/ngpt-set-indices/1pct-forget-random-indices.json"
        if not Path(indices_file).exists():
            print(f"Error: Indices file not found: {indices_file}")
            return False
        
        print(f"\nLoading batch indices from: {indices_file}")
        batch_indices, count = load_batch_indices(indices_file)
        print(f"Loaded {len(batch_indices)} batch indices (count: {count})")
        
        # Load the dummy margin file
        print(f"\nLoading margins from: {temp_margin_file}")
        margin_data = torch.load(temp_margin_file, map_location='cpu')
        all_margins = margin_data['margins']
        print(f"Loaded {len(all_margins):,} margins")
        
        # Extract subset
        print(f"\nExtracting margin subset...")
        subset_margins = extract_margin_subset(all_margins, batch_indices)
        
        # Calculate expected size
        batch_size = 8 * 64 * 1024  # 524,288 tokens per step
        expected_subset_size = len(batch_indices) * batch_size
        
        print(f"\n" + "=" * 60)
        print("EXTRACTION RESULTS")
        print(f"=" * 60)
        print(f"Original margins: {len(all_margins):,}")
        print(f"Batch indices: {len(batch_indices)}")
        print(f"Expected subset size: {expected_subset_size:,}")
        print(f"Actual subset size: {len(subset_margins):,}")
        print(f"Size match: {'✓' if len(subset_margins) == expected_subset_size else '✗'}")
        
        # Test margin indices conversion
        print(f"\nTesting margin indices conversion...")
        margin_indices = get_margin_indices_from_batch_indices(batch_indices)
        print(f"Generated {len(margin_indices):,} margin indices")
        print(f"Index conversion match: {'✓' if len(margin_indices) == len(subset_margins) else '✗'}")
        
        # Show some sample indices
        print(f"\nFirst 10 batch indices: {batch_indices[:10]}")
        print(f"First 10 margin indices: {margin_indices[:10]}")
        print(f"Last 10 margin indices: {margin_indices[-10:]}")
        
        success = len(subset_margins) == expected_subset_size
        print(f"\n{'🎉 TEST PASSED' if success else '❌ TEST FAILED'}")
        return success
        
    finally:
        # Clean up
        if Path(temp_margin_file).exists():
            Path(temp_margin_file).unlink()
            print(f"\nCleaned up temporary file: {temp_margin_file}")

def test_teacher_klom_help():
    """Test that teacher_klom.py shows the new arguments in help"""
    print("\n" + "=" * 80)
    print("TESTING TEACHER_KLOM.PY ARGUMENT PARSING")
    print("=" * 80)
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "teacher_klom.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        help_text = result.stdout
        print("Checking for new arguments in help text...")
        
        has_subset_indices = "--subset-indices" in help_text
        has_use_subset = "--use-subset" in help_text
        
        print(f"--subset-indices argument: {'✓' if has_subset_indices else '✗'}")
        print(f"--use-subset argument: {'✓' if has_use_subset else '✗'}")
        
        if has_subset_indices and has_use_subset:
            print("✓ All new arguments found in help text")
            return True
        else:
            print("❌ Some arguments missing from help text")
            return False
            
    except Exception as e:
        print(f"Error testing help: {e}")
        return False

if __name__ == "__main__":
    print("TEACHER_KLOM MARGIN SUBSET TESTING")
    print("Testing the new margin subset extraction functionality")
    
    # Test margin subset extraction
    test1_success = test_margin_subset_extraction()
    
    # Test argument parsing
    test2_success = test_teacher_klom_help()
    
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Margin subset extraction: {'✓ PASSED' if test1_success else '✗ FAILED'}")
    print(f"Argument parsing: {'✓ PASSED' if test2_success else '✗ FAILED'}")
    
    overall_success = test1_success and test2_success
    print(f"\nOverall: {'🎉 ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}") 