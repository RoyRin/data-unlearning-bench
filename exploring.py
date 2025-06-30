import torch
import os
import sys
import numpy as np
from tqdm import tqdm

# Set seeds for deterministic behavior
torch.manual_seed(42)
np.random.seed(42)

# Add the current directory to path to import lite_gpt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lite_gpt import _load_data_shard


def save_tokens_to_bin(tokens, output_filename):
    """Save tokens to .bin file with proper header format"""
    print(f'Creating {output_filename} with {len(tokens):,} tokens')
    
    # Create header for output file (matching lite_saving_data.py format)
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240520  # magic number
    header[1] = 1         # version
    header[2] = len(tokens)  # number of tokens
    
    # Write header and tokens
    with open(output_filename, 'wb') as f:
        # Write header
        f.write(header.numpy().tobytes())
        # Write tokens
        f.write(tokens.numpy().tobytes())
    
    print(f'Successfully created {output_filename} with {len(tokens):,} tokens')


def extract_tokens_for_batches(all_tokens, batch_indices, batch_size, micro_batch_size):
    """Extract tokens corresponding to specific batch indices"""
    micro_batches_per_step = batch_size // micro_batch_size
    selected_tokens_list = []
    
    for step in tqdm(batch_indices, desc="Extracting tokens for selected batches"):
        for micro_idx in range(micro_batches_per_step):
            micro_batch_idx = step * micro_batches_per_step + micro_idx
            start_pos = micro_batch_idx * micro_batch_size
            end_pos = start_pos + micro_batch_size + 1
            
            if end_pos > len(all_tokens):
                break
                
            micro_batch_tokens = all_tokens[start_pos:end_pos]
            selected_tokens_list.append(micro_batch_tokens)
    
    # Concatenate all selected tokens
    if selected_tokens_list:
        selected_tokens = torch.cat(selected_tokens_list, dim=0)
    else:
        selected_tokens = torch.empty(0, dtype=all_tokens.dtype)
    
    return selected_tokens


def extract_tokens_for_retain_batches(all_tokens, forget_batch_indices, total_steps, batch_size, micro_batch_size):
    """Extract tokens for all batches EXCEPT the forget batch indices"""
    # Get all batch indices except the forget ones
    all_batch_indices = set(range(total_steps))
    retain_batch_indices = sorted(list(all_batch_indices - set(forget_batch_indices)))
    
    return extract_tokens_for_batches(all_tokens, retain_batch_indices, batch_size, micro_batch_size)


class TrainingDataLoader:
    """DataLoader that matches the training behavior from lite_gpt.py"""
    
    def __init__(self, tokens, batch_size, micro_batch_size, selected_batch_indices=None):
        self.tokens = tokens
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.micro_batches_per_step = batch_size // micro_batch_size
        self.num_micro_batches = len(tokens) // (micro_batch_size + 1)
        self.num_steps = self.num_micro_batches // self.micro_batches_per_step
        self.selected_batch_indices = selected_batch_indices
        
    def __len__(self):
        if self.selected_batch_indices is not None:
            return len(self.selected_batch_indices)
        return self.num_steps
    
    def __iter__(self):
        steps_to_iterate = self.selected_batch_indices if self.selected_batch_indices is not None else range(self.num_steps)
        
        for step in steps_to_iterate:
            step_micro_batches = []
            
            for micro_idx in range(self.micro_batches_per_step):
                micro_batch_idx = step * self.micro_batches_per_step + micro_idx
                start_pos = micro_batch_idx * self.micro_batch_size
                end_pos = start_pos + self.micro_batch_size + 1
                
                if end_pos > len(self.tokens):
                    break
                    
                micro_batch_tokens = self.tokens[start_pos:end_pos]
                inputs = micro_batch_tokens[:-1]  # input tokens
                targets = micro_batch_tokens[1:]  # target tokens (shifted by 1)
                
                step_micro_batches.append({
                    'inputs': inputs,
                    'targets': targets,
                    'micro_batch_idx': micro_batch_idx,
                    'start_pos': start_pos,
                    'end_pos': end_pos - 1  # exclude the extra token used for targets
                })
            
            yield {
                'step': step,
                'micro_batches': step_micro_batches,
                'total_tokens': len(step_micro_batches) * self.micro_batch_size
            }


def load_training_data_and_select_subset(target_percentage=0.01):
    """Load training data and select approximately target_percentage of batches randomly but deterministically"""
    
    # Configuration matching lite_saving_data.py
    batch_size = 8*64*1024  # 524,288 tokens per step  
    micro_batch_size = 64 * 1024  # 65,536 tokens per micro batch
    subset_bin_path = 'data/fineweb10B/fineweb_train_subset.bin'
    
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)
    print(f"Batch size: {batch_size:,} tokens")
    print(f"Micro batch size: {micro_batch_size:,} tokens") 
    print(f"Micro batches per step: {batch_size // micro_batch_size}")
    print(f"Loading data from: {subset_bin_path}")
    
    # Load the training subset data
    tokens = _load_data_shard(subset_bin_path)
    print(f'Loaded {len(tokens):,} tokens from {subset_bin_path}')
    
    # Calculate total number of available batches
    num_micro_batches = len(tokens) // (micro_batch_size + 1)
    micro_batches_per_step = batch_size // micro_batch_size
    num_training_steps = num_micro_batches // micro_batches_per_step
    
    print(f'Number of micro batches: {num_micro_batches}')
    print(f'Number of training steps (batches): {num_training_steps}')
    print(f'Total tokens in all batches: {num_training_steps * batch_size:,}')
    
    # Calculate target percentage of the data in terms of batches
    total_tokens_available = num_training_steps * batch_size
    target_tokens = int(total_tokens_available * target_percentage)
    target_batches = max(1, target_tokens // batch_size)  # At least 1 batch
    
    print("\n" + "=" * 60)
    print(f"{target_percentage * 100}% DATA SELECTION")
    print("=" * 60)
    print(f"Target percentage: {target_percentage * 100}%")
    print(f"Target tokens: {target_tokens:,}")
    print(f"Target batches: {target_batches}")
    print(f"Actual tokens with {target_batches} batches: {target_batches * batch_size:,}")
    print(f"Actual percentage: {(target_batches * batch_size / total_tokens_available) * 100:.3f}%")
    
    # Randomly select batch indices with fixed seed for deterministic behavior
    # Use different seed for different percentages to get different selections
    seed = 42 + int(target_percentage * 1000)  # e.g., 42 for 1%, 92 for 5%
    np.random.seed(seed)  # Ensure deterministic selection
    all_batch_indices = np.arange(num_training_steps)
    selected_batch_indices = np.random.choice(all_batch_indices, size=target_batches, replace=False)
    selected_batch_indices = np.sort(selected_batch_indices)  # Sort for easier tracking
    
    print(f"\nSelected batch indices (first 10): {selected_batch_indices[:10]}")
    if len(selected_batch_indices) > 10:
        print(f"Selected batch indices (last 10): {selected_batch_indices[-10:]}")
    print(f"Total selected batches: {len(selected_batch_indices)}")
    print(f"First batch index: {selected_batch_indices[0]}")
    print(f"Last batch index: {selected_batch_indices[-1]}")
    
    return tokens, batch_size, micro_batch_size, selected_batch_indices, num_training_steps


def create_dataloader_and_demonstrate(tokens, batch_size, micro_batch_size, selected_batch_indices):
    """Create dataloader with selected batches and demonstrate usage"""
    
    print("\n" + "=" * 60)
    print("CREATING DATALOADER")
    print("=" * 60)
    
    # Create dataloader with selected batches
    dataloader = TrainingDataLoader(tokens, batch_size, micro_batch_size, selected_batch_indices)
    print(f"DataLoader created with {len(dataloader)} selected training steps")
    
    # Demonstrate usage with all selected batches
    print("\n" + "=" * 60)
    print("DATALOADER DEMONSTRATION")
    print("=" * 60)
    
    demo_count = len(dataloader)
    for i, batch in enumerate(dataloader):
        if i >= demo_count:
            break
            
        print(f"\nBatch {i+1}/{demo_count} (Step {batch['step']}):")
        print(f"  Total tokens in step: {batch['total_tokens']:,}")
        print(f"  Number of micro batches: {len(batch['micro_batches'])}")
        
        # Show details of first micro batch in this step
        first_micro = batch['micro_batches'][0]
        print(f"  First micro batch details:")
        print(f"    Micro batch index: {first_micro['micro_batch_idx']}")
        print(f"    Token positions: {first_micro['start_pos']:,} to {first_micro['end_pos']:,}")
        print(f"    Input shape: {first_micro['inputs'].shape}")
        print(f"    Target shape: {first_micro['targets'].shape}")
        print(f"    First 10 input tokens: {first_micro['inputs'][:10].tolist()}")
        print(f"    First 10 target tokens: {first_micro['targets'][:10].tolist()}")
    
    return dataloader


def process_and_save_percentage(tokens, batch_size, micro_batch_size, num_training_steps, target_percentage):
    """Process a specific percentage and save forget/retain .bin files"""
    
    # Get selection for this percentage
    _, _, _, selected_batch_indices, _ = load_training_data_and_select_subset(target_percentage)
    
    # Create output directory if it doesn't exist
    output_dir = 'data/fineweb10B'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output filenames
    pct_str = f"{int(target_percentage * 100)}pct"
    forget_filename = f'{output_dir}/{pct_str}-forget-random.bin'
    retain_filename = f'{output_dir}/{pct_str}-retain-random.bin'
    
    print(f"\n" + "=" * 60)
    print(f"SAVING {target_percentage * 100}% FORGET/RETAIN DATASETS")
    print("=" * 60)
    
    # Extract and save forget tokens (selected batches)
    print(f"\nExtracting forget tokens ({len(selected_batch_indices)} batches)...")
    forget_tokens = extract_tokens_for_batches(tokens, selected_batch_indices, batch_size, micro_batch_size)
    save_tokens_to_bin(forget_tokens, forget_filename)
    
    # Extract and save retain tokens (remaining batches)
    print(f"\nExtracting retain tokens (remaining batches)...")
    retain_tokens = extract_tokens_for_retain_batches(tokens, selected_batch_indices, num_training_steps, batch_size, micro_batch_size)
    save_tokens_to_bin(retain_tokens, retain_filename)
    
    # Verification
    total_tokens_in_files = len(forget_tokens) + len(retain_tokens)
    expected_tokens = num_training_steps * batch_size
    
    print(f"\n" + "=" * 60)
    print(f"{target_percentage * 100}% DATASET VERIFICATION")
    print("=" * 60)
    print(f"Forget tokens: {len(forget_tokens):,}")
    print(f"Retain tokens: {len(retain_tokens):,}")
    print(f"Total tokens in files: {total_tokens_in_files:,}")
    print(f"Expected total tokens: {expected_tokens:,}")
    print(f"Match: {'✓' if total_tokens_in_files == expected_tokens else '✗'}")
    
    return selected_batch_indices, forget_filename, retain_filename


def main():
    """Main function to run the data loading and selection process"""
    print("DETERMINISTIC RANDOM DATA SELECTION SCRIPT")
    print("Processing 1% and 5% selections with fixed seeds for reproducible results")
    
    # Load data initially with 1% to get the base data
    tokens, batch_size, micro_batch_size, selected_batch_indices_1pct, num_training_steps = load_training_data_and_select_subset(0.01)
    
    # Create dataloader and demonstrate for 1%
    dataloader_1pct = create_dataloader_and_demonstrate(tokens, batch_size, micro_batch_size, selected_batch_indices_1pct)
    
    # Process and save 1% datasets
    selected_1pct, forget_1pct_file, retain_1pct_file = process_and_save_percentage(
        tokens, batch_size, micro_batch_size, num_training_steps, 0.01)
    
    # Process and save 5% datasets  
    selected_5pct, forget_5pct_file, retain_5pct_file = process_and_save_percentage(
        tokens, batch_size, micro_batch_size, num_training_steps, 0.05)
    
    print(f"\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"✓ Loaded {len(tokens):,} total tokens")
    print(f"✓ Total training steps available: {num_training_steps}")
    print(f"✓ Each batch contains {batch_size:,} tokens")
    print(f"\n1% Selection:")
    print(f"  ✓ Selected {len(selected_1pct)} batches")
    print(f"  ✓ Forget dataset: {forget_1pct_file}")
    print(f"  ✓ Retain dataset: {retain_1pct_file}")
    print(f"\n5% Selection:")
    print(f"  ✓ Selected {len(selected_5pct)} batches")
    print(f"  ✓ Forget dataset: {forget_5pct_file}")
    print(f"  ✓ Retain dataset: {retain_5pct_file}")
    print(f"\n✓ All operations are deterministic with fixed seeds")
    
    return {
        'tokens': tokens,
        'dataloader_1pct': dataloader_1pct,
        '1pct_indices': selected_1pct,
        '5pct_indices': selected_5pct,
        'files': {
            '1pct_forget': forget_1pct_file,
            '1pct_retain': retain_1pct_file,
            '5pct_forget': forget_5pct_file,
            '5pct_retain': retain_5pct_file
        }
    }


if __name__ == "__main__":
    results = main()
