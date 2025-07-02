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
    
    print(f"Extracting tokens for {len(batch_indices)} batches...")
    print(f"Total tokens available: {len(all_tokens):,}")
    print(f"Batch size: {batch_size:,}, Micro batch size: {micro_batch_size:,}")
    print(f"Micro batches per step: {micro_batches_per_step}")
    
    tokens_extracted = 0
    
    for step in tqdm(batch_indices, desc="Extracting tokens for selected batches"):
        for micro_idx in range(micro_batches_per_step):
            micro_batch_idx = step * micro_batches_per_step + micro_idx
            start_pos = micro_batch_idx * micro_batch_size
            end_pos = start_pos + micro_batch_size + 1
            
            if end_pos > len(all_tokens):
                print(f"WARNING: Batch {step}, micro batch {micro_idx} exceeds token limit")
                print(f"  Required end_pos: {end_pos:,}, Available tokens: {len(all_tokens):,}")
                break
                
            micro_batch_tokens = all_tokens[start_pos:end_pos]
            selected_tokens_list.append(micro_batch_tokens)
            tokens_extracted += len(micro_batch_tokens)
    
    print(f"Tokens extracted: {tokens_extracted:,}")
    print(f"Expected tokens: {len(batch_indices) * batch_size:,}")
    
    # Concatenate all selected tokens
    if selected_tokens_list:
        selected_tokens = torch.cat(selected_tokens_list, dim=0)
    else:
        selected_tokens = torch.empty(0, dtype=all_tokens.dtype)
    
    print(f"Final concatenated tokens: {len(selected_tokens):,}")
    return selected_tokens


def extract_tokens_for_retain_batches(all_tokens, forget_batch_indices, total_steps, batch_size, micro_batch_size):
    """Extract tokens for all batches EXCEPT the forget batch indices"""
    # Get all batch indices except the forget ones
    all_batch_indices = set(range(total_steps))
    retain_batch_indices = sorted(list(all_batch_indices - set(forget_batch_indices)))
    
    print(f"Retain extraction: {len(retain_batch_indices)} retain batches vs {len(forget_batch_indices)} forget batches")
    print(f"Total expected batches: {total_steps}")
    print(f"Retain + forget = {len(retain_batch_indices) + len(forget_batch_indices)}")
    
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


def load_batch_losses(batch_losses_path):
    """Load batch losses and return sorted indices (highest loss first)"""
    print(f"Loading batch losses from: {batch_losses_path}")
    
    # Load the batch losses
    batch_losses_dict = torch.load(batch_losses_path, map_location='cpu')
    print(f"Loaded batch losses as dict with {len(batch_losses_dict)} entries")
    
    # Convert dictionary to tensors
    batch_indices = torch.tensor(list(batch_losses_dict.keys()))
    batch_losses = torch.tensor(list(batch_losses_dict.values()))
    
    print(f"Batch indices range: {batch_indices.min().item()} to {batch_indices.max().item()}")
    print(f"Batch losses shape: {batch_losses.shape}")
    print(f"Loss statistics:")
    print(f"  Min loss: {batch_losses.min().item():.6f}")
    print(f"  Max loss: {batch_losses.max().item():.6f}")
    print(f"  Mean loss: {batch_losses.mean().item():.6f}")
    print(f"  Std loss: {batch_losses.std().item():.6f}")
    
    # Sort indices by loss (descending order - highest loss first)
    sorted_loss_indices = torch.argsort(batch_losses, descending=True)
    sorted_batch_indices = batch_indices[sorted_loss_indices]
    
    print(f"Highest loss batches (first 10): {sorted_batch_indices[:10].tolist()}")
    print(f"Corresponding losses: {batch_losses[sorted_loss_indices[:10]].tolist()}")
    
    return batch_losses_dict, batch_indices, batch_losses, sorted_batch_indices


def select_batches_by_loss(sorted_batch_indices, target_percentage, num_training_steps, batch_indices):
    """Select top batches based on highest losses"""
    target_batches = max(1, int(num_training_steps * target_percentage))
    selected_batch_indices = sorted_batch_indices[:target_batches].numpy()
    selected_batch_indices = np.sort(selected_batch_indices)  # Sort for easier tracking
    
    # Assert that selected indices are valid
    assert all(idx >= 0 for idx in selected_batch_indices), "Found negative batch index"
    assert all(idx < num_training_steps for idx in selected_batch_indices), f"Found batch index >= {num_training_steps}"
    
    # Assert that all selected indices exist in the batch_indices
    available_indices_set = set(batch_indices.numpy())
    selected_indices_set = set(selected_batch_indices)
    missing_indices = selected_indices_set - available_indices_set
    assert len(missing_indices) == 0, f"Selected indices not found in batch losses: {missing_indices}"
    
    print(f"\n" + "=" * 60)
    print(f"{target_percentage * 100}% LOSS-BASED SELECTION")
    print("=" * 60)
    print(f"Target percentage: {target_percentage * 100}%")
    print(f"Target batches: {target_batches}")
    print(f"Selected batch indices (first 10): {selected_batch_indices[:10]}")
    if len(selected_batch_indices) > 10:
        print(f"Selected batch indices (last 10): {selected_batch_indices[-10:]}")
    print(f"✓ All selected indices are valid and exist in batch losses")
    
    return selected_batch_indices


def process_and_save_percentage_loss(tokens, batch_size, micro_batch_size, num_training_steps, 
                                   target_percentage, sorted_batch_indices, batch_losses, batch_indices):
    """Process a specific percentage based on highest losses and save forget/retain .bin files"""
    
    # Get selection for this percentage based on loss
    selected_batch_indices = select_batches_by_loss(sorted_batch_indices, target_percentage, num_training_steps, batch_indices)
    
    # Create output directory if it doesn't exist
    output_dir = 'data/fineweb10B'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output filenames (using 'loss' instead of 'random')
    pct_str = f"{int(target_percentage * 100)}pct"
    forget_filename = f'{output_dir}/{pct_str}-forget-loss.bin'
    retain_filename = f'{output_dir}/{pct_str}-retain-loss.bin'
    
    print(f"\n" + "=" * 60)
    print(f"SAVING {target_percentage * 100}% LOSS-BASED FORGET/RETAIN DATASETS")
    print("=" * 60)
    
    # Show loss statistics for selected batches
    # Map selected batch indices to their loss values
    batch_losses_dict_from_params = {batch_indices[i].item(): batch_losses[i].item() for i in range(len(batch_indices))}
    selected_losses = torch.tensor([batch_losses_dict_from_params[idx] for idx in selected_batch_indices])
    print(f"Selected batches loss statistics:")
    print(f"  Min loss: {selected_losses.min().item():.6f}")
    print(f"  Max loss: {selected_losses.max().item():.6f}")
    print(f"  Mean loss: {selected_losses.mean().item():.6f}")
    print(f"  Std loss: {selected_losses.std().item():.6f}")
    
    # Extract and save forget tokens (selected batches with highest losses)
    print(f"\nExtracting forget tokens ({len(selected_batch_indices)} highest-loss batches)...")
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
    print(f"{target_percentage * 100}% LOSS-BASED DATASET VERIFICATION")
    print("=" * 60)
    print(f"Forget tokens: {len(forget_tokens):,}")
    print(f"Retain tokens: {len(retain_tokens):,}")
    print(f"Total tokens in files: {total_tokens_in_files:,}")
    print(f"Expected total tokens: {expected_tokens:,}")
    print(f"Match: {'✓' if total_tokens_in_files == expected_tokens else '✗'}")
    
    return selected_batch_indices, forget_filename, retain_filename


def main():
    """Main function to run the data loading and selection process"""
    print("DATA SELECTION SCRIPT: RANDOM AND LOSS-BASED MODES")
    print("Processing 1% and 5% selections with both random and loss-based methods")
    
    # Load data initially with 1% to get the base data
    tokens, batch_size, micro_batch_size, selected_batch_indices_1pct, num_training_steps = load_training_data_and_select_subset(0.01)
    
    # Create dataloader and demonstrate for 1%
    dataloader_1pct = create_dataloader_and_demonstrate(tokens, batch_size, micro_batch_size, selected_batch_indices_1pct)
    
    print(f"\n" + "=" * 80)
    print("RANDOM-BASED SELECTION")
    print("=" * 80)
    
    # Process and save 1% datasets (random)
    selected_1pct_random, forget_1pct_random_file, retain_1pct_random_file = process_and_save_percentage(
        tokens, batch_size, micro_batch_size, num_training_steps, 0.01)
    
    # Process and save 5% datasets (random)
    selected_5pct_random, forget_5pct_random_file, retain_5pct_random_file = process_and_save_percentage(
        tokens, batch_size, micro_batch_size, num_training_steps, 0.05)
    
    print(f"\n" + "=" * 80)
    print("LOSS-BASED SELECTION")
    print("=" * 80)
    
    # Load batch losses and get sorted indices
    batch_losses_path = '/home/ppol/data-unlearning-bench/data/fineweb10B/batch_losses.pt'
    batch_losses_dict, batch_indices, batch_losses, sorted_batch_indices = load_batch_losses(batch_losses_path)
    
    # Verify that the number of losses matches the number of training steps
    if len(batch_losses) != num_training_steps:
        print(f"WARNING: Number of batch losses ({len(batch_losses)}) doesn't match number of training steps ({num_training_steps})")
        print("Using the minimum of the two for safety")
        max_steps = min(len(batch_losses), num_training_steps)
        sorted_batch_indices = sorted_batch_indices[:max_steps]
    
    # Process and save 1% datasets (loss-based)
    selected_1pct_loss, forget_1pct_loss_file, retain_1pct_loss_file = process_and_save_percentage_loss(
        tokens, batch_size, micro_batch_size, num_training_steps, 0.01, sorted_batch_indices, batch_losses, batch_indices)
    
    # Process and save 5% datasets (loss-based)
    selected_5pct_loss, forget_5pct_loss_file, retain_5pct_loss_file = process_and_save_percentage_loss(
        tokens, batch_size, micro_batch_size, num_training_steps, 0.05, sorted_batch_indices, batch_losses, batch_indices)
    
    print(f"\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"✓ Loaded {len(tokens):,} total tokens")
    print(f"✓ Total training steps available: {num_training_steps}")
    print(f"✓ Each batch contains {batch_size:,} tokens")
    
    print(f"\nRANDOM-BASED SELECTION:")
    print(f"  1% Selection: {len(selected_1pct_random)} batches")
    print(f"    ✓ Forget dataset: {forget_1pct_random_file}")
    print(f"    ✓ Retain dataset: {retain_1pct_random_file}")
    print(f"  5% Selection: {len(selected_5pct_random)} batches")
    print(f"    ✓ Forget dataset: {forget_5pct_random_file}")
    print(f"    ✓ Retain dataset: {retain_5pct_random_file}")
    
    print(f"\nLOSS-BASED SELECTION:")
    print(f"  1% Selection: {len(selected_1pct_loss)} batches")
    print(f"    ✓ Forget dataset: {forget_1pct_loss_file}")
    print(f"    ✓ Retain dataset: {retain_1pct_loss_file}")
    print(f"  5% Selection: {len(selected_5pct_loss)} batches")
    print(f"    ✓ Forget dataset: {forget_5pct_loss_file}")
    print(f"    ✓ Retain dataset: {retain_5pct_loss_file}")
    
    print(f"\n✓ All operations are deterministic")
    print(f"✓ Random selections use fixed seeds")
    print(f"✓ Loss-based selections use highest-loss batches")
    
    return {
        'tokens': tokens,
        'dataloader_1pct': dataloader_1pct,
        'random': {
            '1pct_indices': selected_1pct_random,
            '5pct_indices': selected_5pct_random,
        },
        'loss_based': {
            '1pct_indices': selected_1pct_loss,
            '5pct_indices': selected_5pct_loss,
            'batch_losses': batch_losses,
            'sorted_indices': sorted_batch_indices,
        },
        'files': {
            '1pct_forget_random': forget_1pct_random_file,
            '1pct_retain_random': retain_1pct_random_file,
            '5pct_forget_random': forget_5pct_random_file,
            '5pct_retain_random': retain_5pct_random_file,
            '1pct_forget_loss': forget_1pct_loss_file,
            '1pct_retain_loss': retain_1pct_loss_file,
            '5pct_forget_loss': forget_5pct_loss_file,
            '5pct_retain_loss': retain_5pct_loss_file,
        }
    }


if __name__ == "__main__":
    results = main()
