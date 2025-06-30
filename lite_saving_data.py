import glob
import torch
import os
import sys
import argparse
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lite_gpt import GPT, _load_data_shard

def calculate_token_processing_pattern(train_bin_pattern, batch_size, num_iterations):
    """Calculate exactly which tokens will be processed during training"""
    total_tokens_to_process = batch_size * num_iterations
    print(f'Will process exactly {total_tokens_to_process:,} tokens over {num_iterations} steps')
    
    # Simulate the data loader's deterministic behavior
    files = sorted(glob.glob(train_bin_pattern))
    print(f'Training files in order: {files}')
    
    tokens_processed = 0
    file_idx = 0
    file_position = 0
    
    while tokens_processed < total_tokens_to_process and file_idx < len(files):
        # Load file header to get token count
        header = torch.from_file(files[file_idx], False, 256, dtype=torch.int32)
        file_tokens = int(header[2])
        
        remaining_in_file = file_tokens - file_position
        tokens_needed = total_tokens_to_process - tokens_processed
        
        tokens_from_this_file = min(remaining_in_file, tokens_needed)
        
        print(f'File {file_idx} ({files[file_idx]}): '
              f'positions {file_position}:{file_position + tokens_from_this_file} '
              f'({tokens_from_this_file:,} tokens)')
        
        tokens_processed += tokens_from_this_file
        file_position += tokens_from_this_file
        
        if file_position >= file_tokens:
            file_idx += 1
            file_position = 0
    
    print(f'Total tokens that will be processed: {tokens_processed:,}')
    return tokens_processed

def create_training_subset_file(train_bin_pattern, batch_size, num_iterations, output_filename):
    """Create a single .bin file containing exactly the tokens that will be processed during training"""
    total_tokens_to_process = batch_size * num_iterations
    print(f'Creating {output_filename} with {total_tokens_to_process:,} tokens')
    
    files = sorted(glob.glob(train_bin_pattern))
    
    # Create header for output file
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240520  # magic number
    header[1] = 1         # version
    header[2] = total_tokens_to_process  # number of tokens
    
    # Write header and collect tokens
    with open(output_filename, 'wb') as f:
        # Write header
        f.write(header.numpy().tobytes())
        
        tokens_processed = 0
        file_idx = 0
        file_position = 0
        
        while tokens_processed < total_tokens_to_process and file_idx < len(files):
            print(f'Processing file {file_idx}: {files[file_idx]}')
            
            # Load current file's tokens
            with open(files[file_idx], 'rb') as source_file:
                source_file.seek(256 * 4)  # Skip header
                source_tokens = torch.empty(torch.from_file(files[file_idx], False, 256, dtype=torch.int32)[2], dtype=torch.uint16)
                source_file.readinto(source_tokens.numpy())
            
            # Calculate how many tokens to take from this file
            remaining_in_file = len(source_tokens) - file_position
            tokens_needed = total_tokens_to_process - tokens_processed
            tokens_from_this_file = min(remaining_in_file, tokens_needed)
            
            # Extract the relevant slice and write to output
            tokens_slice = source_tokens[file_position:file_position + tokens_from_this_file]
            f.write(tokens_slice.numpy().tobytes())
            
            tokens_processed += tokens_from_this_file
            file_position += tokens_from_this_file
            
            if file_position >= len(source_tokens):
                file_idx += 1
                file_position = 0
    
    print(f'Successfully created {output_filename} with {tokens_processed:,} tokens')

def extract_batch_losses_from_checkpoint(checkpoint_path, bin_path, batch_size, output_path, split_batch_factor=1):
    """
    Load a checkpoint and extract loss values for each batch from the training data.
    Returns and saves a dictionary of batch index -> loss value.
    
    Args:
        split_batch_factor: Factor by which to split micro batches for memory efficiency.
                          Loss will be averaged across splits (like gradient accumulation).
    """
    print(f'Loading checkpoint from: {checkpoint_path}')
    print(f'Processing data from: {bin_path}')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model with same configuration as in lite_gpt.py
    model = GPT(vocab_size=50304, num_layers=12, num_heads=6, model_dim=768)
    
    # Handle state dict from parallel training (strip _orig_mod. prefix)
    state_dict = checkpoint['model']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("Detected parallel training checkpoint, stripping _orig_mod. prefix...")
        state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    
    # Load the training data
    tokens = _load_data_shard(bin_path)
    print(f'Loaded {len(tokens):,} tokens from {bin_path}')
    
    # Use micro batch size like in lite_gpt.py (64*1024 tokens)
    micro_batch_size = 64 * 1024
    print(f'Using micro batch size: {micro_batch_size:,} tokens')
    print(f'Split batch factor: {split_batch_factor}')
    
    # Calculate actual processing size after splitting
    split_micro_batch_size = micro_batch_size // split_batch_factor
    print(f'Effective micro batch size after splitting: {split_micro_batch_size:,} tokens')
    
    # Calculate number of micro batches we can process
    num_micro_batches = len(tokens) // (micro_batch_size + 1)
    print(f'Will process {num_micro_batches} micro batches')
    
    # Calculate how many micro batches per training step
    micro_batches_per_step = batch_size // micro_batch_size
    num_training_steps = num_micro_batches // micro_batches_per_step
    print(f'This corresponds to {num_training_steps} training steps ({micro_batches_per_step} micro batches per step)')
    
    batch_losses = {}
    
    with torch.no_grad():
        for step in tqdm(range(num_training_steps), desc="Processing training steps"):
            step_losses = []
            
            for micro_idx in range(micro_batches_per_step):
                micro_batch_idx = step * micro_batches_per_step + micro_idx
                start_pos = micro_batch_idx * micro_batch_size
                end_pos = start_pos + micro_batch_size + 1
                
                if end_pos > len(tokens):
                    break
                    
                # Get micro batch tokens
                micro_batch_tokens = tokens[start_pos:end_pos]
                
                # Split micro batch if needed for memory efficiency
                split_losses = []
                for split_idx in range(split_batch_factor):
                    split_start = split_idx * split_micro_batch_size
                    split_end = min(split_start + split_micro_batch_size + 1, len(micro_batch_tokens))
                    
                    if split_end <= split_start + 1:
                        break
                        
                    split_tokens = micro_batch_tokens[split_start:split_end]
                    inputs = split_tokens[:-1].to(device='cuda', dtype=torch.int32)
                    targets = split_tokens[1:].to(device='cuda', dtype=torch.int64)
                    
                    # Calculate loss for this split
                    # Use sliding window that matches training (starts at ~1 and increases to ~14)
                    progress = step / max(1, num_training_steps - 1)
                    sliding_window_blocks = int(((1 - progress) * 128 + progress * 1856) // 128)
                    sliding_window_num_blocks = torch.tensor(sliding_window_blocks, dtype=torch.int32, device='cuda')
                    
                    with torch.no_grad():  # Extra safety to ensure no gradients
                        loss = model(inputs, targets, sliding_window_num_blocks)
                        split_losses.append(loss.item())
                
                # Average loss across splits for this micro batch
                if split_losses:
                    avg_micro_loss = sum(split_losses) / len(split_losses)
                    step_losses.append(avg_micro_loss)
            
            # Average loss across micro batches for this training step
            if step_losses:
                avg_step_loss = sum(step_losses) / len(step_losses)
                batch_losses[step] = avg_step_loss
                
                if step % 100 == 0:
                    print(f'Step {step}: avg_loss = {avg_step_loss:.4f} (from {len(step_losses)} micro batches, {split_batch_factor} splits each)')
    
    print(f'Processed {len(batch_losses)} training steps')
    print(f'Average loss: {sum(batch_losses.values()) / len(batch_losses):.4f}')
    
    # Save the dictionary
    torch.save(batch_losses, output_path)
    print(f'Saved batch losses to: {output_path}')
    
    return batch_losses

def load_and_inspect_batch_losses(losses_path, split_batch_factor=1):
    """
    Load and inspect the saved batch losses dictionary.
    
    Args:
        split_batch_factor: For display purposes - shows the split factor used during extraction.
    """
    if not os.path.exists(losses_path):
        print(f"Losses file not found: {losses_path}")
        return None
    
    batch_losses = torch.load(losses_path, map_location='cpu')
    
    print(f"Loaded batch losses from: {losses_path}")
    print(f"Number of training steps: {len(batch_losses)}")
    print(f"Split batch factor used: {split_batch_factor}")
    
    if batch_losses:
        losses = list(batch_losses.values())
        steps = list(batch_losses.keys())
        
        print(f"Step range: {min(steps)} to {max(steps)}")
        print(f"Loss range: {min(losses):.4f} to {max(losses):.4f}")
        print(f"Average loss: {sum(losses) / len(losses):.4f}")
        
        # Show first and last few entries
        print("\nFirst 5 entries:")
        for step in sorted(steps)[:5]:
            print(f"  Step {step}: {batch_losses[step]:.4f}")
        
        if len(steps) > 5:
            print("\nLast 5 entries:")
            for step in sorted(steps)[-5:]:
                print(f"  Step {step}: {batch_losses[step]:.4f}")
    
    return batch_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract batch losses from training checkpoint')
    parser.add_argument('--split_batch_factor', type=int, default=1, 
                        help='Factor by which to split micro batches for memory efficiency (default: 1). Increase if you get CUDA OOM errors.')
    parser.add_argument('--checkpoint_path', type=str, default='logs/0e7c9660-7014-4267-b5f7-fc4ebf0625cc/state_step001390.pt',
                        help='Path to the checkpoint file')
    parser.add_argument('--losses_output_path', type=str, default='data/fineweb10B/batch_losses.pt',
                        help='Path to save the extracted batch losses')
    
    args = parser.parse_args()
    
    # Default hyperparameters from lite_gpt.py
    train_bin = 'data/fineweb10B/fineweb_train_*.bin'
    batch_size = 8*64*1024  # 524,288 tokens per step  
    num_iterations = 1390
    output_file = 'data/fineweb10B/fineweb_train_subset.bin'
    
    # Check if the subset file already exists
    if os.path.exists(output_file):
        print(f'Found existing subset file: {output_file}')
        print('Skipping token processing and subset creation')
    else:
        print(f'Creating new subset file: {output_file}')
        calculate_token_processing_pattern(train_bin, batch_size, num_iterations)
        create_training_subset_file(train_bin, batch_size, num_iterations, output_file)

    # Extract batch losses from checkpoint
    bin_path = output_file
    
    if os.path.exists(args.checkpoint_path):
        print(f'\nExtracting batch losses from checkpoint...')
        batch_losses = extract_batch_losses_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            bin_path=bin_path, 
            batch_size=batch_size,
            output_path=args.losses_output_path,
            split_batch_factor=args.split_batch_factor
        )
        
        # Inspect the saved losses
        print(f'\nInspecting saved batch losses...')
        load_and_inspect_batch_losses(args.losses_output_path, args.split_batch_factor)
        
    else:
        print(f'Checkpoint not found: {args.checkpoint_path}')
        print('Please update the checkpoint_path argument with the correct path')
        
        # If checkpoint doesn't exist but losses file does, just inspect it
        if os.path.exists(args.losses_output_path):
            print(f'\nFound existing batch losses file, inspecting...')
            load_and_inspect_batch_losses(args.losses_output_path, args.split_batch_factor)