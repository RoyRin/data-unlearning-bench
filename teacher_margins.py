#!/usr/bin/env python3
"""
Script to run forward passes with a trained GPT model on data and compute margins.
Usage: python teacher_margins.py <model_path> <data_path> [--margins-folder <folder>]
For example: python teacher_margins.py logs/0e7c9660-7014-4267-b5f7-fc4ebf0625cc/state_step001390.pt data/fineweb10B/fineweb_val_000000.bin --margins-folder ./margins
"""

import argparse
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import hashlib

# Performance optimization imports
from torch.cuda.amp import autocast as cuda_autocast
from torch import compile as torch_compile          # PyTorch ≥2.0

# Import from lite_gpt.py
from lite_gpt import GPT, _load_data_shard

# --------------------------------------------------
# Performance optimization flags:
# getting 20x speedup on single model single gpu
# --------------------------------------------------
USE_AMP    = True          # turn off if you truly need fp32 everywhere
COMPILE    = True          # turn off for PyTorch <2.0 or debugging

# --------------------------------------------------
# Helper utilities
# --------------------------------------------------
def maybe_autocast():
    """Return an autocast context that works on both old & new PyTorch."""
    if USE_AMP:
        # new API (PyTorch ≥2.1) accepts device_type='cuda'
        try:
            return torch.amp.autocast('cuda', dtype=torch.bfloat16)
        except (TypeError, AttributeError):                # old API or torch.amp not available
            return cuda_autocast(dtype=torch.bfloat16)
    else:
        # no-op context
        import contextlib
        return contextlib.nullcontext()

def get_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    For each token position, compute the margin of the correct token.
    margin = logit_correct - log_sum_exp(logit_other)
    
    Args:
        logits: (seq_len, vocab_size) tensor of logits
        targets: (seq_len,) tensor of target token ids
    """
    seq_len = logits.shape[0]
    bindex = torch.arange(seq_len).to(logits.device)
    logits_correct = logits[bindex, targets]
    
    # Clone to avoid modifying original logits
    cloned_logits = logits.clone()
    cloned_logits[bindex, targets] = torch.tensor(-torch.inf,
                                                 device=cloned_logits.device,
                                                 dtype=cloned_logits.dtype)
    return logits_correct - cloned_logits.logsumexp(dim=-1)

def get_margins_filename(model_path, data_path, margins_folder=None):
    """Generate a filename for storing margins based on model and data paths."""
    model_path = Path(model_path)
    data_path = Path(data_path)
    
    # Create a hash of the data path to handle long filenames
    data_hash = hashlib.md5(str(data_path).encode()).hexdigest()[:8]
    
    # Extract model filename without extension
    model_name = model_path.stem  # e.g., "state_step001390"
    
    # Create margins filename
    margins_filename = f"{model_name}_margins_{data_path.stem}_{data_hash}.pt"
    
    # Use specified margins folder or default to model directory
    if margins_folder is not None:
        margins_dir = Path(margins_folder)
        margins_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        margins_path = margins_dir / margins_filename
    else:
        # Save in the same directory as the model (original behavior)
        margins_path = model_path.parent / margins_filename
    
    return margins_path

def load_existing_margins(margins_path):
    """Load margins if they exist, return None otherwise."""
    if margins_path.exists():
        print(f"Found existing margins: {margins_path}")
        margins_data = torch.load(margins_path, map_location='cpu')
        print(f"Loaded {len(margins_data['margins']):,} precomputed margins")
        return margins_data
    return None

def save_margins(margins_path, margins, model_path, data_path, margin_stats):
    """Save margins along with metadata."""
    margins_data = {
        'margins': margins,
        'model_path': str(model_path),
        'data_path': str(data_path),
        'num_tokens': len(margins),
        'margin_stats': margin_stats
    }
    
    torch.save(margins_data, margins_path)
    print(f"Saved margins to: {margins_path}")

def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same parameters as training
    model = GPT(vocab_size=50304, num_layers=12, num_heads=6, model_dim=768)
    
    # Handle state dict from parallel training (strip _orig_mod. prefix)
    state_dict = checkpoint['model']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("Detected parallel training checkpoint, stripping _orig_mod. prefix...")
        state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model

def compute_margins(model_path, data_path, margins_folder=None):
    """Compute margins for the given model and data, with caching."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if margins already exist
    margins_path = get_margins_filename(model_path, data_path, margins_folder)
    existing_margins = load_existing_margins(margins_path)
    
    if existing_margins is not None:
        print("Using existing margins, skipping computation.")
        return existing_margins
    
    print("Computing new margins...")
    
    # Load model
    model = load_model(model_path, device)
    
    # Compile model for better performance
    if COMPILE and hasattr(torch, 'compile'):
        print("Compiling model for better performance...")
        model = torch_compile(model, mode='reduce-overhead')
    
    # Load data
    print(f"Loading data from {data_path}")
    tokens = _load_data_shard(data_path)
    print(f"Loaded {len(tokens):,} tokens")
    
    # Use same batch size as training
    micro_bs = 16 * 1024  # original is 64 * 1024, reduced to prevent oom
    sliding_window_num_blocks = torch.tensor(14, dtype=torch.int32, device=device)  # reasonable default
    
    all_margins = []
    num_batches = 0
    
    # Process data in batches with progress bar
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(tokens) - micro_bs, micro_bs), desc="Computing margins"):
            end_idx = min(start_idx + micro_bs, len(tokens) - 1)
            
            # Get input and target tokens
            batch_tokens = tokens[start_idx:end_idx + 1]
            inputs = batch_tokens[:-1].to(device, non_blocking=True, dtype=torch.int32)
            targets = batch_tokens[1:].to(device, non_blocking=True, dtype=torch.long)
            
            # Forward pass with skip_loss=True using AMP
            with maybe_autocast():
                logits = model(inputs, targets, sliding_window_num_blocks, skip_loss=True)
            
            # Compute margins for each token position
            # logits shape: (1, seq_len, vocab_size), squeeze to (seq_len, vocab_size)
            logits = logits.squeeze(0)      # (seq_len, vocab_size)
            margins = get_margin(logits.float(), targets)   # cast back to fp32 for exactness
            all_margins.append(margins.cpu())
            num_batches += 1
    
    # Concatenate all margins and compute statistics
    all_margins = torch.cat(all_margins)
    
    margin_stats = {
        'mean': all_margins.mean().item(),
        'std': all_margins.std().item(),
        'min': all_margins.min().item(),
        'max': all_margins.max().item(),
        'num_batches': num_batches
    }
    
    print(f"\nCompleted margin computation on {num_batches:,} batches")
    print(f"Total tokens processed: {len(all_margins):,}")
    print(f"Margin statistics:")
    print(f"  Average: {margin_stats['mean']:.4f}")
    print(f"  Std Dev: {margin_stats['std']:.4f}")
    print(f"  Min: {margin_stats['min']:.4f}")
    print(f"  Max: {margin_stats['max']:.4f}")
    
    # Save margins
    save_margins(margins_path, all_margins, model_path, data_path, margin_stats)
    
    margins_data = {
        'margins': all_margins,
        'model_path': str(model_path),
        'data_path': str(data_path),
        'num_tokens': len(all_margins),
        'margin_stats': margin_stats
    }
    
    return margins_data

def main():
    parser = argparse.ArgumentParser(description="Compute and save margins for GPT model on data")
    parser.add_argument("model_path", help="Path to model checkpoint (.pt file)")
    parser.add_argument("data_path", help="Path to data shard (.bin file)")
    parser.add_argument("--margins-folder", help="Directory to save margins files (default: same as model directory)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model_path).exists():
        print(f"Error: Model path {args.model_path} does not exist")
        sys.exit(1)
    
    if not Path(args.data_path).exists():
        print(f"Error: Data path {args.data_path} does not exist")
        sys.exit(1)
    
    margins_data = compute_margins(args.model_path, args.data_path, args.margins_folder)
    print(f"\nMargins computation completed for:")
    print(f"  Model: {margins_data['model_path']}")
    print(f"  Data: {margins_data['data_path']}")
    print(f"  Total tokens: {margins_data['num_tokens']:,}")

if __name__ == "__main__":
    main()
