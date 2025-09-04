#!/usr/bin/env python3
"""
Script to download .pt files from HuggingFace, convert to .npy, and upload back.
Processes files one-by-one to minimize disk usage.
Handles large files (up to 4GB) efficiently with progress tracking.
"""

import os
import sys
import time
import json
import requests
import torch
import numpy as np
import threading
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable
from urllib.parse import quote
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# For uploading, we'll use huggingface_hub as it handles large files better
from huggingface_hub import HfApi, upload_file, login

@dataclass
class FileInfo:
    """Store file information"""
    name: str
    size: int
    path: str

class HuggingFaceConverter:
    def __init__(self, 
                 source_repo: str,
                 source_paths: List[str],
                 target_repo: str,
                 target_paths: List[str],
                 local_dir: str = "./temp_conversion",
                 cleanup_after_upload: bool = True,
                 max_workers: int = 16,
                 rename_functions: Optional[List[Callable[[str], str]]] = None,
                 **kwargs):
        """
        Initialize the converter.
        
        Args:
            source_repo: Source repository ID (e.g., "royrin/KLOM-models")
            source_paths: List of paths within source repo (e.g., ["margins/oracle_loss_1_pct"])
            target_repo: Target repository ID for uploads
            target_paths: List of paths within target repo for uploads (must match source_paths length)
            local_dir: Local directory for temporary storage
            cleanup_after_upload: If True, delete files after successful upload
            max_workers: Maximum number of worker threads for parallel processing (default: 16)
            rename_functions: List of functions to rename files (must match source_paths length if provided)
        """
        # Handle backward compatibility - if old single path args are provided
        if 'source_path' in kwargs and 'target_path' in kwargs:
            source_paths = [kwargs['source_path']]
            target_paths = [kwargs['target_path']]
        
        if len(source_paths) != len(target_paths):
            raise ValueError("source_paths and target_paths must have the same length")
        
        if rename_functions and len(rename_functions) != len(source_paths):
            raise ValueError("rename_functions must have the same length as source_paths if provided")
        
        self.source_repo = source_repo
        self.source_paths = source_paths
        self.target_repo = target_repo
        self.target_paths = target_paths
        self.local_dir = Path(local_dir)
        self.cleanup_after_upload = cleanup_after_upload
        self.max_workers = max_workers
        self.rename_functions = rename_functions or [None] * len(source_paths)
        
        # Create local directories
        self.download_dir = self.local_dir / "downloads"
        self.converted_dir = self.local_dir / "converted"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.converted_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup HuggingFace API - uses cached login from huggingface-cli login
        self.api = HfApi()
        
        # Track processed files across all paths
        self.processed_files = set()
        self.progress_lock = threading.Lock()
        self.load_progress()
    
    def save_progress(self):
        """Save progress to resume later if needed."""
        progress_file = self.local_dir / "progress.json"
        with self.progress_lock:
            with open(progress_file, 'w') as f:
                json.dump(list(self.processed_files), f)
    
    def load_progress(self):
        """Load previous progress if exists."""
        progress_file = self.local_dir / "progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                self.processed_files = set(json.load(f))
            print(f"Resuming from previous run: {len(self.processed_files)} files already processed")
    
    def get_file_info(self) -> Tuple[List[FileInfo], int]:
        """
        Get list of .pt files with their sizes from all source paths in the repository.
        Returns: (list of FileInfo, total size in bytes)
        """
        all_files = []
        total_size = 0
        
        for source_path in self.source_paths:
            print(f"Fetching file list and sizes from {self.source_repo}/{source_path}...")
            
            # Use the tree API to get directory contents with sizes
            api_url = f"https://huggingface.co/api/datasets/{self.source_repo}/tree/main/{source_path}"
            
            headers = {}
            
            try:
                response = requests.get(api_url, headers=headers)
                response.raise_for_status()
                files = response.json()
                
                # Filter for .pt files and get their info
                pt_files = []
                path_size = 0
                
                for f in files:
                    if f['path'].endswith('.pt'):
                        file_info = FileInfo(
                            name=f['path'].split('/')[-1],
                            size=f.get('size', 0),
                            path=f['path']
                        )
                        pt_files.append(file_info)
                        path_size += file_info.size
                
                all_files.extend(pt_files)
                total_size += path_size
                print(f"  Found {len(pt_files)} .pt files ({path_size / 1024 / 1024 / 1024:.2f} GB)")
                
            except requests.exceptions.RequestException as e:
                print(f"  Error fetching file list for {source_path}: {e}")
                print("  You might need to provide file list manually or check the repository path")
        
        print(f"\nTotal across all paths: {len(all_files)} .pt files")
        print(f"Combined dataset size: {total_size / 1024 / 1024 / 1024:.2f} GB")
        
        # Sort by file size (smallest first) to ensure homogeneous batches during parallel processing
        all_files.sort(key=lambda f: f.size)
        print("Files sorted by size (smallest first) for optimal batch processing")
        
        return all_files, total_size
    
    
    def download_file_with_requests(self, file_info: FileInfo) -> bool:
        """
        Download a single file using requests with chunked streaming.
        """
        # Find which source path this file belongs to
        source_path = None
        for path in self.source_paths:
            if file_info.path.startswith(path):
                source_path = path
                break
        
        if not source_path:
            print(f"  ✗ Could not determine source path for {file_info.name}")
            return False
        
        url = f"https://huggingface.co/datasets/{self.source_repo}/resolve/main/{file_info.path}"
        local_path = self.download_dir / file_info.name
        
        # Skip if already downloaded
        if local_path.exists():
            print(f"  Using existing download: {file_info.name}")
            return True
        
        headers = {}
        
        try:
            # Get file size first
            head_response = requests.head(url, headers=headers, allow_redirects=True)
            file_size = int(head_response.headers.get('content-length', 0))
            
            print(f"  Downloading {file_info.name} from {source_path} ({file_size / 1024 / 1024:.1f} MB)...")
            
            # Download with progress bar
            response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            # Use temporary file to avoid corruption
            temp_path = local_path.with_suffix('.tmp')
            
            with open(temp_path, 'wb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_info.name, leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Move temp file to final location
            temp_path.rename(local_path)
            return True
            
        except Exception as e:
            print(f"  ✗ Error downloading {file_info.name}: {e}")
            # Clean up temp file if exists
            temp_path = local_path.with_suffix('.tmp')
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def convert_pt_to_npy(self, pt_filename: str, source_path_index: int = None) -> Optional[str]:
        """
        Convert a .pt file to .npy format.
        Returns the npy filename if successful, None otherwise.
        """
        pt_path = self.download_dir / pt_filename
        
        # Determine the npy filename using rename function if provided
        if source_path_index is not None and self.rename_functions[source_path_index] is not None:
            npy_filename = self.rename_functions[source_path_index](pt_filename)
        else:
            npy_filename = pt_filename.replace('.pt', '.npy')
        
        npy_path = self.converted_dir / npy_filename
        
        try:
            print(f"  Converting {pt_filename} to {npy_filename}...")
            
            # Load PyTorch file
            data = torch.load(pt_path, map_location='cpu')
            
            # Handle different data types and ensure float32 precision
            if isinstance(data, torch.Tensor):
                np_data = data.numpy().astype(np.float32)
            elif isinstance(data, dict):
                # If dict, convert each tensor to numpy with float32 precision
                np_data = {}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        np_data[k] = v.numpy().astype(np.float32)
                    else:
                        np_data[k] = v
            elif isinstance(data, list):
                np_data = []
                for v in data:
                    if isinstance(v, torch.Tensor):
                        np_data.append(v.numpy().astype(np.float32))
                    else:
                        np_data.append(v)
            else:
                np_data = data
            
            # Save as numpy file
            np.save(npy_path, np_data, allow_pickle=True)
            
            # Clean up the .pt file immediately to save space
            if self.cleanup_after_upload:
                pt_path.unlink()
                print(f"    Deleted {pt_filename} to free space")
            
            return npy_filename
            
        except Exception as e:
            print(f"  ✗ Error converting {pt_filename}: {e}")
            # Clean up partial file if exists
            if npy_path.exists():
                npy_path.unlink()
            return None
    
    def upload_single_file(self, npy_filename: str, original_file_info: FileInfo) -> bool:
        """
        Upload a single .npy file to HuggingFace.
        """
        npy_path = self.converted_dir / npy_filename
        
        if not npy_path.exists():
            print(f"  ✗ File not found: {npy_filename}")
            return False
        
        # Find which source/target path pair this file belongs to
        source_path = None
        target_path = None
        for i, path in enumerate(self.source_paths):
            if original_file_info.path.startswith(path):
                source_path = path
                target_path = self.target_paths[i]
                break
        
        if not source_path or not target_path:
            print(f"  ✗ Could not determine target path for {npy_filename}")
            return False
        
        try:
            print(f"  Uploading {npy_filename} to {target_path} ({npy_path.stat().st_size / 1024 / 1024:.1f} MB)...")
            print(f"  Full upload path will be: {target_path}/{npy_filename}")
            
            # Create the target repository if it doesn't exist
            try:
                self.api.create_repo(
                    repo_id=self.target_repo,
                    repo_type="dataset",
                    exist_ok=True
                )
            except:
                pass  # Repo might already exist
            
            # Upload the file
            path_in_repo = f"{target_path}/{npy_filename}"
            
            upload_file(
                path_or_fileobj=str(npy_path),
                path_in_repo=path_in_repo,
                repo_id=self.target_repo,
                repo_type="dataset",
                commit_message=f"Upload {npy_filename} to {target_path}",
            )
            
            print(f"  ✓ Uploaded {npy_filename} to {target_path}")
            
            # Clean up the local file immediately after upload
            if self.cleanup_after_upload:
                npy_path.unlink()
                print(f"    Deleted {npy_filename} to free space")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error uploading {npy_filename}: {e}")
            return False
    
    def process_single_file(self, file_info: FileInfo) -> bool:
        """
        Complete pipeline for a single file: download → convert → upload → cleanup
        """
        print(f"\nProcessing {file_info.name}...")
        
        # Skip if already processed
        with self.progress_lock:
            if file_info.name in self.processed_files:
                print(f"  Already processed in previous run, skipping...")
                return True
        
        # Find which source path this file belongs to
        source_path_index = None
        for i, path in enumerate(self.source_paths):
            if file_info.path.startswith(path):
                source_path_index = i
                print(f"  File belongs to source path {i}: {path}")
                if self.rename_functions[i] is not None:
                    print(f"  Will use rename function: {self.rename_functions[i].__name__}")
                break
        
        # Download
        if not self.download_file_with_requests(file_info):
            return False
        
        # Convert
        npy_filename = self.convert_pt_to_npy(file_info.name, source_path_index)
        if not npy_filename:
            return False
        
        # Upload
        if not self.upload_single_file(npy_filename, file_info):
            return False
        
        # Mark as processed and save progress
        with self.progress_lock:
            self.processed_files.add(file_info.name)
        self.save_progress()
        
        return True
    
    
    def run(self, file_infos: Optional[List[FileInfo]] = None):
        """
        Run the conversion - process files in parallel to minimize disk usage.
        """
        # Get file list
        if file_infos is None:
            file_infos, total_size = self.get_file_info()[:2]
        
        if not file_infos:
            print("No files to process")
            return
        
        print(f"\n{'='*50}")
        print(f"Starting processing with {self.max_workers} parallel workers")
        print(f"Files to process: {len(file_infos)} across {len(self.source_paths)} paths")
        print(f"Each file will be deleted after upload to save space")
        print(f"{'='*50}\n")
        
        # Process all files in parallel using ThreadPoolExecutor
        total_success = 0
        total_failed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_single_file, file_info): file_info 
                             for file_info in file_infos}
            
            # Collect results with progress bar
            with tqdm(total=len(file_infos), desc="Overall Progress") as pbar:
                for future in as_completed(future_to_file):
                    file_info = future_to_file[future]
                    try:
                        if future.result():
                            total_success += 1
                        else:
                            total_failed += 1
                    except Exception as e:
                        print(f"Processing failed for {file_info.name}: {e}")
                        total_failed += 1
                    pbar.update(1)
        
        # Report results
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"  Successful: {total_success}/{len(file_infos)}")
        print(f"  Failed: {total_failed}")
        
        # Clean up any remaining files
        self.cleanup_remaining_files()
        
        
    def cleanup_remaining_files(self):
        """Clean up any remaining local files."""
        pt_files = list(self.download_dir.glob("*.pt"))
        npy_files = list(self.converted_dir.glob("*.npy"))
        
        if pt_files or npy_files:
            print(f"\nCleaning up remaining files...")
            for f in pt_files + npy_files:
                f.unlink()
            print(f"  Deleted {len(pt_files)} .pt files and {len(npy_files)} .npy files")
    


def rename_full_model(filename: str) -> str:
    """
    Rename full model files from 'full_model_X_margins_fineweb_train_subset_*.pt' to 'nanogpt_train_X.npy' or 'nanogpt_val_X.npy'
    """
    import re
    # Extract model index from filename like 'full_model_0_margins_fineweb_train_subset_063c453b.pt'
    match = re.search(r'full_model_(\d+)_', filename)
    if match:
        model_idx = match.group(1)
        # Determine if it's train or val based on filename content
        if 'val' in filename.lower():
            data_split = 'val'
        elif 'train' in filename.lower():
            data_split = 'train'
        else:
            assert False, f"Filename must contain 'train' or 'val': {filename}"
        new_name = f"nanogpt_{data_split}_{model_idx}.npy"
        print(f"  Renaming: {filename} -> {new_name}")
        return new_name
    else:
        # Fallback: just replace extension
        new_name = filename.replace('.pt', '.npy')
        print(f"  Warning: Could not extract model index from {filename}, using {new_name}")
        return new_name

def rename_oracle_loss_1pct(filename: str) -> str:
    """
    Rename oracle loss 1% files from 'oracle_loss_1pct_X_margins_fineweb_train_subset_*.pt' to 'nanogpt_train_X.npy' or 'nanogpt_val_X.npy'
    """
    import re
    # Extract model index from filename like 'oracle_loss_1pct_0_margins_fineweb_train_subset_063c453b.pt'
    match = re.search(r'oracle_loss_1pct_(\d+)_', filename)
    if match:
        model_idx = match.group(1)
        # Determine if it's train or val based on filename content
        if 'val' in filename.lower():
            data_split = 'val'
        elif 'train' in filename.lower():
            data_split = 'train'
        else:
            assert False, f"Filename must contain 'train' or 'val': {filename}"
        new_name = f"nanogpt_{data_split}_{model_idx}.npy"
        print(f"  Renaming: {filename} -> {new_name}")
        return new_name
    else:
        # Fallback: just replace extension
        new_name = filename.replace('.pt', '.npy')
        print(f"  Warning: Could not extract model index from {filename}, using {new_name}")
        return new_name

def rename_oracle_loss_5pct(filename: str) -> str:
    """
    Rename oracle loss 5% files from 'oracle_loss_5pct_X_margins_fineweb_train_subset_*.pt' to 'nanogpt_train_X.npy' or 'nanogpt_val_X.npy'
    """
    import re
    # Extract model index from filename like 'oracle_loss_5pct_0_margins_fineweb_train_subset_063c453b.pt'
    match = re.search(r'oracle_loss_5pct_(\d+)_', filename)
    if match:
        model_idx = match.group(1)
        # Determine if it's train or val based on filename content
        if 'val' in filename.lower():
            data_split = 'val'
        elif 'train' in filename.lower():
            data_split = 'train'
        else:
            assert False, f"Filename must contain 'train' or 'val': {filename}"
        new_name = f"nanogpt_{data_split}_{model_idx}.npy"
        print(f"  Renaming: {filename} -> {new_name}")
        return new_name
    else:
        # Fallback: just replace extension
        new_name = filename.replace('.pt', '.npy')
        print(f"  Warning: Could not extract model index from {filename}, using {new_name}")
        return new_name

def rename_oracle_random_1pct(filename: str) -> str:
    """
    Rename oracle random 1% files from 'oracle_random_1pct_X_margins_fineweb_train_subset_*.pt' to 'nanogpt_train_X.npy' or 'nanogpt_val_X.npy'
    """
    import re
    # Extract model index from filename like 'oracle_random_1pct_0_margins_fineweb_train_subset_063c453b.pt'
    match = re.search(r'oracle_random_1pct_(\d+)_', filename)
    if match:
        model_idx = match.group(1)
        # Determine if it's train or val based on filename content
        if 'val' in filename.lower():
            data_split = 'val'
        elif 'train' in filename.lower():
            data_split = 'train'
        else:
            assert False, f"Filename must contain 'train' or 'val': {filename}"
        new_name = f"nanogpt_{data_split}_{model_idx}.npy"
        print(f"  Renaming: {filename} -> {new_name}")
        return new_name
    else:
        # Fallback: just replace extension
        new_name = filename.replace('.pt', '.npy')
        print(f"  Warning: Could not extract model index from {filename}, using {new_name}")
        return new_name


def main():
    """
    Main function with example usage.
    """
    # Configuration
    config = {
        'source_repo': 'royrin/KLOM-models',
        'source_paths': [
            'margins/full_models',
            'margins/oracle_loss_5_pct',
            'margins/oracle_loss_1_pct',
            'margins/oracle_random_1_pct'
        ],
        'target_repo': 'puigde/data-unlearning',
        'target_paths': [
            'fineweb/margins/pretrain',
            'fineweb/margins/loss_5_pct',
            'fineweb/margins/loss_1_pct',
            'fineweb/margins/random_1_pct'
        ],
        'rename_functions': [
            rename_full_model,           # For margins/full_models
            rename_oracle_loss_5pct,     # For margins/oracle_loss_5_pct
            rename_oracle_loss_1pct,     # For margins/oracle_loss_1_pct
            rename_oracle_random_1pct    # For margins/oracle_random_1_pct
        ],
        'local_dir': './hf_conversion_temp',
        'cleanup_after_upload': True,  # DELETE files after upload to save space
        'max_workers': 16  # Process up to 16 files in parallel
    }
    
    # Initialize converter
    converter = HuggingFaceConverter(**config)
    
    # Run the conversion
    converter.run()


if __name__ == "__main__":
    main()

