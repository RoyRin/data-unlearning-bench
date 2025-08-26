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
from pathlib import Path
from typing import List, Optional, Dict, Tuple
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
                 batch_size: int = 1,
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
            batch_size: Number of files to process before uploading (1 = immediate upload)
        """
        # Handle backward compatibility - if old single path args are provided
        if 'source_path' in kwargs and 'target_path' in kwargs:
            source_paths = [kwargs['source_path']]
            target_paths = [kwargs['target_path']]
        
        if len(source_paths) != len(target_paths):
            raise ValueError("source_paths and target_paths must have the same length")
        
        self.source_repo = source_repo
        self.source_paths = source_paths
        self.target_repo = target_repo
        self.target_paths = target_paths
        self.local_dir = Path(local_dir)
        self.cleanup_after_upload = cleanup_after_upload
        self.batch_size = batch_size
        
        # Create local directories
        self.download_dir = self.local_dir / "downloads"
        self.converted_dir = self.local_dir / "converted"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.converted_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup HuggingFace API - uses cached login from huggingface-cli login
        self.api = HfApi()
        
        # Track processed files across all paths
        self.processed_files = set()
        self.load_progress()
    
    def save_progress(self):
        """Save progress to resume later if needed."""
        progress_file = self.local_dir / "progress.json"
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
        
        return all_files, total_size
    
    def estimate_dataset_size(self):
        """
        Estimate total size and provide recommendation for processing strategy.
        """
        file_infos, total_size = self.get_file_info()
        
        if not file_infos:
            return
        
        # Estimate converted size (numpy files are often similar or slightly smaller)
        estimated_npy_size = total_size * 0.9  # Conservative estimate
        
        # Get available disk space
        stat = os.statvfs(self.local_dir)
        available_space = stat.f_bavail * stat.f_frsize
        
        print(f"\n{'='*60}")
        print("DATASET SIZE ANALYSIS:")
        print(f"{'='*60}")
        print(f"Number of files: {len(file_infos)}")
        print(f"Total .pt size: {total_size / 1024 / 1024 / 1024:.2f} GB")
        print(f"Estimated .npy size: {estimated_npy_size / 1024 / 1024 / 1024:.2f} GB")
        print(f"Available disk space: {available_space / 1024 / 1024 / 1024:.2f} GB")
        print(f"Space needed for batch processing: {(total_size + estimated_npy_size) / 1024 / 1024 / 1024:.2f} GB")
        print(f"Space needed for streaming (one file): {max(f.size for f in file_infos) * 2 / 1024 / 1024 / 1024:.2f} GB")
        
        # Recommendation
        print(f"\n{'='*60}")
        print("RECOMMENDATION:")
        
        if (total_size + estimated_npy_size) < available_space * 0.8:  # Keep 20% buffer
            print("✓ You have enough space for batch processing (download all, then convert).")
            print("  This would be faster but requires more disk space.")
            recommended_batch = len(file_infos)
        else:
            print("✗ Limited disk space - use streaming mode (process one file at a time).")
            print("  This is slower but requires minimal disk space.")
            recommended_batch = 1
        
        print(f"\nRecommended batch_size: {recommended_batch}")
        print(f"{'='*60}\n")
        
        return file_infos, total_size, recommended_batch
    
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
    
    def convert_pt_to_npy(self, pt_filename: str) -> Optional[str]:
        """
        Convert a .pt file to .npy format.
        Returns the npy filename if successful, None otherwise.
        """
        pt_path = self.download_dir / pt_filename
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
        if file_info.name in self.processed_files:
            print(f"  Already processed in previous run, skipping...")
            return True
        
        # Download
        if not self.download_file_with_requests(file_info):
            return False
        
        # Convert
        npy_filename = self.convert_pt_to_npy(file_info.name)
        if not npy_filename:
            return False
        
        # Upload
        if not self.upload_single_file(npy_filename, file_info):
            return False
        
        # Mark as processed and save progress
        self.processed_files.add(file_info.name)
        self.save_progress()
        
        return True
    
    def process_batch(self, file_infos: List[FileInfo]) -> Tuple[int, int]:
        """
        Process a batch of files.
        Returns: (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        for file_info in file_infos:
            if self.process_single_file(file_info):
                successful += 1
            else:
                failed += 1
        
        return successful, failed
    
    def run_streaming(self, file_infos: Optional[List[FileInfo]] = None):
        """
        Run in streaming mode - process one file at a time to minimize disk usage.
        """
        # Get file list
        if file_infos is None:
            file_infos, total_size = self.get_file_info()[:2]
        
        if not file_infos:
            print("No files to process")
            return
        
        print(f"\n{'='*50}")
        print(f"Starting STREAMING mode processing")
        print(f"Files to process: {len(file_infos)} across {len(self.source_paths)} paths")
        print(f"Each file will be deleted after upload to save space")
        print(f"{'='*50}\n")
        
        # Process files one by one
        total_success = 0
        total_failed = 0
        
        with tqdm(total=len(file_infos), desc="Overall Progress") as pbar:
            for i in range(0, len(file_infos), self.batch_size):
                batch = file_infos[i:i+self.batch_size]
                success, failed = self.process_batch(batch)
                total_success += success
                total_failed += failed
                pbar.update(len(batch))
        
        # Report results
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"  Successful: {total_success}/{len(file_infos)}")
        print(f"  Failed: {total_failed}")
        
        # Clean up any remaining files
        self.cleanup_remaining_files()
        
    def run_batch_mode(self, file_infos: Optional[List[FileInfo]] = None):
        """
        Run in batch mode - download all, convert all, then upload all.
        More efficient but requires more disk space.
        """
        if file_infos is None:
            file_infos, _ = self.get_file_info()[:2]
        
        if not file_infos:
            print("No files to process")
            return
        
        print(f"\n{'='*50}")
        print(f"Starting BATCH mode processing")
        print(f"Files to process: {len(file_infos)} across {len(self.source_paths)} paths")
        print(f"{'='*50}\n")
        
        # Download all files
        print("Phase 1: Downloading all files...")
        downloaded = []
        for file_info in tqdm(file_infos, desc="Downloading"):
            if self.download_file_with_requests(file_info):
                downloaded.append(file_info)
        
        # Convert all files
        print("\nPhase 2: Converting all files...")
        converted = []
        for file_info in tqdm(downloaded, desc="Converting"):
            npy_filename = self.convert_pt_to_npy(file_info.name)
            if npy_filename:
                converted.append((npy_filename, file_info))
        
        # Upload all files
        print("\nPhase 3: Uploading all files...")
        uploaded = 0
        for npy_filename, file_info in tqdm(converted, desc="Uploading"):
            if self.upload_single_file(npy_filename, file_info):
                uploaded += 1
        
        print(f"\n{'='*50}")
        print(f"Batch processing complete!")
        print(f"  Downloaded: {len(downloaded)}/{len(file_infos)}")
        print(f"  Converted: {len(converted)}/{len(downloaded)}")
        print(f"  Uploaded: {uploaded}/{len(converted)}")
        
    def cleanup_remaining_files(self):
        """Clean up any remaining local files."""
        pt_files = list(self.download_dir.glob("*.pt"))
        npy_files = list(self.converted_dir.glob("*.npy"))
        
        if pt_files or npy_files:
            print(f"\nCleaning up remaining files...")
            for f in pt_files + npy_files:
                f.unlink()
            print(f"  Deleted {len(pt_files)} .pt files and {len(npy_files)} .npy files")
    
    def run(self, mode: str = "auto"):
        """
        Main entry point for running the conversion.
        
        Args:
            mode: "streaming" for one-by-one processing (minimal disk usage)
                  "batch" for download-all-then-process (faster but needs more space)
                  "auto" to automatically choose based on available space
        """
        if mode == "auto":
            file_infos, total_size, recommended_batch = self.estimate_dataset_size()
            
            if not file_infos:
                print("Could not fetch file information")
                return
            
            if recommended_batch == 1:
                mode = "streaming"
            else:
                print(f"\nChoose processing mode:")
                print("  1. Streaming (one file at a time, minimal disk usage)")
                print("  2. Batch (download all first, faster but needs more space)")
                choice = input("Enter choice (1 or 2): ").strip()
                mode = "streaming" if choice == "1" else "batch"
        
        if mode == "streaming":
            self.run_streaming()
        else:
            self.run_batch_mode()


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
        'target_repo': 'royrin/KLOM-models',
        'target_paths': [
            'margins/full_models_npy',
            'margins/oracle_loss_5_pct_npy',
            'margins/oracle_loss_1_pct_npy',
            'margins/oracle_random_1_pct_npy'
        ],
        'local_dir': './hf_conversion_temp',
        'cleanup_after_upload': True,  # DELETE files after upload to save space
        'batch_size': 1  # Process 1 file at a time for minimal disk usage
    }
    
    # Initialize converter
    converter = HuggingFaceConverter(**config)
    
    # First, analyze the dataset size
    print("Analyzing dataset...")
    file_infos, total_size, recommended = converter.estimate_dataset_size()
    
    if not file_infos:
        print("Could not fetch dataset information. Please check repo path and permissions.")
        return
    
    # Ask user how to proceed
    print("\nHow would you like to proceed?")
    print("  1. Streaming mode (process one file at a time, minimal disk usage)")
    print("  2. Batch mode (download all, then process - needs more disk space)")
    print("  3. Just show dataset info and exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        converter.run_streaming()
    elif choice == "2":
        converter.run_batch_mode()
    elif choice == "3":
        print("Exiting without processing.")
    else:
        print("Invalid choice. Running in auto mode...")
        converter.run(mode="auto")


if __name__ == "__main__":
    main()

