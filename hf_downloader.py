#!/usr/bin/env python3
"""
HuggingFace Dataset Downloader

Downloads files from a HuggingFace dataset repository using parallel workers.
Based on the download functionality from hf_converter.py.
"""

import os
import sys
import argparse
import requests
import threading
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class FileInfo:
    """Store file information"""
    name: str
    size: int
    path: str
    url: str

class HuggingFaceDownloader:
    def __init__(self, 
                 repo_url: str,
                 local_dir: str = "./downloads",
                 max_workers: int = 32):
        """
        Initialize the downloader.
        
        Args:
            repo_url: HuggingFace dataset URL (e.g., "https://huggingface.co/datasets/puigde/data-unlearning/tree/main/fineweb/margins/loss_1_pct")
            local_dir: Local directory for downloads
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.repo_url = repo_url
        self.local_dir = Path(local_dir)
        self.max_workers = max_workers
        
        # Parse the URL to extract repo info
        self.repo_id, self.path = self._parse_repo_url(repo_url)
        
        # Create local directory
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for progress tracking
        self.progress_lock = threading.Lock()
    
    def _parse_repo_url(self, repo_url: str) -> tuple[str, str]:
        """
        Parse HuggingFace URL to extract repository ID and path.
        
        Example:
        https://huggingface.co/datasets/puigde/data-unlearning/tree/main/fineweb/margins/loss_1_pct
        -> ('puigde/data-unlearning', 'fineweb/margins/loss_1_pct')
        """
        # Remove the base URL and extract components
        url_parts = repo_url.replace('https://huggingface.co/datasets/', '').split('/')
        
        if len(url_parts) < 2:
            raise ValueError(f"Invalid HuggingFace URL format: {repo_url}")
        
        # First two parts are the repo ID (user/repo)
        repo_id = f"{url_parts[0]}/{url_parts[1]}"
        
        # Rest is the path (skip 'tree/main' if present)
        path_parts = url_parts[2:]
        if len(path_parts) >= 2 and path_parts[0] == 'tree' and path_parts[1] == 'main':
            path_parts = path_parts[2:]
        
        path = '/'.join(path_parts) if path_parts else ''
        
        return repo_id, path
    
    def get_file_list(self) -> List[FileInfo]:
        """
        Get list of files from the HuggingFace repository path.
        """
        print(f"Fetching file list from {self.repo_id}/{self.path}...")
        
        # Use the tree API to get directory contents with sizes
        api_url = f"https://huggingface.co/api/datasets/{self.repo_id}/tree/main/{self.path}"
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            files = response.json()
            
            file_infos = []
            total_size = 0
            
            for f in files:
                if f['type'] == 'file':  # Only include files, not directories
                    file_info = FileInfo(
                        name=f['path'].split('/')[-1],
                        size=f.get('size', 0),
                        path=f['path'],
                        url=f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{f['path']}"
                    )
                    file_infos.append(file_info)
                    total_size += file_info.size
            
            print(f"Found {len(file_infos)} files ({total_size / 1024 / 1024 / 1024:.2f} GB)")
            
            # Sort by file size (smallest first) for optimal parallel processing
            file_infos.sort(key=lambda f: f.size)
            
            return file_infos
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching file list: {e}")
            return []
    
    def download_single_file(self, file_info: FileInfo) -> bool:
        """
        Download a single file with progress tracking.
        """
        local_path = self.local_dir / file_info.name
        
        # Skip if already downloaded and same size
        if local_path.exists():
            existing_size = local_path.stat().st_size
            if existing_size == file_info.size:
                print(f"  ✓ {file_info.name} already exists (correct size)")
                return True
            else:
                print(f"  ⚠️  {file_info.name} exists but wrong size ({existing_size} vs {file_info.size}), re-downloading...")
        
        try:
            print(f"  Downloading {file_info.name} ({file_info.size / 1024 / 1024:.1f} MB)...")
            
            # Download with progress bar
            response = requests.get(file_info.url, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            # Use temporary file to avoid corruption
            temp_path = local_path.with_suffix('.tmp')
            
            with open(temp_path, 'wb') as f:
                with tqdm(total=file_info.size, unit='B', unit_scale=True, 
                         desc=file_info.name, leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Move temp file to final location
            temp_path.rename(local_path)
            
            # Verify file size
            final_size = local_path.stat().st_size
            if final_size != file_info.size:
                print(f"  ✗ Size mismatch for {file_info.name}: {final_size} vs {file_info.size}")
                local_path.unlink()
                return False
            
            print(f"  ✓ Successfully downloaded {file_info.name}")
            return True
            
        except Exception as e:
            print(f"  ✗ Error downloading {file_info.name}: {e}")
            # Clean up temp file if exists
            temp_path = local_path.with_suffix('.tmp')
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def download_all(self):
        """
        Download all files using parallel workers.
        """
        # Get file list
        file_infos = self.get_file_list()
        
        if not file_infos:
            print("No files to download")
            return
        
        print(f"\n{'='*60}")
        print(f"Starting download with {self.max_workers} parallel workers")
        print(f"Files to download: {len(file_infos)}")
        print(f"Target directory: {self.local_dir.absolute()}")
        print(f"{'='*60}\n")
        
        # Download all files in parallel
        total_success = 0
        total_failed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_file = {executor.submit(self.download_single_file, file_info): file_info 
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
                        print(f"Download failed for {file_info.name}: {e}")
                        total_failed += 1
                    pbar.update(1)
        
        # Report results
        print(f"\n{'='*60}")
        print(f"Download complete!")
        print(f"  Successful: {total_success}/{len(file_infos)}")
        print(f"  Failed: {total_failed}")
        print(f"  Files saved to: {self.local_dir.absolute()}")
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Download files from HuggingFace dataset repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://huggingface.co/datasets/puigde/data-unlearning/tree/main/fineweb/margins/loss_1_pct
  %(prog)s https://huggingface.co/datasets/puigde/data-unlearning/tree/main/fineweb/margins/pretrain --workers 16 --output ./my_downloads
        """
    )
    
    parser.add_argument(
        "repo_url",
        help="HuggingFace dataset URL (e.g., https://huggingface.co/datasets/user/repo/tree/main/path)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./downloads",
        help="Output directory for downloads (default: ./downloads)"
    )
    
    args = parser.parse_args()
    
    # Create downloader and run
    try:
        downloader = HuggingFaceDownloader(
            repo_url=args.repo_url,
            local_dir=args.output,
            max_workers=args.workers
        )
        downloader.download_all()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()