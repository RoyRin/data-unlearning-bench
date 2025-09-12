#!/bin/bash

# HuggingFace Dataset Download Script
# Downloads the data-unlearning datasets to appropriate directories
# Features: Smart file checking, batch processing, skip existing files

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOADER_SCRIPT="$SCRIPT_DIR/hf_downloader.py"
WORKERS=32
BATCH_SIZE=10  # Number of files to process in each batch

# Dataset URLs and target directories
LOSS_1PCT_URL="https://huggingface.co/datasets/puigde/data-unlearning/tree/main/fineweb/margins/loss_1_pct"
LOSS_1PCT_DIR="./fineweb/margins/loss_1_pct"

PRETRAIN_URL="https://huggingface.co/datasets/puigde/data-unlearning/tree/main/fineweb/margins/pretrain"
PRETRAIN_DIR="./fineweb/margins/pretrain"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HuggingFace Dataset Download Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if downloader script exists
if [ ! -f "$DOWNLOADER_SCRIPT" ]; then
    echo -e "${RED}Error: Downloader script not found at $DOWNLOADER_SCRIPT${NC}"
    exit 1
fi

# Function to get remote file list from HuggingFace
get_remote_file_list() {
    local url=$1
    local temp_file="/tmp/hf_files_$(basename "$url").txt"
    
    echo -e "${CYAN}Getting remote file list...${NC}"
    
    # Extract repo info from URL
    local repo_path=$(echo "$url" | sed 's|https://huggingface.co/datasets/||' | sed 's|/tree/main|/resolve/main|')
    local api_url="https://huggingface.co/api/datasets/$(echo "$repo_path" | cut -d'/' -f1-2)/tree/main/$(echo "$repo_path" | cut -d'/' -f3-)"
    
    # Get file list using curl and extract file names and sizes
    curl -s "$api_url" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for item in data:
        if item.get('type') == 'file':
            print(f\"{item['path'].split('/')[-1]}\t{item.get('size', 0)}\")
except:
    pass
" > "$temp_file"
    
    if [ -s "$temp_file" ]; then
        echo -e "${GREEN}Found $(wc -l < "$temp_file") remote files${NC}"
        echo "$temp_file"
    else
        echo -e "${RED}Failed to get remote file list${NC}"
        return 1
    fi
}

# Function to get local file list with sizes
get_local_file_list() {
    local target_dir=$1
    local temp_file="/tmp/local_files_$(basename "$target_dir").txt"
    
    if [ -d "$target_dir" ]; then
        echo -e "${CYAN}Scanning local files in $target_dir...${NC}"
        find "$target_dir" -type f -printf "%f\t%s\n" | sort > "$temp_file"
        echo -e "${GREEN}Found $(wc -l < "$temp_file") local files${NC}"
    else
        echo -e "${YELLOW}Local directory doesn't exist yet${NC}"
        touch "$temp_file"
    fi
    
    echo "$temp_file"
}

# Function to compare and get missing files
get_missing_files() {
    local remote_list=$1
    local local_list=$2
    local missing_file="/tmp/missing_files_$(date +%s).txt"
    
    echo -e "${CYAN}Comparing file lists...${NC}"
    
    # Compare files by name and size
    python3 -c "
import sys

# Read remote files
remote = {}
try:
    with open('$remote_list', 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    remote[parts[0]] = int(parts[1])
except:
    pass

# Read local files  
local = {}
try:
    with open('$local_list', 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    local[parts[0]] = int(parts[1])
except:
    pass

# Find missing or different sized files
missing = []
for filename, size in remote.items():
    if filename not in local or local[filename] != size:
        missing.append((filename, size))

# Write missing files
with open('$missing_file', 'w') as f:
    for filename, size in missing:
        f.write(f'{filename}\t{size}\n')

print(f'Missing/different files: {len(missing)}')
print(f'Already downloaded: {len([f for f in remote if f in local and local[f] == remote[f]])}')
"
    
    echo "$missing_file"
}

# Function to download dataset with smart checking
download_dataset() {
    local url=$1
    local target_dir=$2
    local dataset_name=$3
    
    echo -e "${YELLOW}===========================================${NC}"
    echo -e "${YELLOW}Processing: $dataset_name${NC}"
    echo -e "${YELLOW}===========================================${NC}"
    echo -e "${BLUE}URL: $url${NC}"
    echo -e "${BLUE}Target: $target_dir${NC}"
    echo -e "${BLUE}Workers: $WORKERS${NC}"
    echo ""
    
    # Create target directory
    mkdir -p "$target_dir"
    
    # Get file lists
    local remote_list=$(get_remote_file_list "$url")
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to get remote file list for $dataset_name${NC}"
        return 1
    fi
    
    local local_list=$(get_local_file_list "$target_dir")
    local missing_list=$(get_missing_files "$remote_list" "$local_list")
    
    # Check if we need to download anything
    local missing_count=$(wc -l < "$missing_list")
    if [ "$missing_count" -eq 0 ]; then
        echo -e "${GREEN}✓ All files already downloaded for $dataset_name${NC}"
        echo ""
        # Clean up temp files
        rm -f "$remote_list" "$local_list" "$missing_list"
        return 0
    fi
    
    echo -e "${YELLOW}Need to download $missing_count files${NC}"
    echo ""
    
    # Download missing files (the downloader script already handles individual file checking)
    if python "$DOWNLOADER_SCRIPT" "$url" --workers "$WORKERS" --output "$target_dir"; then
        echo -e "${GREEN}✓ Successfully processed: $dataset_name${NC}"
        echo ""
    else
        echo -e "${RED}✗ Failed to download: $dataset_name${NC}"
        # Clean up temp files
        rm -f "$remote_list" "$local_list" "$missing_list"
        return 1
    fi
    
    # Clean up temp files
    rm -f "$remote_list" "$local_list" "$missing_list"
}

# Download datasets
download_dataset "$LOSS_1PCT_URL" "$LOSS_1PCT_DIR" "Loss 1% Dataset"
download_dataset "$PRETRAIN_URL" "$PRETRAIN_DIR" "Pretrain Dataset"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All downloads completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Downloaded datasets:${NC}"
echo -e "  • Loss 1%: $LOSS_1PCT_DIR"
echo -e "  • Pretrain: $PRETRAIN_DIR"
echo ""
echo -e "${BLUE}Total space used:${NC}"
du -sh "$LOSS_1PCT_DIR" "$PRETRAIN_DIR" 2>/dev/null | awk '{print "  • " $2 ": " $1}' || echo "  (Unable to calculate - directories may not exist yet)"