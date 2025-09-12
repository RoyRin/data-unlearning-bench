#!/bin/bash

# HuggingFace Dataset Download Script
# Downloads the data-unlearning datasets to appropriate directories

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOADER_SCRIPT="$SCRIPT_DIR/hf_downloader.py"
WORKERS=32

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

# Function to download dataset
download_dataset() {
    local url=$1
    local target_dir=$2
    local dataset_name=$3
    
    echo -e "${YELLOW}Starting download: $dataset_name${NC}"
    echo -e "${BLUE}URL: $url${NC}"
    echo -e "${BLUE}Target: $target_dir${NC}"
    echo -e "${BLUE}Workers: $WORKERS${NC}"
    echo ""
    
    if python "$DOWNLOADER_SCRIPT" "$url" --workers "$WORKERS" --output "$target_dir"; then
        echo -e "${GREEN}✓ Successfully downloaded: $dataset_name${NC}"
        echo ""
    else
        echo -e "${RED}✗ Failed to download: $dataset_name${NC}"
        return 1
    fi
}

# Download loss_1_pct dataset
echo -e "${YELLOW}===========================================${NC}"
echo -e "${YELLOW}Downloading Loss 1% Dataset${NC}"
echo -e "${YELLOW}===========================================${NC}"
download_dataset "$LOSS_1PCT_URL" "$LOSS_1PCT_DIR" "Loss 1% Dataset"

# Download pretrain dataset
echo -e "${YELLOW}===========================================${NC}"
echo -e "${YELLOW}Downloading Pretrain Dataset${NC}"
echo -e "${YELLOW}===========================================${NC}"
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