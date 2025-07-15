#!/usr/bin/env bash
# Usage: ./launchmargins.sh <model_path> [margins_folder] [--val-only] [--train-only] [--val-data <path>] [--train-data <path>]
#   model_path         Path to the model checkpoint (.pt file)
#   margins_folder     (optional) Directory to save margins files (default: same as model directory)
#   --val-only         (optional) Only compute margins for validation data
#   --train-only       (optional) Only compute margins for training data  
#   --val-data         (optional) Path to validation data file (default: data/fineweb10B/fineweb_val_000000.bin)
#   --train-data       (optional) Path to training data file (default: data/fineweb10B/fineweb_train_subset.bin)

set -euo pipefail

# Check if model path is provided
if [ $# -lt 1 ]; then
    echo "Error: Model path is required"
    echo "Usage: $0 <model_path> [margins_folder] [--val-only] [--train-only] [--val-data <path>] [--train-data <path>]"
    exit 1
fi

MODEL_PATH="$1"
MARGINS_FOLDER=""
VAL_ONLY=false
TRAIN_ONLY=false
VAL_DATA="data/fineweb10B/fineweb_val_000000.bin"
TRAIN_DATA="data/fineweb10B/fineweb_train_subset.bin"

# Parse arguments
i=2
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --val-only)
            VAL_ONLY=true
            ;;
        --train-only)
            TRAIN_ONLY=true
            ;;
        --val-data)
            i=$((i + 1))
            if [ $i -le $# ]; then
                VAL_DATA="${!i}"
            else
                echo "Error: --val-data requires a file path"
                exit 1
            fi
            ;;
        --train-data)
            i=$((i + 1))
            if [ $i -le $# ]; then
                TRAIN_DATA="${!i}"
            else
                echo "Error: --train-data requires a file path"
                exit 1
            fi
            ;;
        *)
            # If it's not a flag and margins_folder is empty, treat as margins_folder
            if [ -z "$MARGINS_FOLDER" ]; then
                MARGINS_FOLDER="$arg"
            else
                echo "Warning: Unknown argument: $arg"
            fi
            ;;
    esac
    i=$((i + 1))
done

# Validate model path exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

# Set default margins folder if not provided
if [ -z "$MARGINS_FOLDER" ]; then
    MODEL_DIR=$(dirname "$MODEL_PATH")
    MARGINS_FOLDER="$MODEL_DIR"
    echo "No margins folder specified, using model directory: $MARGINS_FOLDER"
fi

# Create margins folder if it doesn't exist
mkdir -p "$MARGINS_FOLDER"

# Validate conflicting flags
if [ "$VAL_ONLY" = true ] && [ "$TRAIN_ONLY" = true ]; then
    echo "Error: Cannot specify both --val-only and --train-only"
    exit 1
fi

echo "============================================"
echo "Margin Computation Configuration"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Margins folder: $MARGINS_FOLDER"
echo "Validation data: $VAL_DATA"
echo "Training data: $TRAIN_DATA"
echo "Val only: $VAL_ONLY"
echo "Train only: $TRAIN_ONLY"
echo "============================================"

# Function to run margin computation with error handling
run_margin_computation() {
    local data_type="$1"
    local data_path="$2"
    
    echo "Computing margins for $data_type data..."
    echo "Data file: $data_path"
    
    if [ ! -f "$data_path" ]; then
        echo "Error: Data file not found: $data_path"
        return 1
    fi
    
    # Build margins folder argument
    MARGINS_ARG=""
    if [ -n "$MARGINS_FOLDER" ]; then
        MARGINS_ARG="--margins-folder $MARGINS_FOLDER"
    fi
    
    # Run the margin computation
    python teacher_margins.py "$MODEL_PATH" "$data_path" $MARGINS_ARG
    
    if [ $? -eq 0 ]; then
        echo "$data_type margins computation completed successfully"
        return 0
    else
        echo "$data_type margins computation failed"
        return 1
    fi
}

# Execute margin computations based on flags
if [ "$TRAIN_ONLY" = true ]; then
    echo "============================================"
    echo "Computing margins for training data only..."
    echo "============================================"
    run_margin_computation "Training" "$TRAIN_DATA"
    
elif [ "$VAL_ONLY" = true ]; then
    echo "============================================"
    echo "Computing margins for validation data only..."
    echo "============================================"
    run_margin_computation "Validation" "$VAL_DATA"
    
else
    # Default: run both, validation first
    echo "============================================"
    echo "STEP 1: Computing margins for validation data..."
    echo "============================================"
    
    if run_margin_computation "Validation" "$VAL_DATA"; then
        echo "============================================"
        echo "STEP 2: Computing margins for training data..."
        echo "============================================"
        
        if run_margin_computation "Training" "$TRAIN_DATA"; then
            echo "============================================"
            echo "All margin computations completed successfully!"
            echo "============================================"
        else
            echo "Training margin computation failed"
            exit 1
        fi
    else
        echo "Validation margin computation failed, skipping training"
        exit 1
    fi
fi

echo "Margin computation script completed."
