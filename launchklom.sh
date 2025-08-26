#!/usr/bin/env bash
# Usage: ./launchklom.sh --unlearned-dir <path> --oracle-dir <path> --output-dir <path> [options]
#   --unlearned-dir <path>    Path to directory containing unlearned margin files
#   --oracle-dir <path>       Path to directory containing oracle margin files 
#   --output-dir <path>       Path to directory to save KL divergence results
#   --forget-indices <path>   Path to JSON file with batch indices for forget set
#   --retain-indices <path>   Path to JSON file with batch indices for retain set
#   --train-data <path>       (optional) Path to training data file (default: data/fineweb10B/fineweb_train_subset.bin)
#   --val-data <path>         (optional) Path to validation data file (default: data/fineweb10B/fineweb_val_000000.bin)
#   --clip-min <value>        (optional) Minimum value for clipping margins (default: -100)
#   --clip-max <value>        (optional) Maximum value for clipping margins (default: 100)
#   --help                    Show this help message
# for example: ./launchklom.sh --unlearned-dir nanogpt_margins/full_model_margins/ --oracle-dir nanogpt_margins/oracle_margins/ --output-dir nanogpt_klom/ --forget-indices data/ngpt-set-indices/1pct-forget-loss-indices.json --retain-indices data/ngpt-set-indices/1pct-retain-loss-indices.json

set -euo pipefail

# Default values
UNLEARNED_DIR=""
ORACLE_DIR=""
OUTPUT_DIR=""
FORGET_INDICES=""
RETAIN_INDICES=""
TRAIN_DATA="data/fineweb10B/fineweb_train_subset.bin"
VAL_DATA="data/fineweb10B/fineweb_val_000000.bin"
CLIP_MIN=-100
CLIP_MAX=100
LEGACY_MODE=false

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 --unlearned-dir <path> --oracle-dir <path> --output-dir <path> [options]

Required arguments:
  --unlearned-dir <path>    Path to directory containing unlearned margin files
  --oracle-dir <path>       Path to directory containing oracle margin files
  --output-dir <path>       Path to directory to save KL divergence results
  --forget-indices <path>   Path to JSON file with batch indices for forget set
  --retain-indices <path>   Path to JSON file with batch indices for retain set

Optional arguments:
  --train-data <path>       Path to training data file (default: data/fineweb10B/fineweb_train_subset.bin)
  --val-data <path>         Path to validation data file (default: data/fineweb10B/fineweb_val_000000.bin)
  --clip-min <value>        Minimum value for clipping margins (default: -100)
  --clip-max <value>        Maximum value for clipping margins (default: 100)
  --legacy                  Use the legacy teacher_klom.py script (old_teacher_klom.py)
  --help                    Show this help message

Examples:
  # Basic KL divergence computation with forget/retain sets
  $0 --unlearned-dir margins/unlearned/ --oracle-dir margins/oracle/ --output-dir results/ \\
     --forget-indices data/forget-indices.json --retain-indices data/retain-indices.json

  # With custom data files and clipping
  $0 --unlearned-dir margins/unlearned/ --oracle-dir margins/oracle/ --output-dir results/ \\
     --forget-indices data/forget-indices.json --retain-indices data/retain-indices.json \\
     --train-data data/custom_train.bin --val-data data/custom_val.bin \\
     --clip-min -50 --clip-max 50

Output files:
  - klom_forget.pt: KL divergence scores for forget set
  - klom_retain.pt: KL divergence scores for retain set  
  - klom_val.pt: KL divergence scores for validation set
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unlearned-dir)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                UNLEARNED_DIR="$1"
                shift
            else
                echo "Error: --unlearned-dir requires a directory path"
                exit 1
            fi
            ;;
        --oracle-dir)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                ORACLE_DIR="$1"
                shift
            else
                echo "Error: --oracle-dir requires a directory path"
                exit 1
            fi
            ;;
        --output-dir)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                OUTPUT_DIR="$1"
                shift
            else
                echo "Error: --output-dir requires a directory path"
                exit 1
            fi
            ;;
        --forget-indices)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                FORGET_INDICES="$1"
                shift
            else
                echo "Error: --forget-indices requires a file path"
                exit 1
            fi
            ;;
        --retain-indices)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                RETAIN_INDICES="$1"
                shift
            else
                echo "Error: --retain-indices requires a file path"
                exit 1
            fi
            ;;
        --train-data)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                TRAIN_DATA="$1"
                shift
            else
                echo "Error: --train-data requires a file path"
                exit 1
            fi
            ;;
        --val-data)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                VAL_DATA="$1"
                shift
            else
                echo "Error: --val-data requires a file path"
                exit 1
            fi
            ;;
        --clip-min)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                CLIP_MIN="$1"
                shift
            else
                echo "Error: --clip-min requires a numeric value"
                exit 1
            fi
            ;;
        --clip-max)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                CLIP_MAX="$1"
                shift
            else
                echo "Error: --clip-max requires a numeric value"
                exit 1
            fi
            ;;
        --legacy)
            LEGACY_MODE=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$UNLEARNED_DIR" ]; then
    echo "Error: --unlearned-dir argument is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$ORACLE_DIR" ]; then
    echo "Error: --oracle-dir argument is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output-dir argument is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$FORGET_INDICES" ]; then
    echo "Error: --forget-indices argument is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$RETAIN_INDICES" ]; then
    echo "Error: --retain-indices argument is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate that directories exist
echo "Validating input directories and files..."

if [ ! -d "$UNLEARNED_DIR" ]; then
    echo "Error: Unlearned margins directory not found: $UNLEARNED_DIR"
    exit 1
fi

if [ ! -d "$ORACLE_DIR" ]; then
    echo "Error: Oracle margins directory not found: $ORACLE_DIR"
    exit 1
fi

if [ ! -f "$FORGET_INDICES" ]; then
    echo "Error: Forget indices file not found: $FORGET_INDICES"
    exit 1
fi

if [ ! -f "$RETAIN_INDICES" ]; then
    echo "Error: Retain indices file not found: $RETAIN_INDICES"
    exit 1
fi

# Note: train-data and val-data files are validated by teacher_klom.py when needed

# Count files in each directory and validate they match
echo "Checking margin files in directories..."

UNLEARNED_COUNT=$(find "$UNLEARNED_DIR" -type f | wc -l)
ORACLE_COUNT=$(find "$ORACLE_DIR" -type f | wc -l)

echo "  Unlearned margins directory: $UNLEARNED_DIR"
echo "  - Found $UNLEARNED_COUNT margin files"
echo "  Oracle margins directory: $ORACLE_DIR"
echo "  - Found $ORACLE_COUNT margin files"

if [ "$UNLEARNED_COUNT" -eq 0 ]; then
    echo "Error: No margin files found in unlearned directory: $UNLEARNED_DIR"
    exit 1
fi

if [ "$ORACLE_COUNT" -eq 0 ]; then
    echo "Error: No margin files found in oracle directory: $ORACLE_DIR"
    exit 1
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "KL Divergence Computation Configuration"
echo "============================================"
echo "Unlearned margins directory: $UNLEARNED_DIR ($UNLEARNED_COUNT files)"
echo "Oracle margins directory: $ORACLE_DIR ($ORACLE_COUNT files)"
echo "Output directory: $OUTPUT_DIR"
echo "Forget indices file: $FORGET_INDICES"
echo "Retain indices file: $RETAIN_INDICES"
echo "Training data file: $TRAIN_DATA"
echo "Validation data file: $VAL_DATA"
echo "Clip range: [$CLIP_MIN, $CLIP_MAX]"
echo "============================================"

# Function to run teacher_klom.py with common arguments
run_teacher_klom() {
    local output_file="$1"
    local subset_indices="$2"
    local use_subset="$3"
    local description="$4"
    local data_split="$5"
    
    echo "Running KL divergence computation for $description..."
    echo "Output file: $output_file"
    if [ -n "$data_split" ]; then
        echo "Data split: $data_split"
    fi
    
    # Build the command arguments
    CMD_ARGS=()
    CMD_ARGS+=("--unlearned-dir")
    CMD_ARGS+=("$UNLEARNED_DIR")
    CMD_ARGS+=("--oracle-dir")
    CMD_ARGS+=("$ORACLE_DIR")
    CMD_ARGS+=("--output")
    CMD_ARGS+=("$output_file")
    CMD_ARGS+=("--clip-min")
    CMD_ARGS+=("$CLIP_MIN")
    CMD_ARGS+=("--clip-max")
    CMD_ARGS+=("$CLIP_MAX")
    
    if [ -n "$data_split" ]; then
        CMD_ARGS+=("--data-split")
        CMD_ARGS+=("$data_split")
    fi
    
    if [ "$use_subset" = true ]; then
        CMD_ARGS+=("--use-subset")
        CMD_ARGS+=("--subset-indices")
        CMD_ARGS+=("$subset_indices")
    fi
    
        local script_to_run="teacher_klom.py"
    if [ "$LEGACY_MODE" = true ]; then
        script_to_run="old_teacher_klom.py"
    fi
    echo "Command: python $script_to_run ${CMD_ARGS[*]}"
    echo "============================================"
    
    # Execute the command
    python "$script_to_run" "${CMD_ARGS[@]}"
    
    if [ $? -eq 0 ]; then
        echo "✓ $description computation completed successfully!"
        echo "Results saved to: $output_file"
    else
        echo "✗ $description computation failed!"
        exit 1
    fi
    echo "============================================"
}

# Run three KL divergence computations sequentially

echo "Starting KL divergence computations..."
echo "Will perform 3 computations: forget set, retain set, and validation set"
echo "============================================"

# 1. Forget set (training data with forget indices)
echo "COMPUTATION 1/3: FORGET SET"
run_teacher_klom "$OUTPUT_DIR/klom_forget.pt" "$FORGET_INDICES" true "forget set" "train"

# 2. Retain set (training data with retain indices)
echo "COMPUTATION 2/3: RETAIN SET"
run_teacher_klom "$OUTPUT_DIR/klom_retain.pt" "$RETAIN_INDICES" true "retain set" "train"

# 3. Validation set (validation data, no subset)
echo "COMPUTATION 3/3: VALIDATION SET"
run_teacher_klom "$OUTPUT_DIR/klom_val.pt" "" false "validation set" "val"

echo "============================================"
echo "ALL KL DIVERGENCE COMPUTATIONS COMPLETED SUCCESSFULLY!"
echo "============================================"
echo "Results saved to:"
echo "  - Forget set: $OUTPUT_DIR/klom_forget.pt"
echo "  - Retain set: $OUTPUT_DIR/klom_retain.pt"
echo "  - Validation set: $OUTPUT_DIR/klom_val.pt"
echo "============================================"

echo "Launch script completed." 
