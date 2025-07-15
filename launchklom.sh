#!/usr/bin/env bash
# Usage: ./launchklom.sh --unlearned <paths...> --oracle <paths...> --output <path> [options]
#   --unlearned <paths...>    Paths to unlearned margin files (space-separated)
#   --oracle <paths...>       Paths to oracle margin files (space-separated) 
#   --output <path>           Path to save KL divergence results
#   --subset-indices <path>   (optional) Path to JSON file with batch indices for subset extraction
#   --use-subset             (optional) Enable margin subset extraction based on indices
#   --clip-min <value>       (optional) Minimum value for clipping margins (default: -100)
#   --clip-max <value>       (optional) Maximum value for clipping margins (default: 100)
#   --help                   Show this help message

set -euo pipefail

# Default values
UNLEARNED_MARGINS=()
ORACLE_MARGINS=()
OUTPUT_PATH=""
SUBSET_INDICES=""
USE_SUBSET=false
CLIP_MIN=-100
CLIP_MAX=100

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 --unlearned <paths...> --oracle <paths...> --output <path> [options]

Required arguments:
  --unlearned <paths...>    Paths to unlearned margin files (space-separated)
  --oracle <paths...>       Paths to oracle margin files (space-separated)
  --output <path>           Path to save KL divergence results

Optional arguments:
  --subset-indices <path>   Path to JSON file with batch indices for subset extraction
  --use-subset             Enable margin subset extraction based on indices
  --clip-min <value>       Minimum value for clipping margins (default: -100)
  --clip-max <value>       Maximum value for clipping margins (default: 100)
  --help                   Show this help message

Examples:
  # Basic KL divergence computation
  $0 --unlearned unlearn1.pt unlearn2.pt --oracle oracle1.pt oracle2.pt --output results.pt

  # With subset extraction
  $0 --unlearned unlearn*.pt --oracle oracle*.pt --output results.pt \\
     --use-subset --subset-indices data/ngpt-set-indices/1pct-forget-random-indices.json

  # With custom clipping
  $0 --unlearned margins/unlearn*.pt --oracle margins/oracle*.pt --output kl_results.pt \\
     --clip-min -50 --clip-max 50
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unlearned)
            shift
            # Collect all arguments until next flag or end
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                UNLEARNED_MARGINS+=("$1")
                shift
            done
            ;;
        --oracle)
            shift
            # Collect all arguments until next flag or end
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ORACLE_MARGINS+=("$1")
                shift
            done
            ;;
        --output)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                OUTPUT_PATH="$1"
                shift
            else
                echo "Error: --output requires a file path"
                exit 1
            fi
            ;;
        --subset-indices)
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                SUBSET_INDICES="$1"
                shift
            else
                echo "Error: --subset-indices requires a file path"
                exit 1
            fi
            ;;
        --use-subset)
            USE_SUBSET=true
            shift
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
if [ ${#UNLEARNED_MARGINS[@]} -eq 0 ]; then
    echo "Error: --unlearned argument is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ ${#ORACLE_MARGINS[@]} -eq 0 ]; then
    echo "Error: --oracle argument is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "Error: --output argument is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate that files exist
echo "Validating input files..."

for file in "${UNLEARNED_MARGINS[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Unlearned margin file not found: $file"
        exit 1
    fi
done

for file in "${ORACLE_MARGINS[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Oracle margin file not found: $file"
        exit 1
    fi
done

# Validate subset arguments
if [ "$USE_SUBSET" = true ] && [ -z "$SUBSET_INDICES" ]; then
    echo "Error: --use-subset requires --subset-indices to be specified"
    exit 1
fi

if [ -n "$SUBSET_INDICES" ] && [ ! -f "$SUBSET_INDICES" ]; then
    echo "Error: Subset indices file not found: $SUBSET_INDICES"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "KL Divergence Computation Configuration"
echo "============================================"
echo "Unlearned margin files (${#UNLEARNED_MARGINS[@]}):"
for file in "${UNLEARNED_MARGINS[@]}"; do
    echo "  - $file"
done
echo "Oracle margin files (${#ORACLE_MARGINS[@]}):"
for file in "${ORACLE_MARGINS[@]}"; do
    echo "  - $file"
done
echo "Output file: $OUTPUT_PATH"
echo "Clip range: [$CLIP_MIN, $CLIP_MAX]"
if [ "$USE_SUBSET" = true ]; then
    echo "Subset extraction: ENABLED"
    echo "Subset indices file: $SUBSET_INDICES"
else
    echo "Subset extraction: DISABLED"
fi
echo "============================================"

# Build the command
CMD_ARGS=()
CMD_ARGS+=("--unlearned-margins")
CMD_ARGS+=("${UNLEARNED_MARGINS[@]}")
CMD_ARGS+=("--oracle-margins")
CMD_ARGS+=("${ORACLE_MARGINS[@]}")
CMD_ARGS+=("--output")
CMD_ARGS+=("$OUTPUT_PATH")
CMD_ARGS+=("--clip-min")
CMD_ARGS+=("$CLIP_MIN")
CMD_ARGS+=("--clip-max")
CMD_ARGS+=("$CLIP_MAX")

if [ "$USE_SUBSET" = true ]; then
    CMD_ARGS+=("--use-subset")
    CMD_ARGS+=("--subset-indices")
    CMD_ARGS+=("$SUBSET_INDICES")
fi

echo "Running KL divergence computation..."
echo "Command: python teacher_klom.py ${CMD_ARGS[*]}"
echo "============================================"

# Execute the command
python teacher_klom.py "${CMD_ARGS[@]}"

if [ $? -eq 0 ]; then
    echo "============================================"
    echo "KL divergence computation completed successfully!"
    echo "Results saved to: $OUTPUT_PATH"
    echo "============================================"
else
    echo "============================================"
    echo "KL divergence computation failed!"
    echo "============================================"
    exit 1
fi

echo "Launch script completed." 