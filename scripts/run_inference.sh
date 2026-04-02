#!/bin/bash

# Default values
MODEL_FILE=""
IMAGE_FILE=""
BACKBONE=""

# Display usage
usage() {
    echo "Usage: $0 -i|--image <path_to_image> [-m|--model <path_to_pt_file>] [-b|--backbone <backbone_name>]"
    echo "Example: $0 -i test_image.jpg -m models/best_vit.pt -b vit"
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_FILE="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_FILE="$2"
            shift 2
            ;;
        -b|--backbone)
            BACKBONE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if image file was provided
if [ -z "$IMAGE_FILE" ]; then
    echo "Error: Image file is required."
    usage
fi

# Ensure PYTHONPATH includes the current directory so imports work correctly
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Build the command
CMD="python -m src.inferencing.predict \"$IMAGE_FILE\""

if [ -n "$MODEL_FILE" ]; then
    CMD="$CMD --model \"$MODEL_FILE\""
fi

if [ -n "$BACKBONE" ]; then
    CMD="$CMD --backbone \"$BACKBONE\""
fi

echo "Executing: $CMD"
eval "$CMD"
