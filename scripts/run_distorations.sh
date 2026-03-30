#!/bin/bash

# 1. Get the absolute path of the directory where THIS script lives
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 2. Derive the Project Root (assuming script is in project_root/scripts/)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# 3. Enter the Project Root
cd "$PROJECT_ROOT"

# 4. Activate your virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $PROJECT_ROOT/.venv"
    exit 1
fi

# 5. Configuration Variables
# Change these to suit your current run
INPUT_DIR="dataset/Data Set 1"
DISTORTION_TYPE=3  # 1=Compression, 2=Moire, 3=Both, 4=None
THREADS=8          # Use more threads since this is CPU intensive
LIMIT=0            # Set to a number (e.g., 100) for a quick test run

echo "--- Starting Preprocessing Pipeline ---"
echo "Project Root: $PROJECT_ROOT"
echo "Input Dir:    $INPUT_DIR"
echo "Type:         $DISTORTION_TYPE"

# 6. Execute the Python Script
# We use -u to get unbuffered output so you can see progress in real-time
if [ "$LIMIT" -gt 0 ]; then
    python -u src/preprocessing/add_distoration.py "$INPUT_DIR" "$DISTORTION_TYPE" --threads "$THREADS" --limit "$LIMIT"
else
    python -u src/preprocessing/add_distoration.py "$INPUT_DIR" "$DISTORTION_TYPE" --threads "$THREADS"
fi

echo "--- Preprocessing Complete ---"