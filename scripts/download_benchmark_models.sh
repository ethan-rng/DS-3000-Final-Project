#!/bin/bash

# 1. Get the absolute path of the project root
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

cd "$PROJECT_ROOT"

# 2. Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found. Please create it first."
    exit 1
fi

# 3. Check for HF_TOKEN
# You can set this in your ~/.bashrc or a .env file in the project root
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not found in environment or .env file."
    echo "If these models are gated, the download will fail."
fi

# 4. Run the downloader
echo "--- Starting Model Acquisition ---"
# We use -u for unbuffered output to see the progress bars in real-time
python -u src/benchmark/download_models.py

echo "--- Downloads Finished ---"