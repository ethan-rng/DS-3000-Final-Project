#!/bin/bash
# Used to download dataset, since dataset is too large to be pushed onto GitHub.
# Usage: bash scripts/download_script.sh
# Compatible with Linux, macOS, and Windows (Git Bash / WSL / MSYS2)

set -e

DATA_DIR="dataset"
mkdir -p "$DATA_DIR"

DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/shivamardeshna/real-and-fake-images-dataset-for-image-forensics"
DATASET_FILE="$DATA_DIR/dataset.zip"

# Check if dataset already exists
if [ "$(ls -A "$DATA_DIR" 2>/dev/null | grep -v .gitkeep)" ]; then
    echo "Dataset already exists in $DATA_DIR/, skipping download."
    exit 0
fi

# Download
echo "Downloading dataset..."
if command -v curl &>/dev/null; then
    curl -L -o "$DATASET_FILE" "$DATASET_URL"
elif command -v wget &>/dev/null; then
    wget -O "$DATASET_FILE" "$DATASET_URL"
elif command -v powershell.exe &>/dev/null; then
    powershell.exe -Command "Invoke-WebRequest -Uri '$DATASET_URL' -OutFile '$DATASET_FILE'"
else
    echo "Error: No download tool found (curl, wget, or powershell)."
    exit 1
fi
echo "Dataset downloaded to $DATASET_FILE"

# Unzip
echo "Unzipping dataset..."
if command -v unzip &>/dev/null; then
    unzip -o "$DATASET_FILE" -d "$DATA_DIR"
elif command -v powershell.exe &>/dev/null; then
    powershell.exe -Command "Expand-Archive -Path '$DATASET_FILE' -DestinationPath '$DATA_DIR' -Force"
elif command -v python3 &>/dev/null; then
    python3 -c "import zipfile; zipfile.ZipFile('$DATASET_FILE').extractall('$DATA_DIR')"
elif command -v python &>/dev/null; then
    python -c "import zipfile; zipfile.ZipFile('$DATASET_FILE').extractall('$DATA_DIR')"
else
    echo "Error: No unzip tool found."
    exit 1
fi

rm "$DATASET_FILE"
echo "Dataset extracted to $DATA_DIR/ and zip removed."