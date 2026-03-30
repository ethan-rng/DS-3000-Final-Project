#!/bin/bash
# Used to download dataset, since dataset is too large to be pushed onto GitHub.
# Usage: bash scripts/download_script.sh
# Compatible with Linux, macOS, and Windows (Git Bash / WSL / MSYS2)

set -e

DATA_DIR="dataset"
RAW_DIR="$DATA_DIR/raw"
TMP_DIR="$DATA_DIR/tmp"
mkdir -p "$DATA_DIR" "$RAW_DIR" "$TMP_DIR"

DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/shivamardeshna/real-and-fake-images-dataset-for-image-forensics"
DATASET_FILE="$DATA_DIR/dataset.zip"

# Check if dataset already exists
if [ "$(ls -A "$RAW_DIR" 2>/dev/null | grep -E 'Data Set')" ]; then
    echo "Dataset already exists in $RAW_DIR/, skipping download."
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
echo "Unzipping dataset to temporary directory..."
if command -v unzip &>/dev/null; then
    unzip -q -o "$DATASET_FILE" -d "$TMP_DIR"
elif command -v powershell.exe &>/dev/null; then
    powershell.exe -Command "Expand-Archive -Path '$DATASET_FILE' -DestinationPath '$TMP_DIR' -Force"
elif command -v python3 &>/dev/null; then
    python3 -c "import zipfile; zipfile.ZipFile('$DATASET_FILE').extractall('$TMP_DIR')"
elif command -v python &>/dev/null; then
    python -c "import zipfile; zipfile.ZipFile('$DATASET_FILE').extractall('$TMP_DIR')"
else
    echo "Error: No unzip tool found."
    exit 1
fi

rm "$DATASET_FILE"

echo "Organizing dataset into raw directory..."
# Move and flatten folders so that we don't have "Data Set 1" inside "Data Set 1"
for ds_path in "$TMP_DIR"/Data\ Set*; do
    if [ -d "$ds_path" ]; then
        ds_name=$(basename "$ds_path")
        mkdir -p "$RAW_DIR/$ds_name"
        
        # Check if there is a nested folder with the same name
        if [ -d "$ds_path/$ds_name" ]; then
            mv "$ds_path/$ds_name"/* "$RAW_DIR/$ds_name/"
        else
            mv "$ds_path"/* "$RAW_DIR/$ds_name/"
        fi
    fi
done

# Clean up tmp directory
rm -rf "$TMP_DIR"
echo "Dataset extracted and organized in $RAW_DIR/"