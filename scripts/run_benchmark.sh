#!/bin/bash
#SBATCH --job-name=fiba3x3_infer
#SBATCH --output=fiba3x3_infer_%j.out
#SBATCH --error=fiba3x3_infer_%j.err
#SBATCH --time=02:00:00

# Exit on error
set -e

# Setup environment
if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv .venv
fi
echo "Activating virtual environment and dependencies..."
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt || true
pip install facenet-pytorch --no-deps

# Create a sub-folder inside src for benchmark (if not exists)
mkdir -p src/benchmark

# Create figures directory
mkdir -p figures

# First testing pass (limited images to ensure the pipeline is smooth)
echo "Running test on a limited sample set (10 images per class) to verify functionality..."
PYTHONPATH=. python src/benchmark/evaluate.py --dataset_dir "dataset/cleaned/Data Set 4/validation" --limit 10

# Note: Once the limited pass is verified, we can append:
# echo "Running complete benchmrk..."
# PYTHONPATH=. python src/benchmark/evaluate.py --dataset_dir "dataset/cleaned/Data Set 4/validation" --limit 100 # Or full dataset size

echo "Job Complete."
