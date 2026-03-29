#!/bin/bash
#SBATCH --job-name=fiba3x3_infer
#SBATCH --output=fiba3x3_infer_%j.out
#SBATCH --error=fiba3x3_infer_%j.err
#SBATCH --partition=gpu
#SBATCH --time=02:00:00

# Exit on error
set -e

# Setup environment
echo "Setting up virtual environment and dependencies..."
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
pip install tensorflow huggingface-hub requests python-dotenv

# Create a sub-folder inside src for benchmark (if not exists)
mkdir -p src/benchmark

# Create figures directory
mkdir -p figures

# First testing pass (limited images to ensure the pipeline is smooth)
echo "Running test on a limited sample set (10 images per class) to verify functionality..."
python src/benchmark/evaluate.py --limit 10

# Note: Once the limited pass is verified, we can append:
# echo "Running complete benchmrk..."
# python src/benchmark/evaluate.py --limit 100 # Or full dataset size

echo "Job Complete."
