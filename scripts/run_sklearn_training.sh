#!/usr/bin/env bash
#SBATCH --job-name=train_sklearn
#SBATCH --output=sklearn_%j.out
#SBATCH --error=sklearn_%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

show_help() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -d PATH   Path to top-level dataset/ directory (default: dataset)
  -t NAME   Model type: logistic_regression, svm, random_forest, knn (default: random_forest)
  -o DIR    Output directory for model .pkl and metrics (default: models)
  -m N      Max samples per class (0 = all, default: 0)
  -h        Show this help

Examples:
  $(basename "$0") -t random_forest
  $(basename "$0") -t svm -m 2000
  $(basename "$0") -t logistic_regression -d dataset -o models/lr
  $(basename "$0") -t knn -m 5000
EOF
}

# defaults
DATASET_ROOT="dataset"
MODEL_TYPE="random_forest"
OUT_DIR="models"
MAX_SAMPLES=0

while getopts ":d:t:o:m:h" opt; do
  case ${opt} in
    d) DATASET_ROOT=${OPTARG} ;;
    t) MODEL_TYPE=${OPTARG} ;;
    o) OUT_DIR=${OPTARG} ;;
    m) MAX_SAMPLES=${OPTARG} ;;
    h) show_help; exit 0 ;;
    *) show_help; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# On HPC nodes SLURM may change the working directory; fall back to known path
if [ -d "/scratch/ethanrng/final-project" ]; then
    REPO_ROOT="/scratch/ethanrng/final-project"
elif [ -d "/home/ethanrng/scratch/final-project" ]; then
    REPO_ROOT="/home/ethanrng/scratch/final-project"
fi
cd "$REPO_ROOT"

# Auto-detect dataset root: download_dataset.sh places data under dataset/raw/,
# but the local workspace may have it directly under dataset/.
if [ "$DATASET_ROOT" = "dataset" ] && [ -d "$DATASET_ROOT/raw" ] && [ ! -d "$DATASET_ROOT/Data Set 1" ]; then
    DATASET_ROOT="$DATASET_ROOT/raw"
    echo "Auto-detected dataset root: $DATASET_ROOT"
fi

echo "=========================================="
echo " sklearn Training Job"
echo "=========================================="
echo "  model_type:   $MODEL_TYPE"
echo "  dataset_root: $DATASET_ROOT"
echo "  out_dir:      $OUT_DIR"
echo "  max_samples:  $MAX_SAMPLES"
echo "=========================================="

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found at .venv"
    exit 1
fi

python -m src.training.train_sklearn \
  --dataset_root "$DATASET_ROOT" \
  --model_type   "$MODEL_TYPE" \
  --out_dir      "$OUT_DIR" \
  --max_samples  "$MAX_SAMPLES"

echo "Training finished. Model and metrics saved to $OUT_DIR"
