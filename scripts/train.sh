#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -d PATH   Path to top-level dataset/ directory (default: dataset/cleaned)
  -b NAME   Backbone name (efficientnet_b0, mobilenet_v3_large, resnet50). Default: efficientnet_b0
  -e N      Number of epochs (default: 10)
  -B N      Batch size (default: 32)
  -l FLOAT  Learning rate (default: 1e-4)
  -o DIR    Output directory for checkpoints (default: models)
  -m N      Max samples per class (0 = all, default: 0)
  -h        Show this help

Example:
  $(basename "$0") -d dataset/cleaned -b efficientnet_b0 -e 10 -B 32 -l 1e-4 -o models
  $(basename "$0") -d dataset/cleaned -m 1000 -e 2   # quick test with 1k samples/class
EOF
}

# defaults
DATASET_ROOT="dataset/cleaned"
BACKBONE="efficientnet_b0"
EPOCHS=10
BATCH_SIZE=32
LR=1e-4
OUT_DIR="models"
MAX_SAMPLES=0

while getopts ":d:b:e:B:l:o:m:h" opt; do
  case ${opt} in
    d) DATASET_ROOT=${OPTARG} ;;
    b) BACKBONE=${OPTARG} ;;
    e) EPOCHS=${OPTARG} ;;
    B) BATCH_SIZE=${OPTARG} ;;
    l) LR=${OPTARG} ;;
    o) OUT_DIR=${OPTARG} ;;
    m) MAX_SAMPLES=${OPTARG} ;;
    h) show_help; exit 0 ;;
    *) show_help; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Starting training"
echo "  dataset_root: $DATASET_ROOT"
echo "  backbone:     $BACKBONE"
echo "  epochs:       $EPOCHS"
echo "  batch_size:   $BATCH_SIZE"
echo "  lr:           $LR"
echo "  out_dir:      $OUT_DIR"
echo "  max_samples:  $MAX_SAMPLES"

# Run training (ensure you have activated your Python environment)
python -m src.train \
  --dataset_root "$DATASET_ROOT" \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --out_dir "$OUT_DIR" \
  --max_samples "$MAX_SAMPLES"

echo "Training finished. Checkpoints saved to $OUT_DIR"
