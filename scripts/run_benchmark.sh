#!/bin/bash
#SBATCH --job-name=benchmark_deepfake
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --time=06:00:00

set -e

# ── Environment ────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt -q || true

mkdir -p figures/distortion_benchmark models

# ── Configuration ──────────────────────────────────────────────────────────
CLEANED_DIR="${1:-dataset/cleaned}"  # path containing Data Set 1..4
SPLIT="${2:-test}"                   # test | validation | train
LIMIT="${3:-500}"                    # images per class per variant (0 = no cap)
API_BUDGET="${4:-3000}"              # max API calls per EdenAI model per variant
OUT="models/benchmark_distortion_results.json"

echo ""
echo "=== Smoke test: 10 images/class, 3 API calls/model ==="
PYTHONPATH=. python src/benchmark/evaluate.py \
    --cleaned_dir "$CLEANED_DIR" \
    --split       "$SPLIT" \
    --limit       10 \
    --api_budget  3 \
    --out         "models/benchmark_smoke.json" \
    --figures_dir "figures/smoke"

echo ""
echo "=== Full distortion benchmark: limit=$LIMIT, api_budget=$API_BUDGET ==="
PYTHONPATH=. python src/benchmark/evaluate.py \
    --cleaned_dir "$CLEANED_DIR" \
    --split       "$SPLIT" \
    --limit       "$LIMIT" \
    --api_budget  "$API_BUDGET" \
    --out         "$OUT" \
    --figures_dir "figures/distortion_benchmark" \
    --no_download

echo ""
echo "Benchmark complete."
echo "  Results : $OUT"
echo "  Figures : figures/distortion_benchmark/"
