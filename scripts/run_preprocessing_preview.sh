#!/bin/bash
# ============================================================
# run_preprocessing_preview.sh
#
# Runs the face-detection / alignment preprocessing step on a
# small sample of images and prints before-vs-after stats so
# you can verify the pipeline is working correctly.
#
# Usage:
#   bash scripts/run_preprocessing_preview.sh [--limit N] [--src PATH]
#
# Flags:
#   --limit N     Number of images to sample (default: 10)
#   --src   PATH  Source dataset root (default: dataset/raw)
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 2. Activate virtual environment
# ---------------------------------------------------------------------------
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found at $PROJECT_ROOT/.venv"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Parse optional flags
# ---------------------------------------------------------------------------
LIMIT=10
SRC_DIR="dataset/raw"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)  LIMIT="$2";   shift 2 ;;
        --src)    SRC_DIR="$2"; shift 2 ;;
        *)        echo "Unknown flag: $1"; exit 1 ;;
    esac
done

DST_DIR="dataset/tmp/preprocessing_preview"

echo ""
echo "============================================================"
echo "  Preprocessing Preview"
echo "============================================================"
echo "  Source  : $SRC_DIR"
echo "  Output  : $DST_DIR"
echo "  Sample  : $LIMIT images"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 4. Inline Python helper — before/after stats
# ---------------------------------------------------------------------------
python - <<PYTHON
import sys, os, random, shutil
from pathlib import Path
from PIL import Image

src   = Path("$SRC_DIR")
dst   = Path("$DST_DIR")
limit = int("$LIMIT")

# ── collect candidate images ─────────────────────────────────────────────
exts  = {".jpg", ".jpeg", ".png", ".bmp"}
all_images = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts]

if not all_images:
    print(f"[ERROR] No images found under '{src}'. Check --src path.")
    sys.exit(1)

random.seed(42)
sample = random.sample(all_images, min(limit, len(all_images)))

# ── BEFORE stats ─────────────────────────────────────────────────────────
print("─" * 60)
print(f"{'BEFORE preprocessing':^60}")
print("─" * 60)
print(f"  {'File':<38} {'Size (WxH)':>12}  {'Mode':>5}")
print("  " + "-" * 56)

total_before_bytes = 0
for p in sample:
    try:
        img = Image.open(p)
        size_str = f"{img.width}x{img.height}"
        mode_str = img.mode
    except Exception as e:
        size_str, mode_str = "ERROR", str(e)[:10]
    kb = p.stat().st_size / 1024
    total_before_bytes += p.stat().st_size
    print(f"  {p.name:<38} {size_str:>12}  {mode_str:>5}  ({kb:.1f} KB)")

print()
print(f"  Total images selected : {len(sample)}")
print(f"  Total size (before)   : {total_before_bytes/1024:.1f} KB")

# ── write sample list to disk so the preprocessing step can reuse it ─────
dst.mkdir(parents=True, exist_ok=True)
sample_list_file = dst / "_sample_paths.txt"
sample_list_file.write_text("\n".join(str(p) for p in sample))
print(f"\n  Sample list written to: {sample_list_file}")
PYTHON

echo ""
echo "Running face detection + alignment on the sample …"
echo ""

# ---------------------------------------------------------------------------
# 5. Run the preprocessing __main__ entrypoint
#    (src/training/preprocessing.py --src <raw> --dst <preview_out>)
# ---------------------------------------------------------------------------
python -u -m src.training.preprocessing \
    --src "$SRC_DIR" \
    --dst "$DST_DIR/crops" \
    --size 224 \
    2>&1 | head -n 200   # cap output to 200 lines for readability

echo ""

# ---------------------------------------------------------------------------
# 6. AFTER stats
# ---------------------------------------------------------------------------
python - <<PYTHON
import sys
from pathlib import Path
from PIL import Image

dst   = Path("$DST_DIR/crops")
limit = int("$LIMIT")

exts  = {".jpg", ".jpeg", ".png", ".bmp"}
crops = [p for p in dst.rglob("*") if p.is_file() and p.suffix.lower() in exts]

if not crops:
    print("[WARN] No cropped images found. Preprocessing may have produced no output.")
    print(f"       Expected output under: {dst}")
    sys.exit(0)

print("─" * 60)
print(f"{'AFTER preprocessing (face crops)':^60}")
print("─" * 60)
print(f"  {'File':<38} {'Size (WxH)':>12}  {'Mode':>5}")
print("  " + "-" * 56)

shown = 0
total_after_bytes = 0
for p in sorted(crops)[:limit]:
    try:
        img  = Image.open(p)
        size_str = f"{img.width}x{img.height}"
        mode_str = img.mode
    except Exception as e:
        size_str, mode_str = "ERROR", str(e)[:10]
    kb = p.stat().st_size / 1024
    total_after_bytes += p.stat().st_size
    # Show relative path from dst for readability
    rel = p.relative_to(dst.parent)
    print(f"  {str(rel):<38} {size_str:>12}  {mode_str:>5}  ({kb:.1f} KB)")
    shown += 1

total_all = sum(p.stat().st_size for p in crops)
print()
print(f"  Total crops saved     : {len(crops)}")
print(f"  First {shown} shown above")
print(f"  Total size (after)    : {total_all/1024:.1f} KB")
print()
print(f"  ✔  Output directory: $DST_DIR/crops")
PYTHON

echo ""
echo "============================================================"
echo "  Preview complete."
echo "  Crops saved to: $DST_DIR/crops"
echo "============================================================"
echo ""
