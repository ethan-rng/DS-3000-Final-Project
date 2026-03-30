"""
Benchmark pretrained deepfake detection models from HuggingFace
against our test dataset.
Models benchmarked:
    1. prithivMLmods/Deep-Fake-Detector-v2-Model (ViT-based)
    2. prithivMLmods/Deep-Fake-Detector-Model (ViT-based)
    3. Wvolf/ViT_Deepfake_Detection (ViT-based)
"""

import os
import glob
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ── Model configs ─────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "Deep-Fake-Detector-v2 (ViT)",
        "repo": "prithivMLmods/Deep-Fake-Detector-v2-Model",
        "fake_labels": ["deepfake", "fake", "Deepfake"],
        "real_labels": ["realism", "real", "Realism", "Real"],
    },
    {
        "name": "Deep-Fake-Detector (ViT)",
        "repo": "prithivMLmods/Deep-Fake-Detector-Model",
        "fake_labels": ["fake", "Fake"],
        "real_labels": ["real", "Real"],
    },
    {
        "name": "ViT-Deepfake-Detection",
        "repo": "Wvolf/ViT_Deepfake_Detection",
        "fake_labels": ["fake", "Fake", "FAKE", "deepfake"],
        "real_labels": ["real", "Real", "REAL"],
    },
]

def load_test_images(dataset_root: str, limit: int = 100):
    """Load test image paths and labels from dataset directory."""
    root = Path(dataset_root)
    real_paths = []
    fake_paths = []

    # Search for test splits across all Data Set subdirectories
    for pattern in ["**/test/real/*.*", "**/test/fake/*.*"]:
        for p in root.glob(pattern):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                if "real" in p.parts:
                    real_paths.append(str(p))
                else:
                    fake_paths.append(str(p))

    real_paths = real_paths[:limit]
    fake_paths = fake_paths[:limit]

    all_paths = real_paths + fake_paths
    # 0 = real, 1 = fake
    y_true = [0] * len(real_paths) + [1] * len(fake_paths)

    print(f"Loaded {len(real_paths)} real and {len(fake_paths)} fake test images")
    return all_paths, y_true


def run_model(model_cfg: dict, image_paths: list) -> list:
    """Run a HuggingFace pipeline on all images. Returns fake probability per image."""
    from transformers import pipeline

    print(f"\nLoading {model_cfg['name']} ...")
    try:
        pipe = pipeline(
            "image-classification",
            model=model_cfg["repo"],
            device=-1,  # CPU
        )
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return None

    scores = []
    for img_path in tqdm(image_paths, desc=model_cfg["name"]):
        try:
            image = Image.open(img_path).convert("RGB")
            result = pipe(image)

            # Find the fake probability
            fake_score = None
            for entry in result:
                label = str(entry["label"]).lower()
                if any(fl.lower() in label for fl in model_cfg["fake_labels"]):
                    fake_score = entry["score"]
                    break

            if fake_score is None:
                # If no fake label found, use 1 - real score
                for entry in result:
                    label = str(entry["label"]).lower()
                    if any(rl.lower() in label for rl in model_cfg["real_labels"]):
                        fake_score = 1.0 - entry["score"]
                        break

            if fake_score is None:
                fake_score = 0.5  # fallback

            scores.append(fake_score)

        except Exception as e:
            scores.append(0.5)  # fallback on error

    return scores


def print_results(name: str, y_true: list, y_scores: list):
    """Print and return metrics for a model."""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    [[TN={cm[0][0]} FP={cm[0][1]}]")
    print(f"     [FN={cm[1][0]} TP={cm[1][1]}]]")

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="dataset",
        help="Path to dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max images per class to test (default: 100)",
    )
    parser.add_argument(
        "--out",
        default="models/benchmark_results.json",
        help="Output file for results (default: models/benchmark_results.json)",
    )
    args = parser.parse_args()

    # Load test images
    image_paths, y_true = load_test_images(args.dataset_root, args.limit)
    if not image_paths:
        print("No test images found! Check your dataset_root path.")
        return

    # Run each model
    all_results = []
    for model_cfg in MODELS:
        scores = run_model(model_cfg, image_paths)
        if scores is not None:
            result = print_results(model_cfg["name"], y_true, scores)
            all_results.append(result)

    # Print summary table
    print(f"\n{'='*70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'Acc':>7} {'AUC':>7} {'F1':>7}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7}")

    # Add our EfficientNet results for comparison
    our_model = {
        "model": "Our EfficientNet-B0 (20 epochs)",
        "accuracy": 0.7075,
        "auc": 0.7911,
        "f1": 0.6913,
    }
    all_results.append(our_model)

    for r in all_results:
        print(f"  {r['model']:<35} {r['accuracy']:>7.4f} {r['auc']:>7.4f} {r['f1']:>7.4f}")

    # Save results
    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
