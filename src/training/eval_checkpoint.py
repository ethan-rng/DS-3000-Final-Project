"""
Evaluate a saved checkpoint against a dataset split.

Usage:
    python -m src.training.eval_checkpoint --checkpoint models/best_efficientnet_cnn.pt --dataset_root dataset/cleaned/raw
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.training.preprocessing import (
    build_combined_file_list,
    FaceDataset,
    get_eval_transform,
)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ys, scores = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys.extend(labels.numpy().tolist())
            scores.extend(probs.tolist())

    y_true = np.array(ys)
    y_scores = np.array(scores)
    y_pred = (y_scores >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["fake", "real"], zero_division=0
        ),
        "y_true": y_true,
        "y_scores": y_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a dataset split.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--dataset_root", default="dataset/cleaned/raw")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    device = _get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_type = ckpt.get("model_type", "efficientnet_cnn")
    print(f"Model type:   {model_type}")
    print(f"Trained epoch: {ckpt.get('epoch', '?')}  |  val AUC: {ckpt.get('val_auc', '?')}")

    model_module = importlib.import_module(f"src.models.{model_type}.model")
    model = model_module.get_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    files = build_combined_file_list(args.dataset_root, split=args.split, max_samples_per_class=args.max_samples)
    if not files:
        print(f"No images found for split='{args.split}' under '{args.dataset_root}'")
        return

    print(f"Images ({args.split}): {len(files)}")
    num_workers = 0 if device.type == "mps" else 4
    loader = DataLoader(
        FaceDataset(files, transform=get_eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    metrics = evaluate(model, loader, device)

    print(f"\n{'='*60}")
    print(f" Results — {args.split.upper()} split")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"  {metrics['confusion_matrix']}")
    print(f"\n{metrics['classification_report']}")

    out_path = Path(args.checkpoint).with_suffix(f".{args.split}_metrics.json")
    serialisable = {k: v for k, v in metrics.items() if k not in ("confusion_matrix", "y_true", "y_scores")}
    serialisable["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
