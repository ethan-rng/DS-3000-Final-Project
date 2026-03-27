"""
PyTorch training loop and evaluation utilities.

Simple CLI to run a training job using `src.preprocessing` utilities.
Produces comprehensive metrics: accuracy, precision, recall, F1, AUC-ROC,
confusion matrix, classification report — and saves plots to ``figures/``.
"""
import os
from pathlib import Path
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import (
    build_combined_file_list,
    FaceDataset,
    get_train_transform,
    get_eval_transform,
)
from src.model import get_model


# ---------------------------------------------------------------------------
# Device helper (CUDA > MPS > CPU)
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Run one-pass evaluation and return comprehensive metrics.

    Returns a dict with keys:
        accuracy, precision, recall, f1, auc,
        confusion_matrix (2x2 ndarray), classification_report (str),
        y_true (ndarray), y_scores (ndarray)
    """
    model.eval()
    ys: list = []
    scores: list = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
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

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=["fake", "real"], zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_scores": y_scores,
    }


def _print_metrics(metrics: dict, header: str = "Evaluation") -> None:
    print(f"\n{'='*60}")
    print(f" {header}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"  [fake, real]")
    print(f"  {metrics['confusion_matrix']}")
    print(f"\n{metrics['classification_report']}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    dataset_root: str,
    backbone: str = "efficientnet_b0",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    out_dir: str = "models",
    max_samples: int = 0,
    progress_callback=None,
):
    device = _get_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data loading — merge all 4 dataset subdirectories
    # ------------------------------------------------------------------
    train_files = build_combined_file_list(dataset_root, split="train", max_samples_per_class=max_samples)
    val_files = build_combined_file_list(dataset_root, split="val", max_samples_per_class=max_samples)

    if not train_files:
        raise RuntimeError(
            f"No training files found under '{dataset_root}'. "
            "Expected layout: dataset/Data Set N/Data Set N/train/{{real,fake}}/*.jpg"
        )

    print(f"Training images:   {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    train_ds = FaceDataset(train_files, transform=get_train_transform())
    val_ds = FaceDataset(val_files, transform=get_eval_transform()) if val_files else None

    num_workers = 0 if device.type == "mps" else 4
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )
        if val_ds
        else None
    )

    # ------------------------------------------------------------------
    # Model / optimizer / loss
    # ------------------------------------------------------------------
    model = get_model(backbone=backbone).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    # Mixed-precision scaler (CUDA only)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_auc = 0.0
    os.makedirs(out_dir, exist_ok=True)

    # Per-epoch tracking for plots
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_aucs: list[float] = []

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            pbar.set_postfix(loss=f"{epoch_loss / n:.4f}")

        epoch_loss /= n
        elapsed = time.time() - start
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch}/{epochs} — train loss: {epoch_loss:.4f} — {elapsed:.1f}s")

        # Notify caller of progress (used by web UI)
        if progress_callback is not None:
            progress_callback(epoch, epochs, epoch_loss, None)

        # ----- validation -----
        if val_loader is not None:
            # Compute validation loss
            model.eval()
            v_loss = 0.0
            v_n = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    logits = model(imgs)
                    v_loss += criterion(logits, labels).item() * imgs.size(0)
                    v_n += imgs.size(0)
            v_loss /= v_n
            val_losses.append(v_loss)

            val_metrics = evaluate(model, val_loader, device)
            val_auc = val_metrics["auc"]
            val_aucs.append(val_auc)
            print(
                f"  val loss: {v_loss:.4f}  |  "
                f"AUC: {val_auc:.4f}  |  "
                f"Acc: {val_metrics['accuracy']:.4f}  |  "
                f"P: {val_metrics['precision']:.4f}  "
                f"R: {val_metrics['recall']:.4f}  "
                f"F1: {val_metrics['f1']:.4f}"
            )

            if progress_callback is not None:
                progress_callback(epoch, epochs, epoch_loss, val_metrics)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                ckpt_path = Path(out_dir) / f"best_{backbone}.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "backbone": backbone,
                        "epoch": epoch,
                        "val_auc": val_auc,
                    },
                    ckpt_path,
                )
                print(f"  >> Saved best checkpoint to {ckpt_path}")

    # ------------------------------------------------------------------
    # Post-training: test evaluation + visualizations
    # ------------------------------------------------------------------
    print("\n--- Post-training evaluation on TEST splits ---")

    # Reload best checkpoint if it exists
    ckpt_path = Path(out_dir) / f"best_{backbone}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint (epoch {ckpt.get('epoch', '?')}, val AUC {ckpt.get('val_auc', '?'):.4f})")

    test_files = build_combined_file_list(dataset_root, split="test", max_samples_per_class=max_samples)
    if test_files:
        print(f"Test images: {len(test_files)}")
        test_ds = FaceDataset(test_files, transform=get_eval_transform())
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )
        test_metrics = evaluate(model, test_loader, device)
        _print_metrics(test_metrics, header="TEST SET RESULTS")

        # Save metrics to JSON
        metrics_path = Path(out_dir) / "test_metrics.json"
        serialisable = {
            k: v for k, v in test_metrics.items()
            if k not in ("confusion_matrix", "y_true", "y_scores")
        }
        serialisable["confusion_matrix"] = test_metrics["confusion_matrix"].tolist()
        with open(metrics_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Saved test metrics to {metrics_path}")
    else:
        test_metrics = None
        print("No test files found — skipping test evaluation.")

    # ----- visualizations -----
    from src.visualize import (
        plot_training_history,
        plot_confusion_matrix,
        plot_roc_curve,
    )

    plot_training_history(train_losses, val_losses, val_aucs)

    if test_metrics is not None:
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            class_names=["Fake", "Real"],
        )
        plot_roc_curve(test_metrics["y_true"], test_metrics["y_scores"])

    return test_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a real-vs-fake face binary classifier."
    )
    parser.add_argument(
        "--dataset_root", default="dataset",
        help="Top-level dataset/ directory containing Data Set 1-4 (default: dataset)",
    )
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", default="models")
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Cap per-class sample count (0 = all). Useful for quick tests.",
    )
    args = parser.parse_args()

    train(
        dataset_root=args.dataset_root,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
    )
