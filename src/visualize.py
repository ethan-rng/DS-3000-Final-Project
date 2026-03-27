"""
Visualization and plotting utilities for PyTorch training results.

All plots are saved to the ``figures/`` directory by default.
"""
import os
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works headless / on servers
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def _ensure_figures_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    val_aucs: Optional[List[float]] = None,
    save: bool = True,
) -> None:
    """Plot training / validation loss and (optionally) validation AUC curves."""
    ncols = 2 if val_aucs else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # --- Loss ---
    axes[0].plot(epochs, train_losses, label="Train Loss")
    if val_losses:
        axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # --- AUC ---
    if val_aucs and ncols > 1:
        axes[1].plot(epochs, val_aucs, label="Val AUC", color="tab:green")
        axes[1].set_title("Validation AUC per Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].legend()

    plt.tight_layout()

    if save:
        _ensure_figures_dir()
        path = os.path.join(FIGURES_DIR, "training_history.png")
        plt.savefig(path, dpi=150)
        print(f"Saved training history plot to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save: bool = True,
) -> None:
    """Plot a 2x2 confusion matrix heatmap."""
    if class_names is None:
        class_names = ["Fake", "Real"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if save:
        _ensure_figures_dir()
        path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"Saved confusion matrix plot to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save: bool = True,
) -> None:
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save:
        _ensure_figures_dir()
        path = os.path.join(FIGURES_DIR, "roc_curve.png")
        plt.savefig(path, dpi=150)
        print(f"Saved ROC curve plot to {path}")
    plt.close(fig)
