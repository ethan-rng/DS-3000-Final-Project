"""
Sklearn-based training pipeline for traditional ML models on the
real-vs-fake face classification task.

Supported model types:
  logistic_regression, svm, random_forest, knn

Uses 64×64 flattened pixel features (12 288-d vectors).
Returns the same metrics dict format as the PyTorch ``train()`` function
so the web UI can display results identically.
"""
import os
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

from src.training.preprocessing import build_combined_file_list


# ── Feature extraction ────────────────────────────────────────────────────

def extract_features(
    file_list: List[Tuple[str, int]],
    size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load images, resize to *size*×*size*, flatten to 1-D vectors.

    Returns (X, y) where X has shape (n, size*size*3) with values in [0, 1].
    """
    X, y = [], []
    for img_path, label in file_list:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((size, size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32).ravel() / 255.0
            X.append(arr)
            y.append(label)
        except Exception:
            continue  # skip corrupt files
    return np.array(X), np.array(y)


# ── Model factory ─────────────────────────────────────────────────────────

_MODEL_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, solver="saga", C=1.0, n_jobs=-1,
    ),
    "svm": lambda: SVC(
        kernel="rbf", probability=True, C=1.0,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42,
    ),
    "knn": lambda: KNeighborsClassifier(
        n_neighbors=5, n_jobs=-1,
    ),
}


# ── Main training entry point ─────────────────────────────────────────────

def train_sklearn(
    dataset_root: str,
    model_type: str = "logistic_regression",
    out_dir: str = "models",
    max_samples: int = 0,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Train a traditional ML model and return evaluation metrics.

    Parameters match the signature expected by ``app._train_worker``.
    ``progress_callback(epoch, total_epochs, train_loss, val_metrics)`` is
    called once when training starts and once when it finishes.
    """
    if model_type not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown sklearn model type '{model_type}'. "
            f"Choose from: {set(_MODEL_BUILDERS.keys())}"
        )

    # Notify UI that training has started
    if progress_callback:
        progress_callback(0, 1, 0.0, None)

    # ── Load data ─────────────────────────────────────────────────────
    print(f"[sklearn] Loading data from {dataset_root} …")
    train_files = build_combined_file_list(
        dataset_root, split="train", max_samples_per_class=max_samples,
    )
    test_files = build_combined_file_list(
        dataset_root, split="test", max_samples_per_class=max_samples,
    )

    if not train_files:
        raise RuntimeError(
            f"No training files found under '{dataset_root}'. "
            "Expected layout: dataset/Data Set N/Data Set N/train/{real,fake}/*.jpg"
        )

    print(f"[sklearn] Extracting features (64×64 flatten) …")
    X_train, y_train = extract_features(train_files)
    X_test, y_test = extract_features(test_files) if test_files else (None, None)
    print(f"[sklearn] Train samples: {len(X_train)}, Test samples: {len(X_test) if X_test is not None else 0}")

    # ── Train ─────────────────────────────────────────────────────────
    print(f"[sklearn] Training {model_type} …")
    start = time.time()
    model = _MODEL_BUILDERS[model_type]()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"[sklearn] Training finished in {elapsed:.1f}s")

    # ── Save model ────────────────────────────────────────────────────
    model_out_dir = os.path.join(out_dir, model_type)
    os.makedirs(model_out_dir, exist_ok=True)
    fig_out_dir = os.path.join("figures", model_type)
    os.makedirs(fig_out_dir, exist_ok=True)
    print(f"[sklearn] Checkpoints & metrics → {model_out_dir}/")
    print(f"[sklearn] Figures               → {fig_out_dir}/")
    model_path = Path(model_out_dir) / f"best_{model_type}.pkl"
    joblib.dump(model, model_path)
    print(f"[sklearn] Saved model to {model_path}")

    # ── Evaluate on test set ──────────────────────────────────────────
    test_metrics = None
    if X_test is not None and len(X_test) > 0:
        y_pred = model.predict(X_test)
        y_scores = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )

        try:
            auc = roc_auc_score(y_test, y_scores)
        except Exception:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=["fake", "real"], zero_division=0,
        )

        test_metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(auc),
            "confusion_matrix": cm,
            "classification_report": report,
            "y_true": y_test,
            "y_scores": y_scores,
        }

        print(f"[sklearn] Test Accuracy: {test_metrics['accuracy']:.4f}  "
              f"AUC: {test_metrics['auc']:.4f}")

        # Save metrics JSON
        metrics_path = Path(model_out_dir) / "test_metrics.json"
        serialisable = {
            k: v for k, v in test_metrics.items()
            if k not in ("confusion_matrix", "y_true", "y_scores")
        }
        serialisable["confusion_matrix"] = cm.tolist()
        with open(metrics_path, "w") as f:
            json.dump(serialisable, f, indent=2)

        # ── Visualizations ────────────────────────────────────────────
        from src.visualize import plot_confusion_matrix, plot_roc_curve
        import datetime
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_confusion_matrix(
            cm,
            class_names=["Fake", "Real"],
            filename=f"confusion_matrix_{model_type}_{_ts}.png",
            out_dir=fig_out_dir,
        )
        plot_roc_curve(
            y_test,
            y_scores,
            filename=f"roc_curve_{model_type}_{_ts}.png",
            out_dir=fig_out_dir,
        )
    else:
        print("[sklearn] No test files found — skipping evaluation.")

    # Notify UI that training is done
    if progress_callback and test_metrics:
        progress_callback(1, 1, 0.0, test_metrics)

    return test_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a traditional ML model for real-vs-fake face classification."
    )
    parser.add_argument(
        "--dataset_root", default="dataset",
        help="Top-level dataset/ directory containing Data Set 1-4 (default: dataset)",
    )
    parser.add_argument(
        "--model_type", default="random_forest",
        choices=["logistic_regression", "svm", "random_forest", "knn"],
        help="Which sklearn model to train (default: random_forest)",
    )
    parser.add_argument(
        "--out_dir", default="models",
        help="Directory to save the model .pkl and metrics JSON (default: models)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Cap per-class sample count (0 = all). Useful for quick tests.",
    )
    args = parser.parse_args()

    train_sklearn(
        dataset_root=args.dataset_root,
        model_type=args.model_type,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
    )
