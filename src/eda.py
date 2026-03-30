"""
Exploratory Data Analysis (EDA) for the real-vs-fake face dataset.

Generates summary statistics and three plots:
  - eda_class_distribution.png  — fake vs real counts per split
  - eda_dataset_breakdown.png   — images per dataset
  - eda_sample_grid.png         — grid of sample images from each class

All plots saved to ``figures/``.
"""
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.preprocessing import build_file_list


FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def _ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def run_eda(dataset_root: str) -> dict:
    """Analyse the dataset and return a summary dict + generate plots.

    Parameters
    ----------
    dataset_root : str
        Top-level directory containing ``Data Set 1/``, ``Data Set 2/``, etc.

    Returns
    -------
    dict with keys: ``summary`` (str), ``stats`` (dict of counts).
    """
    root = Path(dataset_root)

    # ── Gather counts ─────────────────────────────────────────────────
    splits = ["train", "val", "test"]
    # {dataset_name: {split: {label: count}}}
    per_dataset = {}
    # {split: {label: count}}
    totals = defaultdict(lambda: defaultdict(int))

    ds_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    for ds_dir in ds_dirs:
        ds_name = ds_dir.name
        inner = ds_dir / ds_dir.name
        crops_root = str(inner) if inner.is_dir() else str(ds_dir)

        per_dataset[ds_name] = {}
        for split in splits:
            files = build_file_list(crops_root, split=split)
            label_counts = defaultdict(int)
            for _, label in files:
                label_name = "real" if label == 1 else "fake"
                label_counts[label_name] += 1
                totals[split][label_name] += 1
            per_dataset[ds_name][split] = dict(label_counts)

    # ── Build summary string ──────────────────────────────────────────
    lines = ["Dataset Summary", "=" * 50]
    grand_total = 0
    for split in splits:
        split_total = sum(totals[split].values())
        grand_total += split_total
        fake_n = totals[split].get("fake", 0)
        real_n = totals[split].get("real", 0)
        lines.append(f"  {split:>10s}:  {split_total:>6,d} images  "
                      f"(fake: {fake_n:,d}, real: {real_n:,d})")
    lines.append(f"{'':>10s}   {'─' * 36}")
    lines.append(f"  {'TOTAL':>10s}:  {grand_total:>6,d} images")
    lines.append("")

    for ds_name, splits_data in per_dataset.items():
        ds_total = sum(
            sum(lc.values()) for lc in splits_data.values()
        )
        lines.append(f"  {ds_name}: {ds_total:,d} images")
        for split in splits:
            sc = splits_data.get(split, {})
            if sc:
                lines.append(
                    f"    {split:>10s}: fake={sc.get('fake', 0):,d}  "
                    f"real={sc.get('real', 0):,d}"
                )
    summary_text = "\n".join(lines)

    # ── Plot 1: Class distribution per split ──────────────────────────
    _ensure_figures_dir()

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(splits))
    width = 0.35
    fake_counts = [totals[s].get("fake", 0) for s in splits]
    real_counts = [totals[s].get("real", 0) for s in splits]
    bars1 = ax.bar(x - width / 2, fake_counts, width, label="Fake", color="#ef4444")
    bars2 = ax.bar(x + width / 2, real_counts, width, label="Real", color="#22c55e")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution per Split")
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{int(h):,}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_class_distribution.png"), dpi=150)
    plt.close(fig)

    # ── Plot 2: Per-dataset breakdown ─────────────────────────────────
    ds_names = list(per_dataset.keys())
    ds_totals = [
        sum(sum(lc.values()) for lc in per_dataset[d].values())
        for d in ds_names
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"]
    ax.barh(ds_names, ds_totals, color=colors[: len(ds_names)])
    ax.set_xlabel("Number of Images")
    ax.set_title("Images per Dataset")
    for i, v in enumerate(ds_totals):
        ax.text(v + max(ds_totals) * 0.01, i, f"{v:,}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_dataset_breakdown.png"), dpi=150)
    plt.close(fig)

    # ── Plot 3: Sample image grid ─────────────────────────────────────
    # Grab a few sample images from training split
    all_train_files = []
    for ds_dir in ds_dirs:
        inner = ds_dir / ds_dir.name
        crops_root = str(inner) if inner.is_dir() else str(ds_dir)
        all_train_files.extend(build_file_list(crops_root, split="train"))

    random.seed(42)
    fake_files = [f for f in all_train_files if f[1] == 0]
    real_files = [f for f in all_train_files if f[1] == 1]
    n_samples = min(4, len(fake_files), len(real_files))

    if n_samples > 0:
        fake_sample = random.sample(fake_files, n_samples)
        real_sample = random.sample(real_files, n_samples)

        fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
        if n_samples == 1:
            axes = axes.reshape(2, 1)
        for i in range(n_samples):
            for row, (sample, label) in enumerate(
                [(fake_sample[i], "Fake"), (real_sample[i], "Real")]
            ):
                ax = axes[row][i]
                try:
                    img = Image.open(sample[0]).convert("RGB").resize((128, 128))
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center")
                ax.set_title(label, fontsize=10, fontweight="bold",
                             color="#ef4444" if label == "Fake" else "#22c55e")
                ax.axis("off")
        plt.suptitle("Sample Training Images", fontsize=13, fontweight="bold", y=1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "eda_sample_grid.png"), dpi=150)
        plt.close(fig)
    else:
        # Create a placeholder
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No images available for preview",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        plt.savefig(os.path.join(FIGURES_DIR, "eda_sample_grid.png"), dpi=150)
        plt.close(fig)

    return {
        "summary": summary_text,
        "stats": {
            "totals": {s: dict(totals[s]) for s in splits},
            "per_dataset": {
                ds: {s: per_dataset[ds].get(s, {}) for s in splits}
                for ds in per_dataset
            },
            "grand_total": grand_total,
        },
    }
