"""
Advanced Exploratory Data Analysis for the real-vs-fake face dataset.

Usage (from project root):
    python src/eda_visual.py --dataset dataset/cleaned --n 8

Generates the following figures in figures/eda_advanced/:
  1. highlight_grid.png         – side-by-side real vs fake sample pairs
  2. compression_artifacts.png  – visual effect of JPEG compression levels
  3. compression_metrics.png    – PSNR/SSIM vs JPEG quality bar chart
  4. fft_spectrum.png           – average 2-D FFT magnitude: real vs fake
  5. dct_energy.png             – HF vs LF DCT energy boxplot: real vs fake
  6. moire_demo.png             – simulated Moiré artefact + FFT peaks
  7. pixel_stats.png            – RGB + luminance histogram: real vs fake
  8. edge_analysis.png          – averaged Sobel edge maps: real vs fake
"""

import argparse
import io
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import scipy.fft as spfft
from scipy.ndimage import sobel

# Optional but recommended for SSIM/PSNR
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

PALETTE = {
    "real":  "#22c55e",   # green
    "fake":  "#ef4444",   # red
    "real2": "#86efac",
    "fake2": "#fca5a5",
    "bg":    "#0f172a",   # near-black for dark panels
    "panel": "#1e293b",
}

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family":      FONT["family"],
    "axes.facecolor":   PALETTE["panel"],
    "figure.facecolor": PALETTE["bg"],
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "axes.edgecolor":   "#334155",
    "grid.color":       "#334155",
})


def _out(figures_dir: Path, name: str) -> str:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return str(figures_dir / name)


def _load_images(file_list, n: int, size: int = 224) -> list[np.ndarray]:
    """Load up to n images as H×W×3 uint8 arrays."""
    random.shuffle(file_list)
    imgs = []
    for path, _ in file_list:
        try:
            img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
            imgs.append(np.array(img, dtype=np.uint8))
        except Exception:
            continue
        if len(imgs) >= n:
            break
    return imgs


def _build_file_list(root: Path, split: str = "train") -> list[tuple[str, int]]:
    """Collect (path, label) from root/{split}/{real|fake}/."""
    candidates = [root / split]
    if split == "val":
        candidates.append(root / "validation")
    for candidate in candidates:
        if candidate.exists():
            split_dir = candidate
            break
    else:
        return []

    files = []
    for label_name, label_val in (("real", 1), ("fake", 0)):
        cls_dir = split_dir / label_name
        if not cls_dir.exists():
            continue
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                files.append((str(p), label_val))
    return files


def _collect_samples(dataset_root: Path, split: str, n_each: int, size: int):
    """Return (real_imgs, fake_imgs) each a list of n_each uint8 arrays."""
    all_real, all_fake = [], []
    for ds_dir in sorted(dataset_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        inner = ds_dir / ds_dir.name
        base = inner if inner.is_dir() else ds_dir
        flist = _build_file_list(base, split)
        all_real.extend([p for p, l in flist if l == 1])
        all_fake.extend([p for p, l in flist if l == 0])
        if len(all_real) >= n_each * 5 and len(all_fake) >= n_each * 5:
            break

    def _load(paths, n):
        random.shuffle(paths)
        imgs = []
        for p in paths:
            try:
                im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
                imgs.append(np.array(im, dtype=np.uint8))
            except Exception:
                continue
            if len(imgs) >= n:
                break
        return imgs

    return _load(all_real, n_each), _load(all_fake, n_each)


# --------------------------------------------------------------------------- #
# 1. HIGHLIGHT GRID
# --------------------------------------------------------------------------- #

def plot_highlight_grid(real_imgs, fake_imgs, out_path: str, n: int = 8):
    n = min(n, len(real_imgs), len(fake_imgs))
    fig = plt.figure(figsize=(2.2 * n, 5.5), facecolor=PALETTE["bg"])
    fig.suptitle("Real vs Fake Face Samples", fontsize=16, fontweight="bold",
                 color="white", y=1.01)

    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.08, wspace=0.04)
    for i in range(n):
        for row, (imgs, label, color) in enumerate(
            [(real_imgs, "Real", PALETTE["real"]),
             (fake_imgs, "Fake", PALETTE["fake"])]
        ):
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(imgs[i])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
            if i == 0:
                ax.set_ylabel(label, color=color, fontsize=12, fontweight="bold",
                              rotation=90, labelpad=6)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path}")


# --------------------------------------------------------------------------- #
# 2 & 3. COMPRESSION ARTIFACTS + METRICS
# --------------------------------------------------------------------------- #

def _jpeg_compress(img_array: np.ndarray, quality: int) -> np.ndarray:
    pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def plot_compression_artifacts(real_imgs, out_path_vis: str, out_path_metrics: str):
    qualities = [95, 75, 50, 25, 10]
    source = real_imgs[0]

    # --- visual panel ---
    n_cols = len(qualities) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 3.2),
                             facecolor=PALETTE["bg"])
    titles = ["Original"] + [f"Q={q}" for q in qualities]
    images_to_show = [source] + [_jpeg_compress(source, q) for q in qualities]

    for ax, img, title in zip(axes, images_to_show, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, color="white")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")

    fig.suptitle("JPEG Compression Levels on a Real Face", fontsize=13,
                 fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(out_path_vis, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path_vis}")

    # --- metrics panel ---
    if not HAS_SKIMAGE:
        print("  ⚠ skimage not available – skipping compression metrics plot")
        return

    psnr_vals, ssim_vals = [], []
    for q in qualities:
        comp = _jpeg_compress(source, q)
        psnr_vals.append(psnr(source, comp, data_range=255))
        ssim_vals.append(ssim(source, comp, channel_axis=2, data_range=255))

    x = np.arange(len(qualities))
    w = 0.38
    fig, ax1 = plt.subplots(figsize=(8, 4.5), facecolor=PALETTE["bg"])
    ax1.set_facecolor(PALETTE["panel"])
    b1 = ax1.bar(x - w / 2, psnr_vals, w, label="PSNR (dB)",
                 color="#6366f1", alpha=0.9)
    ax1.set_ylabel("PSNR (dB)", color="#6366f1")
    ax1.tick_params(axis="y", colors="#6366f1")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Q={q}" for q in qualities])

    ax2 = ax1.twinx()
    ax2.set_facecolor(PALETTE["panel"])
    b2 = ax2.bar(x + w / 2, ssim_vals, w, label="SSIM",
                 color="#f59e0b", alpha=0.9)
    ax2.set_ylabel("SSIM", color="#f59e0b")
    ax2.tick_params(axis="y", colors="#f59e0b")
    ax2.set_ylim(0, 1.05)

    for bar, val in zip(b1, psnr_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="white")
    for bar, val in zip(b2, ssim_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="white")

    handles = [b1, b2]
    labels = ["PSNR (dB)", "SSIM"]
    ax1.legend(handles, labels, loc="upper right", facecolor=PALETTE["panel"],
               edgecolor="#334155", labelcolor="white")
    ax1.set_title("Compression Quality vs PSNR & SSIM", fontsize=13,
                  fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(out_path_metrics, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path_metrics}")


# --------------------------------------------------------------------------- #
# 4. FFT FREQUENCY SPECTRUM
# --------------------------------------------------------------------------- #

def _average_fft_spectrum(imgs: list[np.ndarray]) -> np.ndarray:
    """Compute log-magnitude of averaged 2-D FFT (grayscale)."""
    accum = None
    for img in imgs:
        gray = np.mean(img.astype(np.float32), axis=2)
        f = spfft.fft2(gray)
        f_shifted = spfft.fftshift(f)
        mag = np.abs(f_shifted)
        accum = mag if accum is None else accum + mag
    avg = accum / len(imgs)
    return np.log1p(avg)


def plot_fft_spectrum(real_imgs, fake_imgs, out_path: str):
    real_spec = _average_fft_spectrum(real_imgs)
    fake_spec = _average_fft_spectrum(fake_imgs)
    diff_spec = fake_spec - real_spec

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=PALETTE["bg"])
    panels = [
        (real_spec, "Real — Avg FFT Spectrum", "inferno"),
        (fake_spec, "Fake — Avg FFT Spectrum", "inferno"),
        (diff_spec, "Fake − Real (Diff)", "RdBu_r"),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold", color="white")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("2-D FFT Frequency Analysis: Real vs Fake", fontsize=14,
                 fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path}")


# --------------------------------------------------------------------------- #
# 5. DCT ENERGY ANALYSIS
# --------------------------------------------------------------------------- #

def _dct_hf_energy_ratio(img: np.ndarray, block_size: int = 8) -> float:
    """Average ratio of high-freq DCT energy to total energy over 8x8 blocks."""
    gray = np.mean(img.astype(np.float32), axis=2)
    h, w = gray.shape
    ratios = []
    for r in range(0, h - block_size + 1, block_size):
        for c in range(0, w - block_size + 1, block_size):
            block = gray[r:r + block_size, c:c + block_size]
            d = spfft.dctn(block, norm="ortho")
            total = np.sum(d ** 2) + 1e-9
            # mask: top-left 3x3 is LF, rest is HF
            lf = np.sum(d[:3, :3] ** 2)
            hf_ratio = 1.0 - lf / total
            ratios.append(hf_ratio)
    return float(np.mean(ratios))


def plot_dct_energy(real_imgs, fake_imgs, out_path: str):
    real_ratios = [_dct_hf_energy_ratio(img) for img in real_imgs]
    fake_ratios = [_dct_hf_energy_ratio(img) for img in fake_imgs]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    bp = ax.boxplot(
        [real_ratios, fake_ratios],
        labels=["Real", "Fake"],
        patch_artist=True,
        notch=True,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#94a3b8"),
        capprops=dict(color="#94a3b8"),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    colors = [PALETTE["real"], PALETTE["fake"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual points
    for i, (data, color) in enumerate(
        [(real_ratios, PALETTE["real"]), (fake_ratios, PALETTE["fake"])], start=1
    ):
        jitter = np.random.uniform(-0.15, 0.15, size=len(data))
        ax.scatter(np.full(len(data), i) + jitter, data, color=color,
                   alpha=0.5, s=18, zorder=4)

    ax.set_ylabel("High-Frequency DCT Energy Ratio", fontsize=11)
    ax.set_title("DCT High-Frequency Energy: Real vs Fake", fontsize=13,
                 fontweight="bold", color="white")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path}")


# --------------------------------------------------------------------------- #
# 6. MOIRÉ PATTERN DEMO
# --------------------------------------------------------------------------- #

def _apply_moire(img: np.ndarray, frequency: float = 0.15,
                 angle_deg: float = 5.0, strength: float = 0.35) -> np.ndarray:
    """Overlay a sinusoidal grid pattern to simulate a Moiré effect."""
    h, w = img.shape[:2]
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    X, Y = np.meshgrid(xs, ys)
    angle = np.deg2rad(angle_deg)
    pattern = np.sin(2 * np.pi * frequency * w * (X * np.cos(angle) + Y * np.sin(angle)))
    normalized = (pattern + 1) / 2  # [0, 1]
    overlay = (normalized[:, :, np.newaxis] * 255 * strength).astype(np.float32)
    result = np.clip(img.astype(np.float32) * (1 - strength) + overlay, 0, 255).astype(np.uint8)
    return result


def plot_moire_demo(real_imgs, out_path: str):
    source = real_imgs[1] if len(real_imgs) > 1 else real_imgs[0]
    moire = _apply_moire(source)
    diff = np.clip(source.astype(np.int16) - moire.astype(np.int16) + 128, 0, 255).astype(np.uint8)

    def _fft_log_mag(img):
        gray = np.mean(img.astype(np.float32), axis=2)
        f = spfft.fftshift(spfft.fft2(gray))
        return np.log1p(np.abs(f))

    fig = plt.figure(figsize=(18, 5), facecolor=PALETTE["bg"])
    specs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.08)
    panels = [
        (source,            "Original",          None),
        (moire,             "Moiré Effect",       None),
        (diff,              "Difference (×)",     None),
        (_fft_log_mag(source), "FFT: Original",   "magma"),
        (_fft_log_mag(moire),  "FFT: Moiré",      "magma"),
    ]
    for i, (data, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(specs[i])
        if cmap:
            ax.imshow(data, cmap=cmap, aspect="auto")
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=10, fontweight="bold", color="white")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")

    fig.suptitle("Moiré Pattern Analysis & Frequency Signature", fontsize=14,
                 fontweight="bold", color="white", y=1.02)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path}")


# --------------------------------------------------------------------------- #
# 7. PIXEL STATISTICS
# --------------------------------------------------------------------------- #

def plot_pixel_stats(real_imgs, fake_imgs, out_path: str):
    channel_labels = ["Red", "Green", "Blue"]
    channel_colors_r = ["#f87171", "#86efac", "#93c5fd"]
    channel_colors_f = ["#dc2626", "#16a34a", "#1d4ed8"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), facecolor=PALETTE["bg"])

    # Per-channel histograms
    for i, (label, cr, cf) in enumerate(
        zip(channel_labels, channel_colors_r, channel_colors_f)
    ):
        ax = axes[i]
        ax.set_facecolor(PALETTE["panel"])
        real_ch = np.concatenate([img[:, :, i].ravel() for img in real_imgs])
        fake_ch = np.concatenate([img[:, :, i].ravel() for img in fake_imgs])
        ax.hist(real_ch, bins=64, range=(0, 255), density=True,
                color=cr, alpha=0.55, label="Real")
        ax.hist(fake_ch, bins=64, range=(0, 255), density=True,
                color=cf, alpha=0.55, label="Fake")
        ax.set_title(f"{label} Channel", fontsize=11, fontweight="bold", color="white")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Density" if i == 0 else "")
        ax.legend(facecolor=PALETTE["panel"], edgecolor="#334155", labelcolor="white",
                  fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Luminance histogram
    ax = axes[3]
    ax.set_facecolor(PALETTE["panel"])
    real_lum = np.concatenate([
        (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).ravel()
        for img in real_imgs
    ])
    fake_lum = np.concatenate([
        (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).ravel()
        for img in fake_imgs
    ])
    ax.hist(real_lum, bins=64, range=(0, 255), density=True,
            color=PALETTE["real"], alpha=0.6, label="Real")
    ax.hist(fake_lum, bins=64, range=(0, 255), density=True,
            color=PALETTE["fake"], alpha=0.6, label="Fake")
    ax.set_title("Luminance", fontsize=11, fontweight="bold", color="white")
    ax.set_xlabel("Luminance Value")
    ax.legend(facecolor=PALETTE["panel"], edgecolor="#334155", labelcolor="white",
              fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Pixel-Level Statistics: Real vs Fake", fontsize=14,
                 fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path}")


# --------------------------------------------------------------------------- #
# 8. EDGE / GRADIENT ANALYSIS
# --------------------------------------------------------------------------- #

def _average_edge_map(imgs: list[np.ndarray]) -> np.ndarray:
    """Average Sobel edge magnitude across a list of images."""
    accum = None
    for img in imgs:
        gray = np.mean(img.astype(np.float32), axis=2)
        sx = sobel(gray, axis=1)
        sy = sobel(gray, axis=0)
        mag = np.hypot(sx, sy)
        accum = mag if accum is None else accum + mag
    return accum / len(imgs)


def plot_edge_analysis(real_imgs, fake_imgs, out_path: str):
    real_edge = _average_edge_map(real_imgs)
    fake_edge = _average_edge_map(fake_imgs)
    diff_edge = fake_edge - real_edge

    # Per-image mean edge magnitude (for box-plot)
    def _mean_edge(imgs):
        vals = []
        for img in imgs:
            gray = np.mean(img.astype(np.float32), axis=2)
            sx = sobel(gray, axis=1)
            sy = sobel(gray, axis=0)
            vals.append(np.mean(np.hypot(sx, sy)))
        return vals

    real_mags = _mean_edge(real_imgs)
    fake_mags = _mean_edge(fake_imgs)

    fig = plt.figure(figsize=(16, 5), facecolor=PALETTE["bg"])
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.12)

    # Edge maps
    for i, (data, title, cmap) in enumerate([
        (real_edge, "Avg Edge Map — Real", "hot"),
        (fake_edge, "Avg Edge Map — Fake", "hot"),
        (diff_edge, "Diff (Fake − Real)",  "RdBu_r"),
    ]):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=10, fontweight="bold", color="white")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Box plot of mean edge magnitudes
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor(PALETTE["panel"])
    bp = ax4.boxplot(
        [real_mags, fake_mags], labels=["Real", "Fake"],
        patch_artist=True, notch=True, widths=0.5,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#94a3b8"),
        capprops=dict(color="#94a3b8"),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], [PALETTE["real"], PALETTE["fake"]]):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    for i_bp, (data, color) in enumerate(
        [(real_mags, PALETTE["real"]), (fake_mags, PALETTE["fake"])], start=1
    ):
        jitter = np.random.uniform(-0.15, 0.15, size=len(data))
        ax4.scatter(np.full(len(data), i_bp) + jitter, data, color=color,
                    alpha=0.5, s=18, zorder=4)
    ax4.set_ylabel("Mean Sobel Magnitude")
    ax4.set_title("Edge Strength Distribution", fontsize=10, fontweight="bold",
                  color="white")
    ax4.yaxis.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Sobel Edge Analysis: Real vs Fake", fontsize=14,
                 fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Advanced EDA for real-vs-fake faces")
    parser.add_argument("--dataset", default="dataset/cleaned",
                        help="Root directory containing 'Data Set N' subdirectories")
    parser.add_argument("--n", type=int, default=64,
                        help="Number of real and fake images to sample for statistics")
    parser.add_argument("--size", type=int, default=224,
                        help="Resize images to this square size")
    parser.add_argument("--split", default="train",
                        help="Dataset split to sample from (train/val/test)")
    parser.add_argument("--figures-dir", default="figures/eda_advanced",
                        help="Output directory for figures")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    dataset_root = Path(args.dataset)
    figures_dir  = Path(args.figures_dir)

    print(f"\n📂  Dataset root : {dataset_root.resolve()}")
    print(f"🖼   Images/class : {args.n}")
    print(f"📐  Resize        : {args.size}×{args.size}")
    print(f"📊  Split         : {args.split}")
    print(f"💾  Output dir    : {figures_dir.resolve()}\n")

    print("Loading images …")
    real_imgs, fake_imgs = _collect_samples(dataset_root, args.split, args.n, args.size)
    if not real_imgs or not fake_imgs:
        print("ERROR: no images found – check --dataset path and split names.")
        return
    print(f"  Loaded {len(real_imgs)} real, {len(fake_imgs)} fake images.\n")

    print("Generating figures …")

    # 1. Highlight Grid
    plot_highlight_grid(
        real_imgs, fake_imgs,
        _out(figures_dir, "highlight_grid.png"),
        n=min(8, len(real_imgs), len(fake_imgs))
    )

    # 2 & 3. Compression Artifacts + Metrics
    plot_compression_artifacts(
        real_imgs,
        _out(figures_dir, "compression_artifacts.png"),
        _out(figures_dir, "compression_metrics.png"),
    )

    # 4. FFT Spectrum
    plot_fft_spectrum(
        real_imgs, fake_imgs,
        _out(figures_dir, "fft_spectrum.png")
    )

    # 5. DCT Energy
    plot_dct_energy(
        real_imgs, fake_imgs,
        _out(figures_dir, "dct_energy.png")
    )

    # 6. Moiré Demo
    plot_moire_demo(
        real_imgs,
        _out(figures_dir, "moire_demo.png")
    )

    # 7. Pixel Statistics
    plot_pixel_stats(
        real_imgs, fake_imgs,
        _out(figures_dir, "pixel_stats.png")
    )

    # 8. Edge Analysis
    plot_edge_analysis(
        real_imgs, fake_imgs,
        _out(figures_dir, "edge_analysis.png")
    )

    print(f"\n✅  All figures saved to: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
