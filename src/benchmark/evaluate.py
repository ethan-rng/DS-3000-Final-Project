import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import sys
import gc
import glob
import time
import json
import random
import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix,
)
import requests

load_dotenv()

HF_TOKEN       = os.environ.get("HF_TOKEN")
EDENAI_API_KEY = os.environ.get("EDENAI_API_KEY")

# ── Distortion variants ───────────────────────────────────────────────────
DATASET_VARIANTS = {
    "Data Set 1": "Original (no distortion)",
    "Data Set 2": "Compression (JPEG/social-media)",
    "Data Set 3": "Moire (camera-capture)",
    "Data Set 4": "Compression + Moire (combined)",
}

# ── Model configs ─────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    {
        "name": "edenai/deepfake_detection/sightengine",
        "type": "edenai_api",
        "model_string": "image/deepfake_detection/sightengine",
        "provider": "sightengine",
    },
    {
        "name": "edenai/ai_detection/winstonai",
        "type": "edenai_api",
        "model_string": "image/ai_detection/winstonai",
        "provider": "winstonai",
    },
    {
        "name": "prithivMLmods/Deep-Fake-Detector-Model",
        "type": "hf_local",
        "hf_repo": "prithivMLmods/Deep-Fake-Detector-Model",
        "fake_labels": ["fake"],
        "real_labels": ["real"],
    },
    {
        "name": "Wvolf/ViT_Deepfake_Detection",
        "type": "hf_local",
        "hf_repo": "Wvolf/ViT_Deepfake_Detection",
        "fake_labels": ["fake", "deepfake"],
        "real_labels": ["real"],
    },
]

# ── Dataset loading ────────────────────────────────────────────────────────

def load_images(dataset_dir: str, limit: int, seed: int = 42):
    """
    Load shuffled, interleaved real/fake image paths from a split directory.
    Returns (paths, y_true) where y_true: 0=real, 1=fake.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    real_paths = [p for p in glob.glob(os.path.join(dataset_dir, "real", "*.*"))
                  if Path(p).suffix.lower() in exts]
    fake_paths = [p for p in glob.glob(os.path.join(dataset_dir, "fake", "*.*"))
                  if Path(p).suffix.lower() in exts]

    rng = random.Random(seed)
    rng.shuffle(real_paths)
    rng.shuffle(fake_paths)

    real_paths = real_paths[:limit]
    fake_paths = fake_paths[:limit]

    combined = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    rng.shuffle(combined)

    paths  = [p for p, _ in combined]
    y_true = np.array([y for _, y in combined])

    print(f"  Loaded {len(real_paths)} real + {len(fake_paths)} fake = {len(paths)} images (shuffled)")
    return paths, y_true


# ── Inference helpers ──────────────────────────────────────────────────────

def edenai_inference(filepath: str, provider: str, feature: str = "deepfake_detection"):
    url = f"https://api.edenai.run/v2/image/{feature}"
    headers = {"Authorization": f"Bearer {EDENAI_API_KEY}"}
    with open(filepath, "rb") as fh:
        try:
            resp = requests.post(url, headers=headers,
                                 files={"file": fh}, data={"providers": provider})
            pr = resp.json().get(provider, {})
            if pr.get("status") == "success":
                return pr.get("deepfake_score", pr.get("ai_score", 0.5))
            return None
        except Exception as e:
            print(f"\n[{provider}] API Error: {e}")
            return None


def hf_local_score(result: list, cfg: dict) -> float:
    fake_labels = [fl.lower() for fl in cfg.get("fake_labels", ["fake", "deepfake"])]
    real_labels = [rl.lower() for rl in cfg.get("real_labels", ["real"])]
    for entry in result:
        if any(fl in str(entry.get("label", "")).lower() for fl in fake_labels):
            return float(entry["score"])
    for entry in result:
        if any(rl in str(entry.get("label", "")).lower() for rl in real_labels):
            return 1.0 - float(entry["score"])
    return 0.5


# ── Model runners ──────────────────────────────────────────────────────────

def _run_edenai_models(all_paths: list, api_budget: int,
                       y_scores: dict, y_valid: dict):
    """Run all EdenAI API models in parallel threads (network-bound, no RAM issue)."""
    api_cfgs = [cfg for cfg in MODEL_CONFIGS if cfg["type"] == "edenai_api"]
    if not api_cfgs:
        return

    api_counts = {cfg["name"]: 0 for cfg in api_cfgs}

    def _run_one_api(cfg):
        name = cfg["name"]
        idx  = MODEL_CONFIGS.index(cfg)
        for i, img_path in tqdm(enumerate(all_paths), total=len(all_paths),
                                 desc=name[:40], position=idx, leave=True):
            if api_counts[name] >= api_budget:
                print(f"\n  [{name}] Budget ({api_budget}) reached — stopping.")
                break
            try:
                feature = cfg["model_string"].split("/")[1]
                score = edenai_inference(img_path, cfg["provider"], feature)
                api_counts[name] += 1
                time.sleep(0.5)
                if score is not None:
                    y_scores[name][i] = score
                    y_valid[name][i]  = True
            except Exception as e:
                print(f"\n[{name}] Error: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(api_cfgs)) as ex:
        concurrent.futures.wait([ex.submit(_run_one_api, cfg) for cfg in api_cfgs])


def _run_hf_model(cfg: dict, all_paths: list, y_scores: dict, y_valid: dict):
    """
    Load one HF pipeline, infer all images, then DELETE it to free RAM.
    Runs on GPU if available (float16), otherwise CPU with low_cpu_mem_usage.
    """
    import torch
    from transformers import pipeline as hf_pipeline

    name      = cfg["name"]
    local_dir = os.path.join("models", cfg["hf_repo"].split("/")[-1])
    src_path  = local_dir if os.path.isdir(local_dir) else cfg["hf_repo"]

    # Prefer GPU to keep weights off system RAM; fall back to CPU
    if torch.cuda.is_available():
        device     = 0
        dtype_kw   = {"torch_dtype": torch.float16}
        device_txt = f"GPU:{torch.cuda.get_device_name(0)}"
    else:
        device     = -1
        dtype_kw   = {"model_kwargs": {"low_cpu_mem_usage": True}}
        device_txt = "CPU"

    print(f"\n  Loading {name} on {device_txt} ...")
    try:
        pipe = hf_pipeline("image-classification", model=src_path,
                           device=device, **dtype_kw)
    except Exception as e:
        print(f"  [ERROR] Could not load {name}: {e}")
        return

    idx = MODEL_CONFIGS.index(cfg)
    for i, img_path in tqdm(enumerate(all_paths), total=len(all_paths),
                             desc=name[:40], position=idx, leave=True):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            score   = hf_local_score(pipe(pil_img), cfg)
            y_scores[name][i] = score
            y_valid[name][i]  = True
        except Exception as e:
            print(f"\n  [{name}] Error on {img_path}: {e}")

    del pipe
    gc.collect()
    print(f"  ✓ Done + unloaded: {name}")


def run_models_on_split(all_paths: list, y_true: np.ndarray, api_budget: int):
    """
    Run ALL models in parallel via a single ThreadPoolExecutor.
    EdenAI models are network-bound; HF local models are GPU-bound — they don't conflict.
    """
    y_scores = {cfg["name"]: np.full(len(all_paths), np.nan) for cfg in MODEL_CONFIGS}
    y_valid  = {cfg["name"]: np.zeros(len(all_paths), dtype=bool) for cfg in MODEL_CONFIGS}

    def _run_cfg(cfg):
        if cfg["type"] == "edenai_api":
            # Inline single-model EdenAI run
            name = cfg["name"]
            api_count = 0
            idx = MODEL_CONFIGS.index(cfg)
            for i, img_path in tqdm(enumerate(all_paths), total=len(all_paths),
                                     desc=name[:40], position=idx, leave=True):
                if api_count >= api_budget:
                    print(f"\n  [{name}] Budget ({api_budget}) reached — stopping.")
                    break
                try:
                    feature = cfg["model_string"].split("/")[1]
                    score = edenai_inference(img_path, cfg["provider"], feature)
                    api_count += 1
                    time.sleep(0.5)
                    if score is not None:
                        y_scores[name][i] = score
                        y_valid[name][i]  = True
                except Exception as e:
                    print(f"\n[{name}] Error: {e}")
        elif cfg["type"] == "hf_local":
            _run_hf_model(cfg, all_paths, y_scores, y_valid)

    print(f"\n  Running all {len(MODEL_CONFIGS)} models in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODEL_CONFIGS)) as ex:
        concurrent.futures.wait([ex.submit(_run_cfg, cfg) for cfg in MODEL_CONFIGS])

    return y_scores, y_valid


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(name: str, y_true: np.ndarray,
                    y_scores_arr: np.ndarray, valid_mask: np.ndarray):
    y_t = y_true[valid_mask]
    y_s = y_scores_arr[valid_mask]
    y_p = (y_s >= 0.5).astype(int)

    acc  = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec  = float(recall_score(y_t, y_p, zero_division=0))
    f1   = float(f1_score(y_t, y_p, zero_division=0))
    try:
        roc_auc = float(auc(*roc_curve(y_t, y_s)[:2])) if len(np.unique(y_t)) > 1 else float("nan")
    except Exception:
        roc_auc = float("nan")

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    return {
        "model":            name,
        "n_valid":          int(valid_mask.sum()),
        "n_total":          len(y_true),
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "auc":              round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "confusion_matrix": cm.tolist(),
    }, y_t, y_s


# ── Figures ────────────────────────────────────────────────────────────────

def save_per_variant_figures(variant_label: str, results: list,
                             roc_series: list, figures_dir: str):
    safe = variant_label.replace(" ", "_").replace("/", "-")
    vdir = os.path.join(figures_dir, safe)
    os.makedirs(vdir, exist_ok=True)

    if roc_series:
        plt.figure(figsize=(8, 6))
        for name, fpr, tpr, roc_auc in roc_series:
            plt.plot(fpr, tpr, label=f"{name.split('/')[-1]} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC – {variant_label}")
        plt.legend(loc="lower right", fontsize=7)
        plt.savefig(os.path.join(vdir, "roc.png"), bbox_inches="tight")
        plt.close()

    for r in results:
        cm = np.array(r["confusion_matrix"])
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        safe_name = r["model"].replace("/", "_")
        plt.title(f"CM – {safe_name}\n{variant_label}", fontsize=7)
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.savefig(os.path.join(vdir, f"cm_{safe_name}.png"), bbox_inches="tight")
        plt.close()


def save_distortion_comparison_chart(all_variant_results: dict, figures_dir: str):
    """Grouped bar chart: model accuracy across all distortion variants."""
    os.makedirs(figures_dir, exist_ok=True)
    variants = list(all_variant_results.keys())
    model_names = [cfg["name"].split("/")[-1] for cfg in MODEL_CONFIGS]

    x     = np.arange(len(variants))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(variants) * 3), 6))
    for i, mname in enumerate(model_names):
        accs = []
        for v in variants:
            match = next((r for r in all_variant_results[v]
                          if r["model"].split("/")[-1] == mname), None)
            accs.append(match["accuracy"] if match else 0.0)
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=mname)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Model Accuracy Across Distortion Variants")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(figures_dir, "distortion_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart → {out}")


def print_summary_table(all_variant_results: dict):
    print(f"\n{'='*80}")
    print("  DISTORTION BENCHMARK SUMMARY")
    print(f"{'='*80}")
    for variant, results in all_variant_results.items():
        print(f"\n  [{variant}]")
        print(f"  {'Model':<45} {'Acc':>7} {'AUC':>7} {'F1':>7} {'Valid':>10}")
        print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")
        for r in results:
            auc_s   = f"{r['auc']:>7.4f}" if r.get("auc") is not None else "    N/A"
            valid_s = f"{r['n_valid']}/{r['n_total']}"
            print(f"  {r['model']:<45} {r['accuracy']:>7.4f} {auc_s} {r['f1']:>7.4f} {valid_s:>10}")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark deepfake detection models across distortion variants"
    )
    parser.add_argument("--cleaned_dir", default="dataset/cleaned",
                        help="Root containing 'Data Set 1'..'Data Set 4' (default: dataset/cleaned)")
    parser.add_argument("--split", default="test",
                        help="Split to evaluate: test|validation|train (default: test)")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max images per class per variant (default: 500)")
    parser.add_argument("--api_budget", type=int, default=3000,
                        help="Max API calls per EdenAI model per variant (default: 3000)")
    parser.add_argument("--out", default="models/benchmark_distortion_results.json")
    parser.add_argument("--figures_dir", default="figures/distortion_benchmark")
    parser.add_argument("--no_download", action="store_true",
                        help="Skip downloading HF models")
    parser.add_argument("--variants", nargs="+", default=list(DATASET_VARIANTS.keys()),
                        help="Which dataset variants to run (default: all four)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.no_download:
        from src.benchmark.download_models import download_benchmark_models
        download_benchmark_models()

    all_variant_results = {}
    all_json_records    = []

    for ds_key in args.variants:
        variant_label = DATASET_VARIANTS.get(ds_key, ds_key)
        split_dir     = os.path.join(args.cleaned_dir, ds_key, args.split)

        print(f"\n{'='*60}")
        print(f"  VARIANT: {variant_label}")
        print(f"  Dir:     {split_dir}")
        print(f"{'='*60}")

        if not os.path.isdir(split_dir):
            print(f"  [SKIP] Not found: {split_dir}")
            continue

        all_paths, y_true = load_images(split_dir, args.limit, seed=args.seed)
        if not all_paths:
            print(f"  [SKIP] No images in {split_dir}")
            continue

        y_scores, y_valid = run_models_on_split(all_paths, y_true, api_budget=args.api_budget)

        variant_results = []
        roc_series      = []

        for cfg in MODEL_CONFIGS:
            name       = cfg["name"]
            valid_mask = y_valid[name]
            if not np.any(valid_mask):
                print(f"\n  [SKIP] {name} — no valid predictions")
                continue

            result, y_t, y_s = compute_metrics(name, y_true, y_scores[name], valid_mask)
            result["variant"] = variant_label
            result["ds_key"]  = ds_key
            variant_results.append(result)
            all_json_records.append(result)

            if len(np.unique(y_t)) > 1:
                fpr, tpr, _ = roc_curve(y_t, y_s)
                roc_series.append((name, fpr, tpr, auc(fpr, tpr)))

        all_variant_results[variant_label] = variant_results
        save_per_variant_figures(variant_label, variant_results, roc_series, args.figures_dir)

    if all_variant_results:
        print_summary_table(all_variant_results)
        save_distortion_comparison_chart(all_variant_results, args.figures_dir)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(all_json_records, fh, indent=2)
    print(f"\nFull results → {args.out}")


if __name__ == "__main__":
    main()
