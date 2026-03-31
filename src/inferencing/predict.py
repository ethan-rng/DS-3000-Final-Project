"""
Predict whether a single face image is REAL or AI-GENERATED.

Runs all trained models by default, or a specific one with --backbone.

Usage (from project root):
    python -m src.inferencing.predict path/to/image.jpg
    python -m src.inferencing.predict path/to/image.jpg --backbone vit
    python -m src.inferencing.predict path/to/image.jpg --backbone efficientnet_cnn
    python -m src.inferencing.predict path/to/image.jpg --model models/best_vit.pt --backbone vit

Supported backbone names:
    efficientnet_cnn, dct_cnn, denoised_cnn, vit
"""

import argparse
import importlib
import sys
from pathlib import Path

import torch
from PIL import Image

from src.training.preprocessing import get_eval_transform

# ── Model registry ─────────────────────────────────────────────────────────────
# Maps backbone name → (model_module_path, checkpoint_filename)
MODEL_REGISTRY = {
    "efficientnet_cnn": ("src.models.efficientnet_cnn.model", "models/efficientnet_cnn/best_efficientnet_cnn.pt"),
    "dct_cnn":          ("src.models.dct_cnn.model",          "models/dct_cnn/best_dct_cnn.pt"),
    "denoised_cnn":     ("src.models.denoised_cnn.model",     "models/denoised_cnn/best_denoised_cnn.pt"),
    "vit":              ("src.models.vit.model",               "models/vit/best_vit.pt"),
}

ALL_BACKBONES = list(MODEL_REGISTRY.keys())


def _load_model(backbone: str, model_path: str, device: torch.device):
    """Load the correct model class and checkpoint for a given backbone."""
    if backbone not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from: {', '.join(ALL_BACKBONES)}"
        )
    module_path, default_ckpt = MODEL_REGISTRY[backbone]
    ckpt = model_path or default_ckpt

    if not Path(ckpt).exists():
        raise FileNotFoundError(ckpt)

    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)

    module = importlib.import_module(module_path)
    model = module.get_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict_single(image_path: str, backbone: str, model_path: str, device: torch.device):
    """Run inference for one backbone and return result dict."""
    _, default_ckpt = MODEL_REGISTRY[backbone]
    ckpt = model_path or default_ckpt

    if not Path(ckpt).exists():
        return {"backbone": backbone, "error": f"checkpoint not found: {ckpt}"}

    try:
        model = _load_model(backbone, model_path, device)
    except Exception as e:
        return {"backbone": backbone, "error": str(e)}

    # Convert any format (PNG, WEBP, BMP, etc.) to JPEG in-memory for consistency
    raw = Image.open(image_path).convert("RGB")
    from io import BytesIO
    buf = BytesIO()
    raw.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    transform = get_eval_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob_real = torch.sigmoid(logit).item()
        prob_fake = 1.0 - prob_real

    label = "REAL 🟢" if prob_real >= 0.5 else "AI-GENERATED / FAKE 🔴"
    confidence = max(prob_real, prob_fake) * 100

    return {
        "backbone": backbone,
        "label": label,
        "confidence": confidence,
        "prob_real": prob_real * 100,
        "prob_fake": prob_fake * 100,
    }


def predict(image_path: str, backbones: list = None, model_path: str = None):  # type: ignore[assignment]
    """Run inference on an image using one or all models."""
    # ── Device ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    targets = backbones if backbones else ALL_BACKBONES

    print(f"\nImage: {Path(image_path).name}  |  Device: {device}\n")
    print("=" * 55)

    for backbone in targets:
        result = predict_single(image_path, backbone, model_path, device)
        if "error" in result:
            print(f"  [{backbone}]  ⚠️  Skipped — {result['error']}")
        else:
            print(f"  [{backbone}]")
            print(f"    Prediction: {result['label']}")
            print(f"    Confidence: {result['confidence']:.1f}%  "
                  f"(P(real)={result['prob_real']:.1f}%  P(fake)={result['prob_fake']:.1f}%)")
        print("-" * 55)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict real vs AI-generated face image.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "--backbone",
        choices=ALL_BACKBONES,
        default=None,
        help="Run a specific model backbone (default: all models)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a specific model checkpoint (.pt). Only used with --backbone.",
    )
    args = parser.parse_args()

    backbones = [args.backbone] if args.backbone else None
    predict(args.image, backbones=backbones, model_path=args.model)
