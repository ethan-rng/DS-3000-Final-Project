"""
Predict whether a single face image is REAL or AI-GENERATED.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --backbone efficientnet_b0
    python predict.py path/to/image.jpg --model models/best_efficientnet_b0.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

try:
    from src.training.preprocessing import get_eval_transform
    from src.training.model import get_model
except ImportError:
    from src.preprocessing import get_eval_transform
    from src.model import get_model


def predict(image_path: str, model_path: str = None, backbone: str = "efficientnet_b0"):
    # ── Device ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Find model checkpoint ─────────────────────────────────────────
    if model_path is None:
        model_path = f"models/best_{backbone}.pt"

    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Have you trained the model yet? Run the web UI and click 'Start Training'.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    backbone_name = checkpoint.get("backbone", backbone)

    model = get_model(backbone=backbone_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # ── Load & preprocess image ───────────────────────────────────────
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    transform = get_eval_transform()
    tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    # ── Run inference ─────────────────────────────────────────────────
    with torch.no_grad():
        logit = model(tensor)
        prob_real = torch.sigmoid(logit).item()
        prob_fake = 1.0 - prob_real

    label = "REAL 🟢" if prob_real >= 0.5 else "AI-GENERATED / FAKE 🔴"
    confidence = max(prob_real, prob_fake) * 100

    print("\n" + "=" * 45)
    print(f"  Image:      {Path(image_path).name}")
    print(f"  Prediction: {label}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  P(real):    {prob_real*100:.1f}%")
    print(f"  P(fake):    {prob_fake*100:.1f}%")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict real vs AI-generated face image.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--backbone", default="efficientnet_b0", help="Model backbone name")
    parser.add_argument("--model", default=None, help="Path to model checkpoint (.pt file)")
    args = parser.parse_args()

    predict(args.image, model_path=args.model, backbone=args.backbone)
