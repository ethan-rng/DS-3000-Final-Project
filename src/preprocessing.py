"""
Image preprocessing utilities for the real-vs-fake face classifier.

Functions:
- gather_image_paths: enumerate images and labels from source folders
- build_file_list: build (path, label) list from a single dataset root
- build_combined_file_list: merge file lists across all 4 datasets
- detect_and_align: detect faces and save aligned crops (uses facenet-pytorch MTCNN if available)
- FaceDataset: PyTorch Dataset loading cropped images and applying transforms
- get_train_transform / get_eval_transform: augmentation vs clean transforms

Notes: facenet-pytorch is optional; if unavailable the code falls back to a center-crop detector.
"""
from pathlib import Path
import os
import random
from typing import List, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform(size: int = 224) -> transforms.Compose:
    """Augmented transform for training images."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform(size: int = 224) -> transforms.Compose:
    """Clean transform for validation / test images."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def gather_image_paths(root_dir: str, exts: Optional[List[str]] = None) -> List[Tuple[str, str, str]]:
    """Recursively find image files and return tuples (dataset_split, label, path).

    Expects folder names containing `train`, `validation`/`val`, or `test`, and class folders named
    `real` and `fake` (case-insensitive). If these names don't match your layout, call this function
    only as a helper and adapt the filtering.
    """
    if exts is None:
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
    root = Path(root_dir)
    results = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            parts = [part.lower() for part in p.parts]
            split = None
            if any(s in parts for s in ("train", "training")):
                split = "train"
            elif any(s in parts for s in ("val", "validation")):
                split = "val"
            elif "test" in parts:
                split = "test"
            else:
                split = "unknown"

            label = "real" if "real" in parts else ("fake" if "fake" in parts else "unknown")
            results.append((split, label, str(p)))
    return results


def _center_crop_face(img: Image.Image, size: int = 224) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.BILINEAR)


def detect_and_align(
    image_path: str,
    output_path: str,
    size: int = 224,
    mtcnn=None,
) -> bool:
    """Detect and align a face from `image_path` and save crop to `output_path`.

    If `mtcnn` (from facenet_pytorch) is provided, it will be used; otherwise a center crop fallback is used.
    Returns True if a face crop was saved.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False

    if mtcnn is not None:
        try:
            # mtcnn returns a PIL image if keep_all=False
            crop = mtcnn(img)
            if crop is None:
                raise RuntimeError("mtcnn returned None")
            if isinstance(crop, torch.Tensor):
                # convert tensor to PIL
                crop = transforms.ToPILImage()(crop)
            crop = crop.resize((size, size), Image.BILINEAR)
            crop.save(out_p)
            return True
        except Exception:
            # fall through to center-crop
            pass

    # fallback
    crop = _center_crop_face(img, size=size)
    crop.save(out_p)
    return True


class FaceDataset(Dataset):
    """PyTorch Dataset for face crops stored in folders: split/label/*.jpg

    Parameters:
    - file_list: list of tuples (image_path, label)
    - transform: torchvision transforms to apply to the PIL image
    """

    def __init__(self, file_list: List[Tuple[str, int]], transform=None):
        self.files = file_list
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        img_path, label = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def build_file_list(crops_root: str, split: str = "train") -> List[Tuple[str, int]]:
    """Build a list of (path, label) from `crops_root/{split}/{real|fake}`.

    Handles the 'validation' vs 'val' folder name mismatch automatically.
    Returns an empty list if paths are not found.
    """
    root = Path(crops_root)
    out: List[Tuple[str, int]] = []

    # Try both 'val' and 'validation' folder names
    candidates = [root / split]
    if split == "val":
        candidates.append(root / "validation")
    elif split == "validation":
        candidates.insert(0, root / "validation")
        candidates.append(root / "val")

    target = None
    for c in candidates:
        if c.exists():
            target = c
            break

    if target is None:
        return out

    for label_name, label_value in (("real", 1), ("fake", 0)):
        class_dir = target / label_name
        if not class_dir.exists():
            continue
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                out.append((str(p), label_value))
    return out


def build_combined_file_list(
    dataset_root: str,
    split: str = "train",
    max_samples_per_class: int = 0,
) -> List[Tuple[str, int]]:
    """Merge file lists across all 'Data Set N/Data Set N/' sub-directories.

    Parameters
    ----------
    dataset_root : str
        Top-level ``dataset/`` directory containing ``Data Set 1/``, etc.
    split : str
        One of ``'train'``, ``'val'``/``'validation'``, ``'test'``.
    max_samples_per_class : int
        If > 0, randomly sample at most this many images *per class* to speed
        up experimentation. 0 means use all images.
    """
    root = Path(dataset_root)
    combined: List[Tuple[str, int]] = []

    # Discover dataset sub-directories: dataset/Data Set N/Data Set N/
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        inner = ds_dir / ds_dir.name  # Data Set N / Data Set N
        if inner.is_dir():
            combined.extend(build_file_list(str(inner), split=split))
        else:
            # Also try the directory itself (flat layout)
            combined.extend(build_file_list(str(ds_dir), split=split))

    # Optional per-class capping
    if max_samples_per_class > 0:
        by_label: dict[int, List[Tuple[str, int]]] = {}
        for item in combined:
            by_label.setdefault(item[1], []).append(item)
        capped: List[Tuple[str, int]] = []
        for label_val, items in by_label.items():
            if len(items) > max_samples_per_class:
                random.seed(42)
                items = random.sample(items, max_samples_per_class)
            capped.extend(items)
        combined = capped

    return combined


if __name__ == "__main__":
    # Simple CLI to crop a dataset using facenet-pytorch if available
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source root with images")
    parser.add_argument("--dst", required=True, help="Destination root for crops")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False)
    except Exception:
        mtcnn = None

    items = gather_image_paths(args.src)
    total = len(items)
    saved = 0
    for split, label, path in items:
        out_dir = Path(args.dst) / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(path).name
        ok = detect_and_align(path, str(out_path), size=args.size, mtcnn=mtcnn)
        if ok:
            saved += 1

    print(f"Processed {total} images, saved {saved} crops to {args.dst}")
