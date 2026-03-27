"""
PyTorch model definitions.

Provides a simple backbone selector and a binary classification head.
"""
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


def build_model(backbone: str = "efficientnet_b0", pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    """Return a PyTorch model with a chosen backbone and a single-node output (logit).

    Supported backbones: 'efficientnet_b0', 'mobilenet_v3_large', 'resnet50'
    """
    backbone = backbone.lower()
    if backbone == "efficientnet_b0":
        net = models.efficientnet_b0(pretrained=pretrained)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, num_classes))
    elif backbone == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(pretrained=pretrained)
        in_features = net.classifier[3].in_features
        net.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, num_classes))
    elif backbone == "resnet50":
        net = models.resnet50(pretrained=pretrained)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return net


class BinaryWrapper(nn.Module):
    """Wraps a backbone to return a single logit and optionally apply sigmoid outside."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if out.ndim == 2 and out.size(1) == 1:
            return out.squeeze(1)
        return out


def get_model(backbone: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    base = build_model(backbone=backbone, pretrained=pretrained, num_classes=1)
    return BinaryWrapper(base)
