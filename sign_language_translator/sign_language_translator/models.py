"""
Model definitions for Sign Language Translator.

We provide a simple factory function to create image classifiers for
static ASL alphabet recognition using torchvision backbones. The models
are configured for single-label classification over N classes.

Explanatory comments are included for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn
from torchvision import models


BackboneName = Literal[
    "resnet18",
    "resnet34",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
]


@dataclass
class ModelConfig:
    """Configuration for building a classifier model.

    Attributes
    ----------
    backbone: str
        Which torchvision backbone to use.
    num_classes: int
        Number of target classes.
    pretrained: bool
        If True, load ImageNet weights. Helpful with small datasets.
    dropout: float | None
        Optional dropout probability for the classification head.
    in_channels: int
        Number of input channels (1 for grayscale, 3 for RGB). If 1, we
        repeat to 3 channels internally via a conv stem.
    """

    backbone: BackboneName = "resnet18"
    num_classes: int = 26
    pretrained: bool = True
    dropout: Optional[float] = 0.2
    in_channels: int = 3


class RepeatTo3Channels(nn.Module):
    """Optional stem to accept single-channel inputs.

    If the dataset is grayscale (e.g., Sign Language MNIST), some
    ImageNet-pretrained backbones expect 3-channel input. This module
    converts 1-channel input to 3 channels with a 1x1 convolution.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        if in_channels == 3:
            self.conv = None
        else:
            self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.conv is None:
            return x
        return self.conv(x)


def _replace_classifier(module: nn.Module, in_features: int, num_classes: int, dropout: Optional[float]) -> nn.Module:
    """Create a simple classification head with optional dropout."""
    layers: list[nn.Module] = []
    if dropout and dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


def build_classifier(config: ModelConfig) -> nn.Module:
    """Build a classifier model based on the provided configuration.

    Parameters
    ----------
    config: ModelConfig
        Model configuration.

    Returns
    -------
    nn.Module
        A PyTorch model ready for training/inference.
    """

    # Some torchvision APIs changed around weights; handle simply.
    weights = None
    if config.pretrained:
        # Use default weights if available for each architecture.
        weights = "DEFAULT"

    backbone_name = config.backbone

    if backbone_name == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = _replace_classifier(model.fc, in_features, config.num_classes, config.dropout)
        stem = RepeatTo3Channels(config.in_channels)
        if config.in_channels != 3:
            # Prepend stem to resnet by overriding forward
            original_forward = model.forward

            def forward(x: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                x = stem(x)
                return original_forward(x)

            model.forward = forward  # type: ignore[assignment]
        return model

    if backbone_name == "resnet34":
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = _replace_classifier(model.fc, in_features, config.num_classes, config.dropout)
        stem = RepeatTo3Channels(config.in_channels)
        if config.in_channels != 3:
            original_forward = model.forward

            def forward(x: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                x = stem(x)
                return original_forward(x)

            model.forward = forward  # type: ignore[assignment]
        return model

    if backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = _replace_classifier(model.classifier[-1], in_features, config.num_classes, config.dropout)
        stem = RepeatTo3Channels(config.in_channels)
        if config.in_channels != 3:
            original_forward = model.forward

            def forward(x: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                x = stem(x)
                return original_forward(x)

            model.forward = forward  # type: ignore[assignment]
        return model

    if backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = _replace_classifier(model.classifier[-1], in_features, config.num_classes, config.dropout)
        stem = RepeatTo3Channels(config.in_channels)
        if config.in_channels != 3:
            original_forward = model.forward

            def forward(x: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                x = stem(x)
                return original_forward(x)

            model.forward = forward  # type: ignore[assignment]
        return model

    if backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = _replace_classifier(model.classifier[-1], in_features, config.num_classes, config.dropout)
        stem = RepeatTo3Channels(config.in_channels)
        if config.in_channels != 3:
            original_forward = model.forward

            def forward(x: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                x = stem(x)
                return original_forward(x)

            model.forward = forward  # type: ignore[assignment]
        return model

    raise ValueError(f"Unsupported backbone: {backbone_name}")

