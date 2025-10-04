"""
Augmentation and preprocessing transforms for Sign Language datasets.

We provide simple training and evaluation transforms using torchvision
and optional albumentations for stronger augmentation.
"""

from __future__ import annotations

from typing import Tuple

from torchvision import transforms as T


def build_train_transforms(image_size: Tuple[int, int] = (224, 224), grayscale: bool = False):
    """Return torchvision transforms for training."""

    transform_list = [
        T.Resize(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2) if not grayscale else T.Lambda(lambda x: x),
        T.ToTensor(),
    ]
    if grayscale:
        transform_list.insert(0, T.Grayscale(num_output_channels=1))
    return T.Compose([t for t in transform_list if not isinstance(t, T.Lambda) or grayscale is True])


def build_eval_transforms(image_size: Tuple[int, int] = (224, 224), grayscale: bool = False):
    """Return torchvision transforms for evaluation."""

    transform_list = [
        T.Resize(image_size),
        T.ToTensor(),
    ]
    if grayscale:
        transform_list.insert(0, T.Grayscale(num_output_channels=1))
    return T.Compose(transform_list)

