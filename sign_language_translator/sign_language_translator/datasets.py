"""
Dataset utilities for Sign Language Translator.

We include a simple ImageFolder-based dataset for images arranged in the
following structure:

    root/
        class_0/
            img1.png
            img2.png
        class_1/
            img3.png
            ...

Additionally, we provide a helper to split a dataset into train/val
folders and a small CSV converter is implemented in scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets


@dataclass
class ImageFolderConfig:
    """Configuration for creating an ImageFolder dataset."""

    root: Path
    image_size: Tuple[int, int] = (224, 224)
    grayscale: bool = False


def build_imagefolder_dataset(config: ImageFolderConfig, transform=None) -> Dataset:
    """Create a torchvision ImageFolder dataset with optional transform."""

    if transform is None:
        transform = _default_transform(config.image_size, grayscale=config.grayscale)
    return datasets.ImageFolder(root=str(config.root), transform=transform)


def _default_transform(image_size: Tuple[int, int], grayscale: bool = False):
    """Basic transform pipeline for training or evaluation.

    We avoid importing albumentations here to keep dependencies light in
    this module; more complex augmentation is defined in transforms.py.
    """

    from torchvision import transforms as T

    transform_list = [
        T.Resize(image_size),
        T.ToTensor(),
    ]
    if grayscale:
        # Convert grayscale to 1 channel tensor
        transform_list.insert(0, T.Grayscale(num_output_channels=1))
    else:
        # Ensure 3-channel RGB
        transform_list.insert(0, T.ConvertImageDtype(torch.float32))
    return T.Compose(transform_list)


class NumpyImageDataset(Dataset):
    """Dataset from in-memory numpy arrays.

    Useful for quick experiments or CSV-based sources already loaded
    into memory. Expects images as (N, H, W) or (N, H, W, C).
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None) -> None:
        assert len(images) == len(labels), "images and labels must have the same length"
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        label = int(self.labels[idx])

        # Convert numpy array to PIL Image for compatibility with torchvision transforms
        if img.ndim == 2:
            pil = Image.fromarray(img.astype(np.uint8), mode="L")
        elif img.ndim == 3:
            if img.shape[2] == 1:
                pil = Image.fromarray(img.squeeze(-1).astype(np.uint8), mode="L")
            elif img.shape[2] == 3:
                pil = Image.fromarray(img.astype(np.uint8), mode="RGB")
            else:
                raise ValueError("Unsupported channel count in image array")
        else:
            raise ValueError("Unsupported image shape; expected 2D or 3D array")

        if self.transform is not None:
            img_tensor = self.transform(pil)
        else:
            from torchvision import transforms as T

            img_tensor = T.ToTensor()(pil)

        return img_tensor, label

