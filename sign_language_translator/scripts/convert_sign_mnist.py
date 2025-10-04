"""
Convert the Sign Language MNIST CSV dataset to an ImageFolder structure.

Dataset source: Kaggle "Sign Language MNIST" (28x28 grayscale digits but for letters)
 - Training CSV has columns: label, pixel0 ... pixel783
 - Test CSV has similar format (without labels or with depending on source)

Usage:
  python scripts/convert_sign_mnist.py --train_csv path/to/sign_mnist_train.csv \
      --test_csv path/to/sign_mnist_test.csv --out_dir data/sign_mnist

This will create:
  out_dir/
    train/<label>/*.png
    test/<label>/*.png   (if labels exist) or test/unknown/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


LABEL_MAP = {
    # Kaggle Sign Language MNIST uses 24 letters (no J, Z static)
    # Map numeric labels 0-23 to letters A-Y excluding J and Z
    # According to known mapping from dataset descriptions
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F",
    6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "M",
    12: "N", 13: "O", 14: "P", 15: "Q", 16: "R", 17: "S",
    18: "T", 19: "U", 20: "V", 21: "W", 22: "X", 23: "Y",
}


def save_images_from_csv(csv_path: Path, out_dir: Path, split: str) -> None:
    df = pd.read_csv(csv_path)
    has_label = "label" in df.columns
    pixel_cols = [c for c in df.columns if c.startswith("pixel")]

    for idx, row in df.iterrows():
        if has_label:
            label_num = int(row["label"])  # 0-23
            label = LABEL_MAP.get(label_num, str(label_num))
            class_dir = out_dir / split / label
        else:
            class_dir = out_dir / split / "unknown"
        class_dir.mkdir(parents=True, exist_ok=True)

        pixels = row[pixel_cols].to_numpy(dtype=np.uint8)
        img = pixels.reshape(28, 28)
        pil = Image.fromarray(img, mode="L")
        img_name = f"{idx:06d}.png"
        pil.save(class_dir / img_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Sign MNIST CSVs to ImageFolder")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=False)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_images_from_csv(Path(args.train_csv), out_dir, split="train")
    if args.test_csv:
        save_images_from_csv(Path(args.test_csv), out_dir, split="test")

    print(f"Saved images to {out_dir}")


if __name__ == "__main__":
    main()

