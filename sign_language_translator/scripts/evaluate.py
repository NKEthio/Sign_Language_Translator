"""
Evaluate a trained ASL model on an ImageFolder dataset.

Computes accuracy and prints a confusion matrix and classification report.

Usage:
  python scripts/evaluate.py --model artifacts/model_best.pt --data_dir data/sign_mnist/test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from sklearn.metrics import confusion_matrix, classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASL model")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ImageFolder root")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_cm", type=str, default=None, help="Optional path to save confusion matrix PNG")
    parser.add_argument("--normalize_cm", action="store_true", help="Normalize confusion matrix rows")
    return parser.parse_args()


def load_model(model_path: Path):
    data = torch.load(model_path, map_location="cpu")
    classes = data.get("classes") or []
    backbone = data.get("backbone", "resnet18")
    grayscale = bool(data.get("grayscale", False))
    image_size = tuple(data.get("image_size", (224, 224)))

    from sign_language_translator.models import ModelConfig, build_classifier

    model = build_classifier(ModelConfig(
        backbone=backbone, num_classes=len(classes) or 26, pretrained=False, in_channels=1 if grayscale else 3
    ))
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, classes, grayscale, image_size


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, grayscale, image_size = load_model(Path(args.model))
    model.to(device)

    tfms = [T.Resize(image_size), T.ToTensor()]
    if grayscale:
        tfms.insert(0, T.Grayscale(num_output_channels=1))
    ds = datasets.ImageFolder(args.data_dir, transform=T.Compose(tfms))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = (y_true == y_pred).mean()
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    if args.normalize_cm:
        with np.errstate(all='ignore'):
            cm = (cm.T / cm.sum(axis=1)).T
            cm = np.nan_to_num(cm)

    print("Confusion matrix:")
    print(cm)

    print("Classification report:")
    target_names = ds.classes if hasattr(ds, 'classes') else [str(i) for i in range(cm.shape[0])]
    print(classification_report(y_true, y_pred, target_names=target_names))

    if args.save_cm:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        out_path = Path(args.save_cm)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
