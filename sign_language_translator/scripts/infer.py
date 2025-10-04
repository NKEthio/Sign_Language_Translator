"""
Inference script for Sign Language Translator.

Supports image file input and webcam streaming.

Examples:
  - Single image: python scripts/infer.py --model artifacts/model_best.pt --image path/to/img.png
  - Webcam:       python scripts/infer.py --model artifacts/model_best.pt --webcam
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import torch
from torchvision import transforms as T
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained ASL model")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--webcam", action="store_true", help="Use webcam stream for inference")
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def load_model(model_path: Path):
    data = torch.load(model_path, map_location="cpu")
    model_state = data["model_state"]
    classes: List[str] = data.get("classes") or []
    backbone = data.get("backbone", "resnet18")
    grayscale = bool(data.get("grayscale", False))
    image_size = tuple(data.get("image_size", (224, 224)))

    from sign_language_translator.models import ModelConfig, build_classifier

    model = build_classifier(ModelConfig(
        backbone=backbone, num_classes=len(classes) or 26, pretrained=False, in_channels=1 if grayscale else 3
    ))
    model.load_state_dict(model_state)
    model.eval()
    return model, classes, grayscale, image_size


def build_transform(image_size, grayscale: bool):
    tfms = [T.Resize(image_size), T.ToTensor()]
    if grayscale:
        tfms.insert(0, T.Grayscale(num_output_channels=1))
    return T.Compose(tfms)


def predict_image(model, image_path: Path, classes: List[str], grayscale: bool, image_size):
    img = Image.open(image_path).convert("RGB")
    if grayscale:
        img = img.convert("L")
    transform = build_transform(image_size, grayscale)
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    return probs


def run_webcam(model, classes: List[str], grayscale: bool, image_size, topk: int):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    transform = build_transform(image_size, grayscale)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        if grayscale:
            pil = pil.convert("L")
        x = transform(pil).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        topk_probs, topk_idx = torch.topk(probs, k=min(topk, probs.numel()))
        labels = [classes[i] if classes else str(i) for i in topk_idx.tolist()]
        text = ", ".join([f"{l}:{p:.2f}" for l, p in zip(labels, topk_probs.tolist())])
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("ASL Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    model_path = Path(args.model)
    model, classes, grayscale, image_size = load_model(model_path)

    if args.webcam:
        run_webcam(model, classes, grayscale, image_size, args.topk)
        return

    if args.image:
        probs = predict_image(model, Path(args.image), classes, grayscale, image_size)
        topk_probs, topk_idx = torch.topk(probs, k=min(args.topk, probs.numel()))
        labels = [classes[i] if classes else str(i) for i in topk_idx.tolist()]
        print("Top predictions:")
        for l, p in zip(labels, topk_probs.tolist()):
            print(f"  {l}: {p:.3f}")
    else:
        print("Provide --image or --webcam")


if __name__ == "__main__":
    main()

