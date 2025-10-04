"""
Export a trained ASL model to ONNX and run a quick inference check.

Usage:
  python scripts/export_onnx.py --model artifacts/model_best.pt --out artifacts/model.onnx --opset 12 --dynamic_batch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ASL model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--out", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic_batch", action="store_true", help="Export with dynamic batch dimension")
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


def export_to_onnx(model: torch.nn.Module, out_path: Path, in_channels: int, image_size, opset: int, dynamic_batch: bool):
    dummy = torch.randn(1, in_channels, image_size[0], image_size[1], dtype=torch.float32)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )


def quick_check(onnx_path: Path, batch_size: int, in_channels: int, image_size):
    try:
        import onnxruntime as ort
    except Exception as e:
        print("onnxruntime not available; skipping runtime check.")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]) 
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    x = np.random.randn(batch_size, in_channels, image_size[0], image_size[1]).astype(np.float32)
    y = sess.run([output_name], {input_name: x})
    print(f"ONNX inference ok. Output shape: {np.array(y[0]).shape}")


def main() -> None:
    args = parse_args()
    model, classes, grayscale, image_size = load_model(Path(args.model))
    in_channels = 1 if grayscale else 3

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(model, out_path, in_channels, image_size, args.opset, args.dynamic_batch)
    print(f"Exported ONNX model to {out_path}")

    quick_check(out_path, batch_size=2, in_channels=in_channels, image_size=image_size)


if __name__ == "__main__":
    main()
