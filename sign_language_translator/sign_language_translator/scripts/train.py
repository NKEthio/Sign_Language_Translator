"""
Training script for Sign Language Translator.

Example usage:
    python scripts/train.py \
        --data_dir data/asl_alphabet/train \
        --val_dir data/asl_alphabet/val \
        --backbone resnet18 --epochs 10 --batch_size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from sign_language_translator.models import ModelConfig, build_classifier
from sign_language_translator.transforms import build_eval_transforms, build_train_transforms
from sign_language_translator.datasets import ImageFolderConfig, build_imagefolder_dataset
from sign_language_translator.utils import (
    Checkpoint,
    evaluate,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASL image classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training ImageFolder root")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation ImageFolder root")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=[
        "resnet18", "resnet34", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"
    ])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--grayscale", action="store_true", help="Treat inputs as grayscale (1-channel)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = (args.image_size, args.image_size)

    train_tfms = build_train_transforms(image_size=image_size, grayscale=bool(args.grayscale))
    val_tfms = build_eval_transforms(image_size=image_size, grayscale=bool(args.grayscale))

    train_ds = build_imagefolder_dataset(
        ImageFolderConfig(root=Path(args.data_dir), image_size=image_size, grayscale=bool(args.grayscale)),
        transform=train_tfms,
    )
    val_ds = build_imagefolder_dataset(
        ImageFolderConfig(root=Path(args.val_dir), image_size=image_size, grayscale=bool(args.grayscale)),
        transform=val_tfms,
    )

    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_classifier(ModelConfig(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=True,
        dropout=0.2,
        in_channels=1 if args.grayscale else 3,
    ))

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        # Save latest checkpoint
        ckpt_path = output_dir / "checkpoint_latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "classes": train_ds.classes,
            "backbone": args.backbone,
            "grayscale": bool(args.grayscale),
            "image_size": image_size,
        }, ckpt_path)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = output_dir / "model_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "backbone": args.backbone,
                "grayscale": bool(args.grayscale),
                "image_size": image_size,
            }, best_path)
            print(f"Saved new best model to {best_path}")


if __name__ == "__main__":
    main()

