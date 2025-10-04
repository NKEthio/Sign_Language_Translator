"""
Example: Quick start with Sign Language Translator

This example demonstrates basic usage of the package, including:
1. Creating a simple model
2. Training on synthetic data
3. Running inference

For real-world usage with the Sign Language MNIST dataset, see the README.
"""

import torch
import numpy as np
from sign_language_translator import (
    build_classifier,
    ModelConfig,
    NumpyImageDataset,
    build_train_transforms,
    build_eval_transforms,
    train_one_epoch,
    evaluate,
    set_seed,
)
from torch.utils.data import DataLoader
from torch import nn


def create_dummy_data(num_samples=100, num_classes=24):
    """Create dummy grayscale image data for testing."""
    # Random 28x28 grayscale images
    images = np.random.randint(0, 255, size=(num_samples, 28, 28), dtype=np.uint8)
    labels = np.random.randint(0, num_classes, size=num_samples)
    return images, labels


def main():
    print("Sign Language Translator - Quick Start Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create dummy training and validation data
    print("\n1. Creating synthetic data...")
    train_images, train_labels = create_dummy_data(num_samples=100, num_classes=24)
    val_images, val_labels = create_dummy_data(num_samples=20, num_classes=24)
    
    # Build transforms
    image_size = (28, 28)  # Keep small for this example
    train_tfms = build_train_transforms(image_size=image_size, grayscale=True)
    val_tfms = build_eval_transforms(image_size=image_size, grayscale=True)
    
    # Create datasets
    train_ds = NumpyImageDataset(train_images, train_labels, transform=train_tfms)
    val_ds = NumpyImageDataset(val_images, val_labels, transform=val_tfms)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    print(f"   Training samples: {len(train_ds)}")
    print(f"   Validation samples: {len(val_ds)}")
    
    # Build a small model
    print("\n2. Building classifier model...")
    config = ModelConfig(
        backbone="resnet18",
        num_classes=24,
        pretrained=True,  # Use ImageNet pretrained weights
        dropout=0.2,
        in_channels=1,  # Grayscale input
    )
    model = build_classifier(config)
    print(f"   Model: {config.backbone}")
    print(f"   Classes: {config.num_classes}")
    print(f"   Input channels: {config.in_channels}")
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Train for a few epochs
    print("\n3. Training model...")
    epochs = 3
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"   Epoch {epoch}/{epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    
    # Simple inference example
    print("\n4. Running inference on a single sample...")
    model.eval()
    test_image = torch.randn(1, 1, 28, 28).to(device)  # Random test image
    with torch.no_grad():
        logits = model(test_image)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    print(f"   Predicted class: {pred_class}")
    print(f"   Confidence: {confidence:.4f}")
    
    print("\n" + "=" * 50)
    print("✓ Example completed successfully!")
    print("\nFor real-world usage:")
    print("  1. Install: pip install -e .")
    print("  2. Convert data: slt-convert --train_csv ... --out_dir data/")
    print("  3. Train: slt-train --data_dir data/train --val_dir data/val")
    print("  4. Infer: slt-infer --model artifacts/model_best.pt --image img.png")


if __name__ == "__main__":
    main()
