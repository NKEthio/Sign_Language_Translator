# Sign Language Translator (ASL Alphabet)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clear, well-commented PyTorch baseline for recognizing static American Sign Language (ASL) alphabet hand signs.

## Features

- 📊 **Data conversion** from Kaggle Sign Language MNIST CSVs to ImageFolder layout
- 🏋️ **Training pipeline** with pretrained backbones (ResNet, MobileNet, EfficientNet)
- 🎯 **Inference support** for single images and real-time webcam streams
- 📝 **Simple, readable code** with explanatory comments
- 🚀 **Easy installation** with pip and command-line tools

Perfect for beginners learning deep learning and computer vision!

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Notes and Tips](#notes-and-tips)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/NKEthio/Sign_Language_Translator.git
cd Sign_Language_Translator
```

2. Create a Python 3.10+ environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:

```bash
pip install -e .
```

This will install the `sign-language-translator` package along with all dependencies and command-line tools (`slt-train`, `slt-infer`, `slt-convert`).

Note: For GPU training, ensure you have the CUDA-enabled PyTorch build. The requirements will install the latest PyTorch with CUDA support. If you need a specific CUDA version, install PyTorch separately before running `pip install -e .`.

## Data Preparation

### 2) Get Data

- Download Kaggle “Sign Language MNIST” CSVs (`sign_mnist_train.csv`, `sign_mnist_test.csv`).
  - These are 28x28 grayscale images for letters (24 classes, excluding J and Z which are dynamic).
- Convert CSVs to images in an ImageFolder structure:

```bash
slt-convert \
  --train_csv /path/to/sign_mnist_train.csv \
  --test_csv /path/to/sign_mnist_test.csv \
  --out_dir data/sign_mnist
```

This produces:

```
data/sign_mnist/
  train/<LETTER>/*.png
  test/<LETTER>/*.png  # or test/unknown/*.png if labels missing
```

Optionally, create a small validation split by moving a few images from each `train/<LETTER>` folder into a `val/<LETTER>` folder.

## Training

### 3) Train

Train a classifier using a pretrained backbone. For Sign MNIST, pass `--grayscale` since images are 1-channel.

```bash
slt-train \
  --data_dir data/sign_mnist/train \
  --val_dir data/sign_mnist/test \
  --backbone resnet18 \
  --epochs 10 \
  --batch_size 64 \
  --grayscale \
  --output_dir artifacts
```

Artifacts saved to `artifacts/` include the latest checkpoint and the best-performing model `model_best.pt` with class names embedded.

Key arguments:
- `--backbone`: `resnet18`, `resnet34`, `mobilenet_v3_small`, `mobilenet_v3_large`, `efficientnet_b0`
- `--image_size`: input resolution (default 224)
- `--grayscale`: expect grayscale inputs (Sign MNIST)

## Inference

### 4) Inference

- Single image:

```bash
slt-infer \
  --model artifacts/model_best.pt \
  --image path/to/hand.png
```

- Webcam (press `q` to quit):

```bash
slt-infer \
  --model artifacts/model_best.pt \
  --webcam
```

The model file stores the class list, chosen backbone, and image size for simpler reuse.

### Project Structure

```
Sign_Language_Translator/
├── sign_language_translator/
│   └── sign_language_translator/
│       ├── __init__.py       # Package exports
│       ├── datasets.py       # ImageFolder/Numpy datasets
│       ├── transforms.py     # Train/eval transforms
│       ├── models.py         # Model factory with pretrained backbones
│       ├── utils.py          # Training loop helpers and checkpoint I/O
│       └── scripts/
│           ├── convert_sign_mnist.py  # CSV -> ImageFolder converter
│           ├── train.py               # Training script (CLI)
│           └── infer.py               # Inference for images and webcam
├── pyproject.toml       # Package configuration
├── requirements.txt     # Dependencies
├── LICENSE              # MIT License
├── .gitignore          # Git ignore rules
├── README.md           # This file
├── data/               # (you create) datasets live here
└── artifacts/          # saved checkpoints and best model
```

After installation (`pip install -e .`), you can use the following commands from anywhere:
- `slt-train` - Train a model
- `slt-infer` - Run inference
- `slt-convert` - Convert CSV datasets to images

## Notes and Tips

- **Dataset scope**: Sign MNIST covers 24 static letters (excluding J and Z which require motion). Real-world ASL includes dynamic gestures and word-level signs.
- **Extending to video**: For temporal gestures, consider 3D CNNs, temporal segment networks (TSN/TSM), or transformer-based models over frame features.
- **Custom datasets**: For your own photos, organize them in ImageFolder format and omit `--grayscale` if using RGB images.
- **Training stability**: If training is unstable, try a smaller learning rate (e.g., `--lr 3e-4`) or train for more epochs.
- **Model selection**: MobileNet and EfficientNet variants are faster for real-time webcam inference.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up the development environment
- Code style and formatting
- Adding new features (backbones, datasets, augmentations)
- Submitting pull requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: Sign Language MNIST from Kaggle
- **Framework**: PyTorch and torchvision
- **Pretrained models**: ImageNet-pretrained weights from torchvision

---

**Educational Purpose**: This project is designed as a learning resource for computer vision and deep learning. For production ASL recognition systems, consider more sophisticated architectures and larger, more diverse datasets.

