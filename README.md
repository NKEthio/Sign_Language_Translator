### Sign Language Translator (ASL Alphabet) — PyTorch

This repository provides a clear, well-commented baseline for recognizing static American Sign Language (ASL) alphabet hand signs using PyTorch. It includes:

- Data conversion from the Kaggle Sign Language MNIST CSVs to an ImageFolder layout
- A training pipeline with pretrained backbones (ResNet, MobileNet, EfficientNet)
- Inference for single images and real-time webcam streams
- Simple, easy-to-read code with explanatory comments

If you are new to deep learning, follow the steps below end-to-end.

### 1) Setup

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

### Notes and Tips

- Sign MNIST covers 24 static letters; real-world ASL includes dynamic motions (J, Z) and word-level signs. Extending to video requires a temporal model (e.g., 3D CNN, TSN, TSM, or transformer over frame features).
- For your own photo dataset, organize images as `ImageFolder` and omit `--grayscale` if using RGB.
- If training is unstable, try a smaller learning rate (`--lr 3e-4`) or more epochs.
- MobileNet and EfficientNet are good for faster webcam inference.

### License

This starter is provided as-is for educational purposes. Ensure dataset use complies with its original license.
