"""Sign Language Translator: PyTorch-based ASL alphabet classifier."""

__version__ = "0.1.0"

from sign_language_translator.datasets import (
    ImageFolderConfig,
    NumpyImageDataset,
    build_imagefolder_dataset,
)
from sign_language_translator.models import (
    BackboneName,
    ModelConfig,
    RepeatTo3Channels,
    build_classifier,
)
from sign_language_translator.transforms import (
    build_eval_transforms,
    build_train_transforms,
)
from sign_language_translator.utils import (
    Checkpoint,
    accuracy_from_logits,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)

__all__ = [
    "__version__",
    # datasets
    "ImageFolderConfig",
    "NumpyImageDataset",
    "build_imagefolder_dataset",
    # models
    "BackboneName",
    "ModelConfig",
    "RepeatTo3Channels",
    "build_classifier",
    # transforms
    "build_eval_transforms",
    "build_train_transforms",
    # utils
    "Checkpoint",
    "accuracy_from_logits",
    "evaluate",
    "load_checkpoint",
    "save_checkpoint",
    "set_seed",
    "train_one_epoch",
]
