"""
Sign Language Translator package.

Convenience exports for model building.
"""

from .models import ModelConfig, build_classifier

__all__ = [
    "ModelConfig",
    "build_classifier",
]
