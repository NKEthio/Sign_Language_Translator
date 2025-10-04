"""
Utility functions: metrics, training helpers, checkpointing, and seeding.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class Checkpoint:
    epoch: int
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, torch.Tensor]
    scheduler_state: Dict[str, torch.Tensor]
    best_val_acc: float


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / max(total, 1)


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": checkpoint.epoch,
            "model_state": checkpoint.model_state,
            "optimizer_state": checkpoint.optimizer_state,
            "scheduler_state": checkpoint.scheduler_state,
            "best_val_acc": checkpoint.best_val_acc,
        },
        path,
    )


def load_checkpoint(path: Path) -> Checkpoint:
    data = torch.load(path, map_location="cpu")
    return Checkpoint(
        epoch=int(data.get("epoch", 0)),
        model_state=data["model_state"],
        optimizer_state=data.get("optimizer_state", {}),
        scheduler_state=data.get("scheduler_state", {}),
        best_val_acc=float(data.get("best_val_acc", 0.0)),
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(outputs.detach(), targets) * batch_size
        total += batch_size

    return running_loss / max(total, 1), running_acc / max(total, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_from_logits(outputs, targets) * batch_size
            total += batch_size

    return running_loss / max(total, 1), running_acc / max(total, 1)

