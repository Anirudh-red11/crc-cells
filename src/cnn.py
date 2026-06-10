"""
cnn.py — Convolutional models (workflow steps D & E)
=====================================================
Contains:
  • SmallCNN          — compact net designed for 27×27 inputs (from scratch)
  • TransferCNN       — ResNet18 backbone, images upsampled to 64×64 (transfer learning)
  • CellImageDataset  — PyTorch Dataset with optional augmentation
  • train_model       — training loop with early stopping, returns history
  • predict_proba     — softmax probabilities for a fitted model

Device priority: Apple MPS  >  CUDA  >  CPU  (auto-detected).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class CellImageDataset(Dataset):
    """
    Loads 27×27 cell PNGs and a label column.

    target : 'isCancerous' or 'cellType'
    augment: training-time flips/rotations/jitter (step E)
    upscale: resize to this size (used by transfer learning, e.g. 64)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str = "isCancerous",
        augment: bool = False,
        upscale: Optional[int] = None,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.paths = df["path"].tolist()
        self.labels = df[target].astype(int).tolist()

        tfm: List = []
        if upscale:
            tfm.append(transforms.Resize((upscale, upscale)))
        if augment:
            tfm += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                       saturation=0.10),
            ]
        tfm += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        self.transform = transforms.Compose(tfm)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def make_loader(ds: Dataset, batch_size=64, shuffle=False, workers=2) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=(workers > 0))


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    """Compact CNN for tiny 27×27 patches (trained from scratch)."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 27→13
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 13→6
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                           # →1×1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TransferCNN(nn.Module):
    """ResNet18 pretrained on ImageNet; expects ~64×64 upscaled inputs."""

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False):
        super().__init__()
        from torchvision import models
        from torchvision.models import ResNet18_Weights
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)


def _epoch(model, loader, criterion, device, optimizer=None):
    from sklearn.metrics import f1_score
    train = optimizer is not None
    model.train(train)
    tot_loss = correct = total = 0
    preds_all, labels_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            bs = y.size(0)
            tot_loss += loss.item() * bs
            p = logits.argmax(1)
            correct += (p == y).sum().item()
            total += bs
            preds_all.append(p.cpu().numpy())
            labels_all.append(y.cpu().numpy())
    f1 = f1_score(np.concatenate(labels_all), np.concatenate(preds_all),
                  average="macro")
    return tot_loss / total, correct / total, f1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    patience: int = 8,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, History]:
    """Train with early stopping on validation macro-F1. Returns best model + history."""
    device = device or get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=0.05,
    )
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist = History()
    best_f1, best_state, wait = -1.0, None, 0
    for ep in range(1, epochs + 1):
        tl, ta, _ = _epoch(model, train_loader, criterion, device, opt)
        vl, va, vf = _epoch(model, val_loader, criterion, device, None)
        sched.step()
        hist.train_loss.append(tl); hist.val_loss.append(vl)
        hist.train_acc.append(ta); hist.val_acc.append(va); hist.val_f1.append(vf)
        if verbose:
            print(f"  ep{ep:02d}  train_loss={tl:.3f} acc={ta:.3f} │ "
                  f"val_loss={vl:.3f} acc={va:.3f} f1={vf:.3f}")
        if vf > best_f1:
            best_f1, best_state, wait = vf, {k: v.cpu().clone()
                                             for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  early stop @ epoch {ep} (best val_f1={best_f1:.3f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader,
                  device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probabilities (N,C), true_labels (N,))."""
    device = device or get_device()
    model = model.to(device).eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss (handles imbalance)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
