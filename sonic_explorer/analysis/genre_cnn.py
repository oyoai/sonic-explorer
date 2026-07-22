"""Small CNN genre classifier trained on mel-spectrograms -- a real trained
baseline alongside the hand-crafted/pretrained-embedding facets (CLAP,
chroma), which are otherwise the only "does genre separate from audio"
signal anywhere in this project. The parallel-comparison story spec section
9 calls for: does a model actually TRAINED on this library's genre labels
separate genre better or worse than facets that were never trained on it at
all (CLAP is pretrained-and-frozen, chroma is hand-crafted)?

Deliberately small (three conv blocks) -- this is a baseline comparison
point, not a research architecture; small enough to train on CPU in minutes
on ~1000 spectrograms.

torch is a real, module-level runtime dependency here (unlike pipeline/
vocal_presence.py's lazily-imported tagger, an nn.Module can't be defined
without it) -- this module is never imported by the deployed app or the
main test suite (see analysis/mel_features.py for the torch-free half of
this pipeline), only by scripts/train_genre_cnn.py and its own
torch-gated tests."""

from dataclasses import dataclass

import torch
import torch.nn as nn


class SmallGenreCNN(nn.Module):
    """(batch, 1, n_mels, n_frames) -> (batch, n_classes) logits. Three conv
    blocks (16 -> 32 -> 64 channels), global average pool, linear head."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(features)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


def train_one_epoch(model: nn.Module, loader, optimizer, device: str) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> tuple[float, float]:
    """Returns (mean_loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item())
        n_batches += 1
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += y.shape[0]
    accuracy = correct / total if total > 0 else 0.0
    return total_loss / max(1, n_batches), accuracy
