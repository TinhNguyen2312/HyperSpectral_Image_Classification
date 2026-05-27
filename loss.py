from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if isinstance(weight, torch.Tensor) else None)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def build_criterion(config=None):
    """Build a classification criterion from a DCT-IML-style config."""
    config = config or {}
    name = str(config.get("name", "cross_entropy")).lower()

    if name in {"ce", "cross_entropy", "crossentropyloss"}:
        weight = config.get("weight")
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)

    if name == "focal":
        weight = config.get("weight")
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        return FocalLoss(gamma=float(config.get("gamma", 2.0)), weight=weight)

    raise ValueError(f"Unsupported loss: {name}")


def compute_loss(criterion, outputs, targets) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return total loss and a DCT-IML-like loss dictionary."""
    total = criterion(outputs, targets)
    if isinstance(criterion, FocalLoss):
        main_key = "focal"
    else:
        main_key = "ce"
    return total, {main_key: total.detach(), "total": total.detach()}
