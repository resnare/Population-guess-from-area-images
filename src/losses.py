"""Loss: weighted binary cross-entropy + soft Dice."""
from __future__ import annotations

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Weighted BCE handles the pixel class imbalance (few building pixels);
    Dice directly optimises overlap, which is what IoU / F1 measure."""

    def __init__(self, pos_weight: float = 3.0, dice_weight: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dims)
        dice = 1 - (2 * inter + self.eps) / (prob.sum(dims) + target.sum(dims) + self.eps)
        return bce + self.dice_weight * dice.mean()
