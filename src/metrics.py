"""Pixel-level segmentation metrics from accumulated confusion counts."""
from __future__ import annotations

import numpy as np


class SegmentationMeter:
    """Accumulates TP / FP / FN / TN over many images, ignoring 'no data' pixels."""

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None):
        pred = pred.astype(bool)
        target = target.astype(bool)
        if valid is None:
            valid = np.ones_like(target, dtype=bool)
        self.tp += int(np.sum(pred & target & valid))
        self.fp += int(np.sum(pred & ~target & valid))
        self.fn += int(np.sum(~pred & target & valid))
        self.tn += int(np.sum(~pred & ~target & valid))

    def compute(self) -> dict:
        return scores(self.tp, self.fp, self.fn, self.tn)


def scores(tp: int, fp: int, fn: int, tn: int) -> dict:
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),  # = Dice
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def best_threshold(probs: list[np.ndarray], targets: list[np.ndarray], valids: list[np.ndarray],
                   candidates=np.arange(0.20, 0.81, 0.05)) -> tuple[float, dict]:
    """Pick the probability threshold with the highest F1 on a held-out split."""
    best_t, best = 0.5, {"f1": -1.0}
    for t in candidates:
        meter = SegmentationMeter()
        for p, y, v in zip(probs, targets, valids):
            meter.update(p >= t, y, v)
        result = meter.compute()
        if result["f1"] > best["f1"]:
            best_t, best = float(round(t, 2)), result
    return best_t, best
