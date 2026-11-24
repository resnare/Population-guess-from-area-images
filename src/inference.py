"""Sliding-window inference on full-resolution tiles and result visualisation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import to_tensor
from .model import build_model


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


@torch.no_grad()
def predict_probabilities(model, img: np.ndarray, device, tile_size=512, overlap=64) -> np.ndarray:
    """Probability map (H, W) for a whole image, averaging overlapping tiles."""
    model.eval()
    h, w = img.shape[:2]
    stride = tile_size - overlap
    prob = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    for y in _starts(h, tile_size, stride):
        for x in _starts(w, tile_size, stride):
            tile = img[y:y + tile_size, x:x + tile_size]
            logits = model(to_tensor(tile).unsqueeze(0).to(device))
            p = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
            prob[y:y + p.shape[0], x:x + p.shape[1]] += p
            weight[y:y + p.shape[0], x:x + p.shape[1]] += 1
    return prob / np.maximum(weight, 1)


def load_checkpoint(path: str | Path, device):
    """Load a model saved by train.py; returns (model, config, threshold)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, float(ckpt.get("threshold", cfg["inference"]["threshold"]))


def error_overlay(img: np.ndarray, pred: np.ndarray, target: np.ndarray | None = None, alpha=0.45) -> np.ndarray:
    """Colour overlay. With a target: green = correct, red = false positive, blue = missed.
    Without a target: predicted buildings in yellow."""
    out = img.astype(np.float32).copy()
    pred = pred.astype(bool)
    if target is None:
        layers = [(pred, (255, 215, 0))]
    else:
        target = target.astype(bool)
        layers = [(pred & target, (0, 200, 0)), (pred & ~target, (230, 30, 30)), (~pred & target, (30, 90, 255))]
    for region, color in layers:
        out[region] = (1 - alpha) * out[region] + alpha * np.array(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def save_rgb(path: str | Path, img: np.ndarray):
    """Write an RGB image (Unicode-safe on Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(str(path))
