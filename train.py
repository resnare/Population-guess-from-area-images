"""Train the U-Net building segmentation model.

Usage:
    python train.py
    python train.py --set train.epochs=60 train.batch_size=16 data.num_workers=4
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import RandomPatchDataset, list_pairs, nodata_mask, read_image, read_mask, worker_init_fn
from src.inference import predict_probabilities
from src.losses import BCEDiceLoss
from src.metrics import SegmentationMeter, best_threshold
from src.model import build_model
from src.utils import get_device, set_seed


def evaluate_split(model, images, masks, device, cfg, threshold=0.5, return_probs=False):
    """Full-resolution metrics on a list of pre-loaded images."""
    meter, probs, valids = SegmentationMeter(), [], []
    for img, mask in zip(images, masks):
        prob = predict_probabilities(model, img, device, cfg["inference"]["tile_size"], cfg["inference"]["overlap"])
        valid = ~nodata_mask(img)
        meter.update(prob >= threshold, mask, valid)
        if return_probs:
            probs.append(prob)
            valids.append(valid)
    return (meter.compute(), probs, valids) if return_probs else meter.compute()


def plot_history(history: list[dict], path: Path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_title("Training loss"); axes[0].grid(alpha=0.3)
    for key, label in (("val_iou", "validation IoU"), ("val_f1", "validation F1")):
        axes[1].plot(epochs, [h[key] for h in history], label=label)
    axes[1].set_xlabel("epoch"); axes[1].set_title("Validation metrics (threshold 0.5)")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=60")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    d, t = cfg["data"], cfg["train"]
    set_seed(t["seed"])
    device = get_device()
    ckpt_dir, out_dir = Path(t["checkpoint_dir"]), Path(t["output_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(d["root"])
    train_pairs = list_pairs(root / "train", d["image_suffix"], d["mask_suffix"])
    val_pairs = list_pairs(root / "val", d["image_suffix"], d["mask_suffix"])
    print(f"Device: {device} | train images: {len(train_pairs)} | val images: {len(val_pairs)}")

    train_ds = RandomPatchDataset(train_pairs, d["patch_size"], d["patches_per_image"], d["max_nodata_fraction"],
                                  d["cache_in_memory"], seed=t["seed"], train=True)
    loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=d["num_workers"],
                        pin_memory=device.type == "cuda", drop_last=True,
                        worker_init_fn=worker_init_fn if d["num_workers"] > 0 else None,
                        persistent_workers=d["num_workers"] > 0)
    val_images = [read_image(p) for p, _ in val_pairs]
    val_masks = [read_mask(m) for _, m in val_pairs]

    model = build_model(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    criterion = BCEDiceLoss(t["pos_weight"], t["dice_weight"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"], eta_min=t["lr"] * 0.02)
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_iou, bad_epochs, history = -1.0, 0, []
    best_path = ckpt_dir / "best_model.pt"
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        start, loss_sum = time.time(), 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
        scheduler.step()

        val = evaluate_split(model, val_images, val_masks, device, cfg)
        row = {"epoch": epoch, "train_loss": loss_sum / len(loader), "val_iou": val["iou"], "val_f1": val["f1"],
               "val_precision": val["precision"], "val_recall": val["recall"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - start}
        history.append(row)
        print(f"Epoch {epoch:3d}/{t['epochs']} | loss {row['train_loss']:.4f} | val IoU {val['iou']:.4f} "
              f"| val F1 {val['f1']:.4f} | {row['seconds']:.0f}s")

        if val["iou"] > best_iou:
            best_iou, bad_epochs = val["iou"], 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch,
                        "val_metrics": val, "threshold": 0.5}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= t["early_stopping_patience"]:
                print(f"Early stopping: no validation IoU improvement for {bad_epochs} epochs.")
                break

    with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, out_dir / "training_curves.png")

    # Tune the decision threshold on the validation set with the best weights.
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, probs, valids = evaluate_split(model, val_images, val_masks, device, cfg, return_probs=True)
    threshold, tuned = best_threshold(probs, val_masks, valids)
    ckpt["threshold"] = threshold
    ckpt["val_metrics_tuned"] = tuned
    torch.save(ckpt, best_path)
    summary = {"best_epoch": ckpt["epoch"], "threshold": threshold, "val_metrics_at_0.5": ckpt["val_metrics"],
               "val_metrics_tuned": tuned}
    with open(out_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Best epoch {ckpt['epoch']} | tuned threshold {threshold} | val IoU {tuned['iou']:.4f} "
          f"| val F1 {tuned['f1']:.4f}\nSaved: {best_path}")


if __name__ == "__main__":
    main()
