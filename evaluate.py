"""Evaluate a trained model on the test set.

Writes to outputs/:
    test_metrics.json         pixel IoU / F1 / precision / recall (threshold tuned on validation)
    test_per_image.csv        per-image metrics and population from predicted vs. true masks
    figures/<image>.png       image | ground truth | prediction | error overlay

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import list_pairs, nodata_mask, read_image, read_mask
from src.inference import error_overlay, load_checkpoint, predict_probabilities
from src.metrics import SegmentationMeter, scores
from src.population import estimate
from src.utils import get_device


def save_figure(path: Path, img, mask, pred, title: str):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = [(img, "Satellite image"), (mask, "Ground truth"), (pred, "Prediction"),
              (error_overlay(img, pred, mask), "Green: correct  Red: false positive  Blue: missed")]
    for ax, (data, name) in zip(axes, panels):
        ax.imshow(data, cmap="gray" if data.ndim == 2 else None)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-root", default=None, help="overrides data.root from the checkpoint config")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    device = get_device()
    model, cfg, threshold = load_checkpoint(args.checkpoint, device)
    d = cfg["data"]
    pairs = list_pairs(Path(args.data_root or d["root"]) / args.split, d["image_suffix"], d["mask_suffix"])
    out_dir = Path(args.output_dir)
    print(f"Device: {device} | {args.split} images: {len(pairs)} | threshold: {threshold}")

    meter, rows = SegmentationMeter(), []
    for img_path, mask_path in pairs:
        img, mask = read_image(img_path), read_mask(mask_path)
        valid = ~nodata_mask(img)
        prob = predict_probabilities(model, img, device, cfg["inference"]["tile_size"], cfg["inference"]["overlap"])
        pred = (prob >= threshold).astype(np.uint8)
        meter.update(pred, mask, valid)
        single = SegmentationMeter()
        single.update(pred, mask, valid)
        m = single.compute()
        pop_pred = estimate(pred * valid, cfg, valid)
        pop_true = estimate(mask * valid, cfg, valid)
        rows.append({
            "image": img_path.stem, **{k: round(v, 4) for k, v in m.items()},
            "population_from_prediction": pop_pred.estimated_population,
            "population_from_ground_truth": pop_true.estimated_population,
            "buildings_predicted": pop_pred.building_count,
            "buildings_ground_truth": pop_true.building_count,
        })
        save_figure(out_dir / "figures" / f"{img_path.stem}.png", img, mask, pred,
                    f"{img_path.stem}  |  IoU {m['iou']:.3f}  |  F1 {m['f1']:.3f}  |  "
                    f"population: predicted {pop_pred.estimated_population:,} vs. ground truth "
                    f"{pop_true.estimated_population:,}")
        print(f"{img_path.stem}: IoU {m['iou']:.4f} F1 {m['f1']:.4f}")

    overall = meter.compute()
    pred_pop = np.array([r["population_from_prediction"] for r in rows], dtype=float)
    true_pop = np.array([r["population_from_ground_truth"] for r in rows], dtype=float)
    ape = np.abs(pred_pop - true_pop) / np.maximum(true_pop, 1)
    result = {
        "split": args.split, "images": len(rows), "threshold": threshold,
        "pixel_metrics": {k: round(v, 4) for k, v in overall.items()},
        "population_mape_vs_ground_truth_masks": round(float(ape.mean()), 4),
        "note": "Population error compares estimates from predicted masks with estimates from the true "
                "building masks under the same assumptions; it measures segmentation-driven error only.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / f"{args.split}_per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
