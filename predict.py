"""Run the trained model on new satellite images and estimate population and urban needs.

Usage:
    python predict.py path/to/image.tiff [more images ...] --checkpoint checkpoints/best_model.pt

For each image, writes to outputs/predictions/:
    <name>_mask.png      binary building mask
    <name>_overlay.png   detected buildings over the image
    <name>_report.json   building count, footprint area, population, hospitals, markets, base stations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.dataset import nodata_mask, read_image
from src.inference import error_overlay, load_checkpoint, predict_probabilities, save_rgb
from src.population import estimate
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="image files (.tiff, .tif, .png, .jpg)")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--threshold", type=float, default=None, help="default: value tuned on validation")
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="metres per pixel of your imagery (Massachusetts = 1.0)")
    args = parser.parse_args()

    device = get_device()
    model, cfg, threshold = load_checkpoint(args.checkpoint, device)
    if args.threshold is not None:
        threshold = args.threshold
    if args.pixel_size is not None:
        cfg["population"]["pixel_size_m"] = args.pixel_size
    out_dir = Path(args.output_dir)

    for path in map(Path, args.images):
        img = read_image(path)
        valid = ~nodata_mask(img)
        prob = predict_probabilities(model, img, device, cfg["inference"]["tile_size"], cfg["inference"]["overlap"])
        pred = ((prob >= threshold) & valid).astype(np.uint8)
        report = estimate(pred, cfg, valid).to_dict()
        report.update({"image": path.name, "threshold": threshold, "assumptions": cfg["population"]})
        save_rgb(out_dir / f"{path.stem}_mask.png", np.repeat(pred[..., None] * 255, 3, axis=2))
        save_rgb(out_dir / f"{path.stem}_overlay.png", error_overlay(img, pred))
        with open(out_dir / f"{path.stem}_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"{path.name}: {report['building_count']} buildings, "
              f"{report['footprint_area_m2']:,.0f} m2 footprint, population ~{report['estimated_population']:,} "
              f"-> hospitals {report['hospitals_needed']}, markets {report['markets_needed']}, "
              f"base stations {report['base_stations_needed']}")


if __name__ == "__main__":
    main()
