"""From a building mask to population and urban infrastructure needs.

The model is intentionally simple and transparent; every assumption lives in
``configs/default.yaml`` under ``population`` so it can be adapted to a city:

    footprint area  A   = building pixels x pixel_size_m^2
    living area         = A x residential_share x avg_floors
    population      P   = living area / m2_per_person
    hospitals           = ceil(P / people_per_hospital)
    markets             = ceil(P / people_per_market)
    base stations       = ceil(P / people_per_base_station)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class UrbanReport:
    building_count: int
    footprint_area_m2: float
    built_up_ratio: float
    estimated_population: int
    hospitals_needed: int
    markets_needed: int
    base_stations_needed: int

    def to_dict(self) -> dict:
        return asdict(self)


def clean_mask(mask: np.ndarray, min_area_px: int) -> tuple[np.ndarray, int]:
    """Remove connected components smaller than ``min_area_px``; return mask and building count."""
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    return keep[labels].astype(np.uint8), int(keep.sum())


def _needed(population: int, people_per_unit: float) -> int:
    return int(math.ceil(population / people_per_unit)) if population > 0 else 0


def estimate(mask: np.ndarray, cfg: dict, valid: np.ndarray | None = None) -> UrbanReport:
    """Estimate population and facility needs from a binary building mask."""
    p = cfg["population"]
    pixel_area = p["pixel_size_m"] ** 2
    min_area_px = max(1, int(round(p["min_building_area_m2"] / pixel_area)))
    mask, count = clean_mask(mask, min_area_px)
    area_m2 = float(mask.sum()) * pixel_area
    valid_px = float(valid.sum()) if valid is not None else float(mask.size)
    living_area = area_m2 * p["residential_share"] * p["avg_floors"]
    population = int(living_area / p["m2_per_person"])
    return UrbanReport(
        building_count=count,
        footprint_area_m2=round(area_m2, 1),
        built_up_ratio=round(float(mask.sum()) / max(valid_px, 1.0), 4),
        estimated_population=population,
        hospitals_needed=_needed(population, p["people_per_hospital"]),
        markets_needed=_needed(population, p["people_per_market"]),
        base_stations_needed=_needed(population, p["people_per_base_station"]),
    )
