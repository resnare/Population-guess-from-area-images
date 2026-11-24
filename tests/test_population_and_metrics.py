import numpy as np

from src.config import load_config
from src.metrics import SegmentationMeter, best_threshold
from src.population import clean_mask, estimate


def test_clean_mask_removes_small_components():
    mask = np.zeros((50, 50), np.uint8)
    mask[5:15, 5:15] = 1      # 100 px building
    mask[30:32, 30:32] = 1    # 4 px noise
    cleaned, count = clean_mask(mask, min_area_px=20)
    assert count == 1
    assert cleaned.sum() == 100


def test_population_follows_thesis_formula():
    cfg = load_config()
    mask = np.zeros((200, 200), np.uint8)
    mask[0:100, 0:100] = 1    # 10,000 m2 footprint at 1 m/pixel
    report = estimate(mask, cfg)
    # 10,000 m2 x 3 floors / 30 m2 per person = 1,000 people
    assert report.footprint_area_m2 == 10000
    assert report.estimated_population == 1000
    assert report.markets_needed == 1          # ceil(1000 / 1500)
    assert report.hospitals_needed == 1        # ceil(1000 / 10000)
    assert report.built_up_ratio == 0.25


def test_empty_mask_needs_nothing():
    report = estimate(np.zeros((64, 64), np.uint8), load_config())
    assert report.estimated_population == 0 and report.hospitals_needed == 0


def test_config_override():
    cfg = load_config(overrides=["population.avg_floors=5", "train.epochs=3"])
    assert cfg["population"]["avg_floors"] == 5 and cfg["train"]["epochs"] == 3


def test_metrics_ignore_nodata_pixels():
    pred = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [0, 1]])
    valid = np.array([[1, 1], [1, 0]], bool)
    meter = SegmentationMeter()
    meter.update(pred, target, valid)
    m = meter.compute()
    assert (meter.tp, meter.fp, meter.fn, meter.tn) == (1, 1, 0, 1)
    assert abs(m["iou"] - 0.5) < 1e-6


def test_best_threshold_picks_separating_value():
    prob = np.array([[0.9, 0.7], [0.3, 0.1]])
    target = np.array([[1, 1], [0, 0]])
    t, m = best_threshold([prob], [target], [np.ones_like(target, bool)])
    assert 0.3 < t <= 0.7 and m["f1"] > 0.99
