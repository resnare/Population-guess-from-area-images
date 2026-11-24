# Satellite Population Estimation

**Building segmentation from satellite imagery with a U-Net, turned into population and urban infrastructure estimates.**

This project detects buildings in 1 m/pixel aerial images with a U-Net trained in PyTorch, then converts the detected building footprints into an estimated population and the number of hospitals, markets and mobile base stations an area needs. It started as my B.Sc. thesis in Mathematical Engineering at Yildiz Technical University (*Deep Learning-Based Population Estimation and Urban Needs Analysis from Satellite Imagery*, 2026) and was later rebuilt into a reproducible, tested pipeline.

<!-- RESULTS:START -->
> Results are added here after the final training run (pixel IoU / F1 on the test set, sample predictions and training curves).
<!-- RESULTS:END -->

## How it works

```
satellite image (1500 x 1500 px, 1 m/px)
        │
        ▼
U-Net (sliding window, 512 px tiles, 64 px overlap)  ──►  building probability map
        │                                                   │
        │                                        threshold tuned on validation set
        ▼                                                   ▼
building mask  ──►  remove blobs < 20 m²  ──►  footprint area, building count
        │
        ▼
population = footprint area × floors / m² per person
        │
        ▼
hospitals, markets, base stations needed
```

| Step | Details |
| --- | --- |
| Data | [Massachusetts Buildings Dataset](https://www.cs.toronto.edu/~vmnih/data/) (Mnih, 2013): 151 aerial tiles of the Boston area, 1500 × 1500 px at 1 m/px; 137 train, 4 validation, 10 test |
| Training input | Random 256 × 256 crops at **full resolution**, 16 per image per epoch; crops that are mostly white "no data" border are skipped; random 90° rotations, flips, brightness/contrast jitter |
| Model | U-Net, 4 levels (32 → 512 channels), BatchNorm, dropout in the bottleneck, ~7.8 M parameters, outputs logits |
| Loss | Weighted BCE (`pos_weight` 3, for the few building pixels) + soft Dice |
| Optimisation | AdamW, cosine learning-rate schedule, mixed precision on GPU, early stopping on validation IoU |
| Evaluation | Full-resolution sliding-window inference; pixel IoU, F1 (Dice), precision, recall; "no data" pixels excluded; decision threshold tuned on the validation set |
| Population model | Transparent, configurable formula (see below); compared on the test set against the same formula applied to the ground-truth masks |

## Improvements over the original thesis code

The thesis prototype (`main2.py`, kept out of this repo) proved the idea; this version fixes the issues that limited it:

| Original prototype | This version |
| --- | --- |
| 1500 × 1500 images resized to 256 × 256, so small houses shrank to a few pixels | Trains on full-resolution crops and predicts with a sliding window |
| Network ended in `sigmoid`, trained with plain `BCELoss`; the weighted loss was defined but never used | Network outputs logits; weighted `BCEWithLogitsLoss` + Dice is actually applied |
| No validation or test evaluation; results shown on a training image | Separate train / validation / test splits, IoU / F1 / precision / recall, per-image report |
| Fixed threshold of 0.3 | Threshold tuned on the validation set and stored with the model |
| Model not saved; everything in one script | Best checkpoint saved; `train.py`, `evaluate.py`, `predict.py`; config file; unit tests |
| Population and facility ratios hard-coded | All assumptions in `configs/default.yaml` |

## Population and urban needs model

```
footprint area A   = building pixels × (pixel size)²
population      P  = A × residential share × average floors / m² per person
hospitals          = ⌈P / 10,000⌉      markets = ⌈P / 1,500⌉      base stations = ⌈P / 2,000⌉
```

Defaults follow the thesis (3 floors, 30 m² per person). They are assumptions, not measured values: adjust them in `configs/default.yaml` for a specific city. Because the dataset has no census labels, `evaluate.py` measures how much the **segmentation** error changes the estimate (population from predicted masks vs. from ground-truth masks), not the accuracy of the assumptions themselves.

## Project structure

```
satellite-population-estimation/
├── configs/default.yaml   all settings: data, model, training, inference, population assumptions
├── src/
│   ├── dataset.py         reading tiles, random crops, augmentation
│   ├── model.py           U-Net
│   ├── losses.py          weighted BCE + Dice
│   ├── metrics.py         IoU / F1 / precision / recall, threshold tuning
│   ├── inference.py       sliding-window prediction, overlays, checkpoint loading
│   ├── population.py      population and infrastructure estimation
│   ├── config.py          YAML config with command-line overrides
│   └── utils.py           seeding, device selection
├── train.py               training with validation, early stopping, threshold tuning
├── evaluate.py            test metrics, per-image report, figures
├── predict.py             run on new images -> mask, overlay, JSON report
└── tests/                 unit tests (pytest)
```

## Getting started

```bash
git clone https://github.com/resnare/satellite-population-estimation.git
cd satellite-population-estimation
pip install -r requirements.txt     # install the PyTorch build for your GPU from pytorch.org first if needed
```

**Data.** Download the Massachusetts Buildings Dataset ([original source](https://www.cs.toronto.edu/~vmnih/data/); mirrors exist on Kaggle) and arrange it as:

```
data/
├── train/  images/*.tiff   masks/*.tif
├── val/    images/*.tiff   masks/*.tif
└── test/   images/*.tiff   masks/*.tif
```

Image and mask share the same file name (`22678915_15.tiff` ↔ `22678915_15.tif`).

**Train, evaluate, predict.**

```bash
python train.py                                   # ~40 epochs; best model -> checkpoints/best_model.pt
python evaluate.py                                # test metrics + figures -> outputs/
python predict.py data/test/images/22828930_15.tiff
python -m pytest                                  # unit tests
```

Any setting can be changed from the command line, e.g. `python train.py --set train.epochs=60 train.batch_size=16`. Training is practical on a GPU; on a free Google Colab GPU, set `data.num_workers=2`.

## Limitations

- The labels come from OpenStreetMap footprints and contain known errors (missing or shifted buildings), which caps the measurable accuracy.
- A single floor count is assumed for every building. The thesis proposes estimating building height from shadow length (H = shadow length × tan(sun elevation)) as the next step.
- The facility ratios are planning rules of thumb, not location-specific standards.

## References

- V. Mnih, *Machine Learning for Aerial Image Labeling*, PhD thesis, University of Toronto, 2013.
- O. Ronneberger, P. Fischer, T. Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.

## Author

**Ruhi Arslan** · [LinkedIn](https://www.linkedin.com/in/ruhiarslan/) · [GitHub](https://github.com/resnare)

## License

MIT
