"""Massachusetts Buildings dataset: loading, random full-resolution crops and augmentation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:  # torch is only needed for training, not for the image helpers below
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


def read_image(path: str | Path) -> np.ndarray:
    """Read an RGB image as uint8 (H, W, 3). Works with Unicode paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a building mask as uint8 {0, 1} (H, W)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 127).astype(np.uint8)


def nodata_mask(img: np.ndarray) -> np.ndarray:
    """True where the image is the pure-white 'no data' border of Massachusetts tiles."""
    return np.all(img >= 250, axis=-1)


def list_pairs(split_dir: str | Path, image_suffix=".tiff", mask_suffix=".tif") -> list[tuple[Path, Path]]:
    """Return (image, mask) path pairs of one split folder (``images/`` + ``masks/``)."""
    split_dir = Path(split_dir)
    pairs = []
    for img_path in sorted((split_dir / "images").glob(f"*{image_suffix}")):
        mask_path = split_dir / "masks" / (img_path.stem + mask_suffix)
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found in {split_dir}")
    return pairs


def to_tensor(img: np.ndarray):
    """uint8 (H, W, 3) -> float tensor (3, H, W) in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div_(255.0)


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Random 90-degree rotation and flips; buildings have no preferred orientation."""
    k = int(rng.integers(4))
    img, mask = np.rot90(img, k), np.rot90(mask, k)
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:  # mild brightness / contrast jitter
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img, mask


class RandomPatchDataset(Dataset):
    """Draws random full-resolution crops from the training images.

    Training on crops instead of resizing 1500x1500 tiles down to 256x256 keeps
    the 1 m/pixel resolution, so small houses stay visible to the network.
    """

    def __init__(self, pairs, patch_size=256, patches_per_image=16, max_nodata_fraction=0.5,
                 cache_in_memory=True, seed=42, train=True):
        self.pairs = list(pairs)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.max_nodata_fraction = max_nodata_fraction
        self.train = train
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cache = [self._load(i) for i in range(len(self.pairs))] if cache_in_memory else None

    def _load(self, idx):
        img_path, mask_path = self.pairs[idx]
        return read_image(img_path), read_mask(mask_path)

    def __len__(self):
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, index):
        img, mask = self.cache[index // self.patches_per_image] if self.cache else \
            self._load(index // self.patches_per_image)
        h, w = mask.shape
        p = self.patch_size
        for _ in range(20):  # retry until the crop is mostly real imagery
            y = int(self.rng.integers(0, h - p + 1))
            x = int(self.rng.integers(0, w - p + 1))
            img_crop = img[y:y + p, x:x + p]
            if nodata_mask(img_crop).mean() <= self.max_nodata_fraction:
                break
        mask_crop = mask[y:y + p, x:x + p]
        if self.train:
            img_crop, mask_crop = augment(img_crop, mask_crop, self.rng)
        return to_tensor(img_crop), torch.from_numpy(np.ascontiguousarray(mask_crop)).float().unsqueeze(0)


def worker_init_fn(worker_id: int):
    """Give every DataLoader worker its own random stream."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.dataset.seed + 1000 * (worker_id + 1)
                                             + int(torch.initial_seed() % 1000))
