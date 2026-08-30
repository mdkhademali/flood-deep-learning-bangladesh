"""
src/data/patch_dataset.py
---------------------------------------------------------------------------
Loads exported Sentinel-1 GeoTIFFs (from gee/sentinel1_preprocessing.js and
gee/flood_dataset_export.js), extracts fixed-size patches, and performs a
SPATIALLY separated train/val/test split (no random pixel mixing) to avoid
spatial autocorrelation leakage between splits — see docs/methodology.md
Section 3.5 for the rationale.
---------------------------------------------------------------------------
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch optional at import time
    Dataset = object


@dataclass
class PatchIndex:
    image_path: str
    label_path: str
    row_off: int
    col_off: int
    size: int
    split: str  # 'train' | 'val' | 'test'


def read_raster(path: str) -> Tuple[np.ndarray, dict]:
    """Read a GeoTIFF and return (array [C,H,W], profile)."""
    with rasterio.open(path) as src:
        arr = src.read()
        profile = src.profile
    return arr, profile


def build_spatial_split_index(
    image_path: str,
    label_path: str,
    patch_size: int = 256,
    stride: int = 128,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> List[PatchIndex]:
    """Build a list of patch locations, assigning each patch to train/val/test
    based on its ROW position in the raster (spatial blocks, not random pixels).

    This mirrors the standard remote-sensing practice of holding out
    contiguous geographic regions for validation/test rather than
    interleaving pixels, which would leak spatial autocorrelation between
    splits and inflate reported accuracy.
    """
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width

    n_rows = max(1, (height - patch_size) // stride + 1)
    train_row_cut = int(n_rows * train_frac)
    val_row_cut = int(n_rows * (train_frac + val_frac))

    index = []
    for r in range(n_rows):
        row_off = r * stride
        if row_off + patch_size > height:
            continue
        if r < train_row_cut:
            split = "train"
        elif r < val_row_cut:
            split = "val"
        else:
            split = "test"

        n_cols = max(1, (width - patch_size) // stride + 1)
        for c in range(n_cols):
            col_off = c * stride
            if col_off + patch_size > width:
                continue
            index.append(
                PatchIndex(
                    image_path=image_path,
                    label_path=label_path,
                    row_off=row_off,
                    col_off=col_off,
                    size=patch_size,
                    split=split,
                )
            )
    return index


class FloodPatchDataset(Dataset):
    """PyTorch Dataset serving (image_patch, label_patch) pairs for a given split."""

    def __init__(self, index: List[PatchIndex], split: str, bands: List[int] = None,
                 transform=None, nodata_value: float = -9999):
        self.index = [p for p in index if p.split == split]
        self.bands = bands  # None -> use all bands in file
        self.transform = transform
        self.nodata_value = nodata_value

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        p = self.index[idx]
        window = Window(p.col_off, p.row_off, p.size, p.size)

        with rasterio.open(p.image_path) as src:
            img = src.read(self.bands, window=window).astype(np.float32)
        with rasterio.open(p.label_path) as src:
            lbl = src.read(1, window=window).astype(np.float32)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img[img == self.nodata_value] = 0.0

        if self.transform is not None:
            img, lbl = self.transform(img, lbl)

        return img, lbl[np.newaxis, ...]


def normalize_sar(img: np.ndarray, db_min: float = -30.0, db_max: float = 5.0) -> np.ndarray:
    """Min-max normalize dB-scaled SAR bands to [0, 1]."""
    img = np.clip(img, db_min, db_max)
    return (img - db_min) / (db_max - db_min)
