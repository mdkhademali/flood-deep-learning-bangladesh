"""tests/test_patch_split.py — verifies spatial (non-random-pixel) splitting logic."""
import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.data.patch_dataset import build_spatial_split_index


def _write_dummy_raster(path, height=512, width=512, count=2):
    transform = from_origin(0, 0, 10, 10)
    data = np.random.rand(count, height, width).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=count, dtype="float32", crs="EPSG:32646", transform=transform,
    ) as dst:
        dst.write(data)


def test_spatial_split_no_row_overlap(tmp_path):
    img_path = tmp_path / "img.tif"
    lbl_path = tmp_path / "lbl.tif"
    _write_dummy_raster(str(img_path), count=2)
    _write_dummy_raster(str(lbl_path), count=1)

    index = build_spatial_split_index(str(img_path), str(lbl_path),
                                       patch_size=64, stride=64,
                                       train_frac=0.6, val_frac=0.2)

    train_rows = {p.row_off for p in index if p.split == "train"}
    test_rows = {p.row_off for p in index if p.split == "test"}

    assert len(train_rows) > 0
    assert len(test_rows) > 0
    # No row offset should appear in both train and test (spatial separation)
    assert train_rows.isdisjoint(test_rows)


def test_all_patches_assigned_a_split(tmp_path):
    img_path = tmp_path / "img.tif"
    lbl_path = tmp_path / "lbl.tif"
    _write_dummy_raster(str(img_path))
    _write_dummy_raster(str(lbl_path), count=1)

    index = build_spatial_split_index(str(img_path), str(lbl_path), patch_size=64, stride=64)
    splits = {p.split for p in index}
    assert splits.issubset({"train", "val", "test"})
    assert len(index) > 0
