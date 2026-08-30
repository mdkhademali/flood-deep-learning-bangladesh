"""
src/mapping/flood_mapping.py
---------------------------------------------------------------------------
Applies a trained U-Net to full Sentinel-1 composites (sliding-window
inference), writes flood extent GeoTIFFs, and computes multi-temporal
flood dynamics statistics (Sections 13-14).
---------------------------------------------------------------------------
"""
from pathlib import Path

import numpy as np
import rasterio
import torch

from src.data.patch_dataset import normalize_sar
from src.models.unet import UNet
from src.utils.common import get_device, load_config


def sliding_window_inference(model, image: np.ndarray, patch_size: int = 256,
                              stride: int = 128, device=None, threshold: float = 0.5) -> np.ndarray:
    """Run the model over a full [C,H,W] image with overlapping patches,
    averaging predictions in overlap regions.
    """
    C, H, W = image.shape
    prob_sum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for row in range(0, H - patch_size + 1, stride):
            for col in range(0, W - patch_size + 1, stride):
                patch = image[:, row:row + patch_size, col:col + patch_size]
                tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                logits = model(tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                prob_sum[row:row + patch_size, col:col + patch_size] += probs
                count[row:row + patch_size, col:col + patch_size] += 1

    count[count == 0] = 1
    prob_map = prob_sum / count
    flood_map = (prob_map > threshold).astype(np.uint8)
    return flood_map, prob_map


def load_model(checkpoint_path: str, cfg: dict, in_channels: int, device):
    model = UNet(
        in_channels=in_channels,
        num_classes=cfg["model"]["num_classes"],
        base_filters=cfg["model"]["base_filters"],
        depth=cfg["model"]["depth"],
        use_batchnorm=cfg["model"]["use_batchnorm"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def map_flood_extent(image_path: str, checkpoint_path: str, out_path: str,
                      config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    device = get_device()

    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
        profile = src.profile

    image = normalize_sar(image)
    model = load_model(checkpoint_path, cfg, in_channels=image.shape[0], device=device)
    flood_map, prob_map = sliding_window_inference(model, image, device=device)

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(flood_map, 1)

    return flood_map, prob_map


def flooded_area_stats(flood_map: np.ndarray, pixel_area_m2: float = 100.0) -> dict:
    """Compute flooded area (km2) and percentage of scene flooded."""
    total_pixels = flood_map.size
    flood_pixels = int(flood_map.sum())
    flooded_area_km2 = flood_pixels * pixel_area_m2 / 1e6
    total_area_km2 = total_pixels * pixel_area_m2 / 1e6
    pct_flooded = 100 * flood_pixels / total_pixels
    return {
        "flooded_area_km2": round(flooded_area_km2, 3),
        "total_area_km2": round(total_area_km2, 3),
        "pct_flooded": round(pct_flooded, 2),
        "flood_pixel_count": flood_pixels,
    }


def flood_dynamics(pre_flood_map: np.ndarray, peak_flood_map: np.ndarray,
                    post_flood_map: np.ndarray, pixel_area_m2: float = 100.0) -> dict:
    """Compute persistence, expansion, and recession between three time steps."""
    persistent = np.logical_and(peak_flood_map == 1, post_flood_map == 1)
    receded = np.logical_and(peak_flood_map == 1, post_flood_map == 0)
    expanded = np.logical_and(pre_flood_map == 0, peak_flood_map == 1)

    return {
        "pre_flood": flooded_area_stats(pre_flood_map, pixel_area_m2),
        "peak_flood": flooded_area_stats(peak_flood_map, pixel_area_m2),
        "post_flood": flooded_area_stats(post_flood_map, pixel_area_m2),
        "persistent_flood_km2": round(persistent.sum() * pixel_area_m2 / 1e6, 3),
        "receded_flood_km2": round(receded.sum() * pixel_area_m2 / 1e6, 3),
        "expanded_flood_km2": round(expanded.sum() * pixel_area_m2 / 1e6, 3),
    }
