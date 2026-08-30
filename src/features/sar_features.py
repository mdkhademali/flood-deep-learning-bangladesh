"""
src/features/sar_features.py
---------------------------------------------------------------------------
SAR-derived feature engineering for the Random Forest / XGBoost baseline
(Section 7). All features are computed from VV/VH backscatter (dB).
---------------------------------------------------------------------------
"""
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def vv_vh_ratio(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """Linear-scale VV/VH ratio computed from dB inputs."""
    vv_lin = 10 ** (vv_db / 10.0)
    vh_lin = 10 ** (vh_db / 10.0)
    return vv_lin / (vh_lin + 1e-6)


def vv_vh_diff(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """Simple dB difference (proportional to log of the linear ratio)."""
    return vv_db - vh_db


def temporal_change(peak_db: np.ndarray, pre_db: np.ndarray) -> np.ndarray:
    """Backscatter change between the peak-flood and pre-flood composites."""
    return peak_db - pre_db


def glcm_contrast(band_db: np.ndarray, levels: int = 32, patch: int = 7) -> np.ndarray:
    """Compute a local GLCM-contrast texture map via a sliding window.

    NOTE: this is a straightforward, moderately slow reference implementation
    intended for patch-scale feature extraction, not full-scene processing.
    For large scenes, precompute on a downsampled grid or restrict to
    sampled training locations.
    """
    h, w = band_db.shape
    half = patch // 2
    out = np.zeros_like(band_db, dtype=np.float32)

    # Quantize to `levels` grey levels for GLCM computation
    b_min, b_max = np.nanpercentile(band_db, 1), np.nanpercentile(band_db, 99)
    quant = np.clip((band_db - b_min) / (b_max - b_min + 1e-6), 0, 1)
    quant = (quant * (levels - 1)).astype(np.uint8)

    for i in range(half, h - half, 4):  # stride of 4 for tractability
        for j in range(half, w - half, 4):
            window = quant[i - half:i + half + 1, j - half:j + half + 1]
            glcm = graycomatrix(
                window, distances=[1], angles=[0], levels=levels,
                symmetric=True, normed=True,
            )
            contrast = graycoprops(glcm, "contrast")[0, 0]
            out[i - 2:i + 2, j - 2:j + 2] = contrast

    return out


def build_feature_stack(vv_pre, vh_pre, vv_peak, vh_peak) -> dict:
    """Assemble the named feature stack used by the RF/XGBoost baseline
    and by Experiment C (VV + VH + derived features).
    """
    features = {
        "VV": vv_peak,
        "VH": vh_peak,
        "VV_VH_ratio": vv_vh_ratio(vv_peak, vh_peak),
        "VV_VH_diff": vv_vh_diff(vv_peak, vh_peak),
        "VV_change": temporal_change(vv_peak, vv_pre),
        "VH_change": temporal_change(vh_peak, vh_pre),
        "glcm_contrast_VV": glcm_contrast(vv_peak),
    }
    return features
