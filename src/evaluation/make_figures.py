"""
src/evaluation/make_figures.py
---------------------------------------------------------------------------
Generates publication-quality figures (Sections 16-17) from pipeline
outputs. Each function saves PNG (300 DPI) + PDF and is safe to call
independently once the corresponding upstream artifact exists. Figures
whose inputs do not yet exist are skipped with a clear log message rather
than fabricated.
---------------------------------------------------------------------------
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

from src.utils.common import ensure_dirs, setup_logger

logger = setup_logger("make_figures")

FLOOD_CMAP = ListedColormap(["#f0f0f0", "#2166ac"])  # non-flood, flood
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
})


def _save(fig, name: str, out_dir: str = "outputs/figures", dpi: int = 300):
    ensure_dirs(out_dir)
    fig.savefig(f"{out_dir}/{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s.png / .pdf", name)


def fig07_08_training_curves(history_path: str = "outputs/metrics/unet_training_history.json"):
    """Figure 7 (loss curves) and Figure 8 (IoU curves)."""
    if not Path(history_path).exists():
        logger.warning("No training history found at %s — run training first.", history_path)
        return
    with open(history_path) as f:
        hist = json.load(f)

    epochs = range(1, len(hist["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, hist["train_loss"], label="Train loss", color="#2166ac")
    ax.plot(epochs, hist["val_loss"], label="Validation loss", color="#b2182b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE + Dice Loss")
    ax.set_title("U-Net Training / Validation Loss")
    ax.legend(frameon=False)
    _save(fig, "figure07_loss_curves")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, hist["train_iou"], label="Train IoU", color="#2166ac")
    ax.plot(epochs, hist["val_iou"], label="Validation IoU", color="#b2182b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean IoU")
    ax.set_title("U-Net Training / Validation IoU")
    ax.legend(frameon=False)
    _save(fig, "figure08_iou_curves")


def fig09_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "unet"):
    """Figure 9: confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-flood", "Flood"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-flood", "Flood"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, f"figure09_confusion_matrix_{model_name}")


def fig06_feature_importance(importance_path: str = "outputs/metrics/rf_feature_importance.json"):
    """Figure 6: Random Forest feature importance bar chart."""
    if not Path(importance_path).exists():
        logger.warning("No feature importance file found at %s — run baseline training first.", importance_path)
        return
    with open(importance_path) as f:
        pairs = json.load(f)
    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names[::-1], values[::-1], color="#2166ac")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance")
    _save(fig, "figure06_feature_importance")


def fig11_flood_extent_map(flood_map: np.ndarray, title: str = "Flood Extent Map"):
    """Figure 11: single-date flood extent map."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(flood_map, cmap=FLOOD_CMAP)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.02, "Data: Sentinel-1 SAR (Copernicus) | CRS: EPSG:32646",
            transform=ax.transAxes, fontsize=7, color="gray")
    _save(fig, "figure11_flood_extent_map")


def fig12_multitemporal_dynamics(pre_map: np.ndarray, peak_map: np.ndarray, post_map: np.ndarray):
    """Figure 12: pre / peak / post flood maps side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, m, title in zip(axes, [pre_map, peak_map, post_map],
                             ["Pre-flood", "Peak-flood", "Post-flood"]):
        ax.imshow(m, cmap=FLOOD_CMAP)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Multi-Temporal Flood Dynamics")
    _save(fig, "figure12_multitemporal_dynamics")


def fig13_flooded_area_timeseries(dates: list, areas_km2: list):
    """Figure 13: flooded-area time series line chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dates, areas_km2, marker="o", color="#2166ac")
    ax.set_xlabel("Date")
    ax.set_ylabel("Flooded Area (km²)")
    ax.set_title("Flooded Area Time Series")
    fig.autofmt_xdate()
    _save(fig, "figure13_flooded_area_timeseries")


def fig14_exposure_by_lulc(exposure_dict: dict):
    """Figure 14: flood exposure by land-use/land-cover class."""
    if not exposure_dict:
        logger.warning("Empty LULC exposure dict — run exposure_analysis first.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    classes = list(exposure_dict.keys())
    values = list(exposure_dict.values())
    ax.bar(classes, values, color="#4393c3")
    ax.set_ylabel("Exposed Area (km²)")
    ax.set_title("Flood Exposure by Land-Use/Land-Cover Class")
    plt.xticks(rotation=30, ha="right")
    _save(fig, "figure14_exposure_by_lulc")


def fig15_exposure_by_admin(exposure_dict: dict):
    """Figure 15: flood exposure by administrative unit."""
    if not exposure_dict:
        logger.warning("Empty admin exposure dict — run exposure_analysis first.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    units = list(exposure_dict.keys())
    values = list(exposure_dict.values())
    order = np.argsort(values)[::-1]
    ax.barh([units[i] for i in order][::-1], [values[i] for i in order][::-1], color="#b2182b")
    ax.set_xlabel("Exposed Area (km²)")
    ax.set_title("Flood Exposure by Administrative Unit")
    _save(fig, "figure15_exposure_by_admin")


if __name__ == "__main__":
    logger.info("Run this module's functions from a notebook or evaluation "
                "script once upstream artifacts (training history, baseline "
                "results, flood maps) exist. Nothing is fabricated here.")
