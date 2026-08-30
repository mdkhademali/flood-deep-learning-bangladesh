"""
src/training/train_baseline.py
---------------------------------------------------------------------------
Trains the Random Forest / XGBoost pixel-wise baselines on the SAR-derived
feature stack (Section 7), evaluates them with src/evaluation/metrics.py,
and saves feature-importance tables used for Figure 6.
---------------------------------------------------------------------------
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import rasterio

from src.evaluation.metrics import compute_binary_metrics
from src.features.sar_features import build_feature_stack
from src.models.baseline_ml import (
    build_random_forest,
    build_xgboost,
    feature_importance_table,
    stack_to_matrix,
)
from src.utils.common import ensure_dirs, load_config, set_seed, setup_logger


def load_band(path: str, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(band).astype(np.float32)


def spatial_train_test_mask(shape, train_frac=0.6, val_frac=0.2):
    """Row-block split mirroring src/data/patch_dataset.build_spatial_split_index,
    but returning a per-pixel split label array of the same shape as the input.
    """
    h, w = shape
    train_cut = int(h * train_frac)
    val_cut = int(h * (train_frac + val_frac))
    split = np.empty(shape, dtype=object)
    split[:train_cut, :] = "train"
    split[train_cut:val_cut, :] = "val"
    split[val_cut:, :] = "test"
    return split


def main(config_path: str, pre_vv: str, pre_vh: str, peak_vv: str, peak_vh: str, label_path: str):
    cfg = load_config(config_path)
    logger = setup_logger("train_baseline", "outputs/metrics/train_baseline.log")
    set_seed(cfg["project"]["random_seed"])
    ensure_dirs("outputs/models", "outputs/metrics")

    vv_pre = load_band(pre_vv)
    vh_pre = load_band(pre_vh)
    vv_peak = load_band(peak_vv)
    vh_peak = load_band(peak_vh)
    label = load_band(label_path).astype(np.uint8)

    feature_stack = build_feature_stack(vv_pre, vh_pre, vv_peak, vh_peak)
    X, feature_names = stack_to_matrix(feature_stack)
    y = label.reshape(-1)

    split_mask = spatial_train_test_mask(vv_peak.shape,
                                          cfg["data"]["train_region_frac"],
                                          cfg["data"]["val_region_frac"]).reshape(-1)

    train_idx = split_mask == "train"
    test_idx = split_mask == "test"

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    results = {}

    logger.info("Training Random Forest on %d pixels...", X_train.shape[0])
    rf = build_random_forest(cfg)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["random_forest"] = compute_binary_metrics(y_test, rf_pred)
    joblib.dump(rf, "outputs/models/random_forest.joblib")

    rf_importance = feature_importance_table(rf, feature_names)
    with open("outputs/metrics/rf_feature_importance.json", "w") as f:
        json.dump(rf_importance, f, indent=2)

    try:
        logger.info("Training XGBoost on %d pixels...", X_train.shape[0])
        xgb = build_xgboost(cfg)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        results["xgboost"] = compute_binary_metrics(y_test, xgb_pred)
        joblib.dump(xgb, "outputs/models/xgboost.joblib")
    except ImportError as e:
        logger.warning(str(e))
        results["xgboost"] = None

    with open("outputs/metrics/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Baseline training complete. Results saved to outputs/metrics/baseline_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RF/XGBoost baseline flood classifiers")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--pre_vv", required=True)
    parser.add_argument("--pre_vh", required=True)
    parser.add_argument("--peak_vv", required=True)
    parser.add_argument("--peak_vh", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    main(args.config, args.pre_vv, args.pre_vh, args.peak_vv, args.peak_vh, args.label)
