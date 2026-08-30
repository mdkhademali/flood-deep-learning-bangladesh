"""
src/models/baseline_ml.py
---------------------------------------------------------------------------
Pixel-wise Random Forest and XGBoost baseline classifiers (Section 7).
Operate on the feature stack produced by src/features/sar_features.py.
---------------------------------------------------------------------------
"""
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def stack_to_matrix(feature_stack: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list]:
    """Flatten a dict of 2D feature arrays into an (N_pixels, N_features) matrix."""
    names = list(feature_stack.keys())
    flat = [feature_stack[n].reshape(-1) for n in names]
    X = np.stack(flat, axis=1)
    return X, names


def build_random_forest(cfg: dict) -> RandomForestClassifier:
    rf_cfg = cfg["baseline_ml"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        n_jobs=rf_cfg["n_jobs"],
        random_state=cfg["project"]["random_seed"],
        class_weight="balanced",
    )


def build_xgboost(cfg: dict):
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is not installed. Install with `pip install xgboost` "
            "or skip the XGBoost baseline and use Random Forest only."
        )
    xgb_cfg = cfg["baseline_ml"]["xgboost"]
    return XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        random_state=cfg["project"]["random_seed"],
        eval_metric="logloss",
        n_jobs=-1,
    )


def feature_importance_table(model, feature_names: list) -> list:
    """Return a sorted list of (feature_name, importance) tuples."""
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return pairs
