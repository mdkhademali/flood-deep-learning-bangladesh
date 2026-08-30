"""
src/evaluation/metrics.py
---------------------------------------------------------------------------
Segmentation / classification metrics shared by the RF, XGBoost, and U-Net
evaluation pipelines (Section 12). Accuracy alone is intentionally NOT used
as the headline metric because flood pixels are a minority class.
---------------------------------------------------------------------------
"""
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the full metric suite for binary flood/non-flood predictions.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Flattened binary arrays (0 = non-flood, 1 = flood).
    """
    y_true = y_true.astype(np.uint8).ravel()
    y_pred = y_pred.astype(np.uint8).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    iou_flood = tp / (tp + fp + fn + 1e-9)
    iou_background = tn / (tn + fp + fn + 1e-9)
    mean_iou = (iou_flood + iou_background) / 2
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "iou_flood": iou_flood,
        "iou_background": iou_background,
        "mean_iou": mean_iou,
        "dice": dice,
        "kappa": cohen_kappa_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics


def format_metrics_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format a {model_name: metrics_dict} mapping as a Markdown table."""
    cols = ["accuracy", "precision", "recall", "f1", "mean_iou", "dice", "kappa"]
    header = "| Model | " + " | ".join(c.title() for c in cols) + " |\n"
    sep = "|---|" + "---|" * len(cols) + "\n"
    rows = ""
    for model_name, m in results.items():
        if m is None:
            rows += f"| {model_name} | " + " | ".join(["N/A (to be generated after running the pipeline)"] * len(cols)) + " |\n"
        else:
            rows += f"| {model_name} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |\n"
    return header + sep + rows
