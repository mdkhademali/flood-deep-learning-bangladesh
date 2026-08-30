"""tests/test_metrics.py — sanity tests for evaluation metrics."""
import numpy as np

from src.evaluation.metrics import compute_binary_metrics


def test_perfect_prediction():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    m = compute_binary_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert abs(m["mean_iou"] - 1.0) < 1e-6


def test_all_wrong_prediction():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = compute_binary_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["f1"] == 0.0


def test_metrics_keys_present():
    y_true = np.random.randint(0, 2, size=100)
    y_pred = np.random.randint(0, 2, size=100)
    m = compute_binary_metrics(y_true, y_pred)
    for key in ["accuracy", "precision", "recall", "f1", "iou_flood", "mean_iou", "dice", "kappa"]:
        assert key in m
