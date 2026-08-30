"""tests/test_features.py — SAR feature engineering sanity checks."""
import numpy as np

from src.features.sar_features import temporal_change, vv_vh_diff, vv_vh_ratio


def test_vv_vh_diff_shape():
    vv = np.random.uniform(-25, 0, (32, 32)).astype(np.float32)
    vh = np.random.uniform(-30, -5, (32, 32)).astype(np.float32)
    diff = vv_vh_diff(vv, vh)
    assert diff.shape == vv.shape
    np.testing.assert_allclose(diff, vv - vh)


def test_vv_vh_ratio_positive():
    vv = np.random.uniform(-25, 0, (16, 16)).astype(np.float32)
    vh = np.random.uniform(-30, -5, (16, 16)).astype(np.float32)
    ratio = vv_vh_ratio(vv, vh)
    assert np.all(ratio > 0)


def test_temporal_change_zero_when_equal():
    a = np.ones((10, 10), dtype=np.float32) * -15
    change = temporal_change(a, a)
    assert np.allclose(change, 0)
