"""tests/test_unet.py — architecture sanity checks."""
import torch

from src.models.unet import UNet


def test_unet_output_shape():
    model = UNet(in_channels=2, num_classes=1, base_filters=16, depth=3)
    x = torch.randn(2, 2, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)


def test_unet_single_channel():
    model = UNet(in_channels=1, num_classes=1, base_filters=16, depth=2)
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_unet_five_channel_experiment_c():
    model = UNet(in_channels=5, num_classes=1, base_filters=16, depth=3)
    x = torch.randn(1, 5, 128, 128)
    out = model(x)
    assert out.shape == (1, 1, 128, 128)
