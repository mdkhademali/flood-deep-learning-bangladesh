"""
src/models/unet.py
---------------------------------------------------------------------------
A standard U-Net for binary flood/non-flood semantic segmentation of
Sentinel-1 SAR patches. Encoder-decoder with skip connections, batch
normalization, and dropout in the bottleneck — intentionally conservative
(no architectural novelty) per Section 8: the goal is scientific validity
and reproducibility.
---------------------------------------------------------------------------
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Configurable-depth U-Net.

    Parameters
    ----------
    in_channels : int
        Number of input SAR bands/features (1 for VV-only, 2 for VV+VH, etc.)
    num_classes : int
        1 for binary segmentation with sigmoid output.
    base_filters : int
        Number of filters in the first encoder block; doubles at each depth.
    depth : int
        Number of downsampling stages.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 1,
                 base_filters: int = 32, depth: int = 4,
                 use_batchnorm: bool = True, dropout: float = 0.2):
        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        f = base_filters
        enc_channels = []
        for d in range(depth):
            self.encoders.append(ConvBlock(ch, f, use_batchnorm))
            self.pools.append(nn.MaxPool2d(2))
            enc_channels.append(f)
            ch = f
            f *= 2

        self.bottleneck = ConvBlock(ch, f, use_batchnorm, dropout=dropout)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for d in reversed(range(depth)):
            self.upconvs.append(nn.ConvTranspose2d(f, enc_channels[d], 2, stride=2))
            self.decoders.append(ConvBlock(enc_channels[d] * 2, enc_channels[d], use_batchnorm))
            f = enc_channels[d]

        self.out_conv = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            # handle any off-by-one size mismatch from odd input dims
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.out_conv(x)


if __name__ == "__main__":
    model = UNet(in_channels=2, num_classes=1, base_filters=32, depth=4)
    dummy = torch.randn(2, 2, 256, 256)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: (2, 1, 256, 256)
