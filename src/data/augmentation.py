"""
src/data/augmentation.py
---------------------------------------------------------------------------
Lightweight, SAR-appropriate data augmentation. Only geometric transforms
that preserve backscatter physics are used (no color jitter / blur, which
would be meaningless or misleading for radar amplitude data).
---------------------------------------------------------------------------
"""
import random

import numpy as np


class SarAugment:
    """Random flips and 90-degree rotations applied identically to image and label."""

    def __init__(self, h_flip_p: float = 0.5, v_flip_p: float = 0.5, rot90_p: float = 0.5):
        self.h_flip_p = h_flip_p
        self.v_flip_p = v_flip_p
        self.rot90_p = rot90_p

    def __call__(self, img: np.ndarray, lbl: np.ndarray):
        # img: [C, H, W], lbl: [H, W]
        if random.random() < self.h_flip_p:
            img = img[:, :, ::-1].copy()
            lbl = lbl[:, ::-1].copy()
        if random.random() < self.v_flip_p:
            img = img[:, ::-1, :].copy()
            lbl = lbl[::-1, :].copy()
        if random.random() < self.rot90_p:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k, axes=(0, 1)).copy()
        return img, lbl
