"""
src/models/losses.py
---------------------------------------------------------------------------
Segmentation losses. Combined BCE + Dice loss is used by default: BCE gives
stable pixel-wise gradients while Dice directly addresses the flood/
non-flood class imbalance typical of flood extent mapping.
---------------------------------------------------------------------------
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + (1 - self.bce_weight) * self.dice(logits, targets)


LOSS_REGISTRY = {
    "bce": lambda: nn.BCEWithLogitsLoss(),
    "dice": lambda: DiceLoss(),
    "bce_dice": lambda: BCEDiceLoss(),
}


def get_loss(name: str = "bce_dice"):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Options: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]()
