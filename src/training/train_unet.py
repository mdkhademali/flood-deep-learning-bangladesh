"""
src/training/train_unet.py
---------------------------------------------------------------------------
U-Net training loop (Section 11). Supports CPU/GPU auto-detection, early
stopping, checkpoint saving, and a --debug/--small-run mode for fast
pipeline verification before a full training run.
---------------------------------------------------------------------------
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.augmentation import SarAugment
from src.data.patch_dataset import FloodPatchDataset, build_spatial_split_index
from src.models.losses import get_loss
from src.models.unet import UNet
from src.utils.common import ensure_dirs, get_device, load_config, set_seed, setup_logger


def iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = (intersection + 1e-9) / (union + 1e-9)
    return iou.mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    model.train() if train else model.eval()
    total_loss, total_iou, n_batches = 0.0, 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, lbls)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_iou += iou_from_logits(logits, lbls)
            n_batches += 1

    n_batches = max(n_batches, 1)
    return total_loss / n_batches, total_iou / n_batches


def train(config_path: str = "configs/config.yaml", debug: bool = False,
          image_path: str = None, label_path: str = None, experiment: str = "B_vv_vh"):
    cfg = load_config(config_path)
    logger = setup_logger("train_unet", "outputs/metrics/train.log")
    set_seed(cfg["project"]["random_seed"])
    device = get_device()

    debug = debug or cfg["project"].get("debug", False)

    if image_path is None or label_path is None:
        logger.warning(
            "No --image_path/--label_path provided. This script expects "
            "GeoTIFFs exported from gee/sentinel1_preprocessing.js and "
            "gee/flood_dataset_export.js. Run those first, then re-run "
            "with --image_path and --label_path set."
        )
        return

    data_cfg = cfg["data"]
    index = build_spatial_split_index(
        image_path, label_path,
        patch_size=data_cfg["patch_size"],
        stride=data_cfg["patch_stride"],
        train_frac=data_cfg["train_region_frac"],
        val_frac=data_cfg["val_region_frac"],
    )

    aug = SarAugment(**cfg["augmentation"] if False else {})  # config keys mapped below
    aug = SarAugment(
        h_flip_p=cfg["augmentation"]["horizontal_flip_p"],
        v_flip_p=cfg["augmentation"]["vertical_flip_p"],
        rot90_p=cfg["augmentation"]["rotate90_p"],
    )

    train_ds = FloodPatchDataset(index, split="train", transform=aug)
    val_ds = FloodPatchDataset(index, split="val", transform=None)

    if debug:
        n = min(cfg["project"].get("debug_num_patches", 32), len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        val_ds = Subset(val_ds, list(range(min(n // 4, len(val_ds)))))
        logger.info("DEBUG MODE: using %d train / %d val patches", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                               shuffle=True, num_workers=cfg["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=cfg["training"]["num_workers"])

    in_channels = len(cfg["experiments"][experiment]["bands"])
    model = UNet(
        in_channels=in_channels,
        num_classes=cfg["model"]["num_classes"],
        base_filters=cfg["model"]["base_filters"],
        depth=cfg["model"]["depth"],
        use_batchnorm=cfg["model"]["use_batchnorm"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = get_loss(cfg["model"]["loss"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg["training"]["scheduler_patience"]
    )

    ensure_dirs(cfg["training"]["checkpoint_dir"], cfg["training"]["log_dir"])

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    epochs = 2 if debug else cfg["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_iou = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_iou = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f | train_iou=%.4f val_iou=%.4f | %.1fs",
            epoch, epochs, train_loss, val_loss, train_iou, val_iou, time.time() - t0,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / "unet_best.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "config": cfg}, ckpt_path)
            logger.info("Saved new best checkpoint: %s", ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    history_path = Path(cfg["training"]["log_dir"]) / "unet_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net flood segmentation model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--image_path", default=None, help="Path to exported multi-band S1 GeoTIFF")
    parser.add_argument("--label_path", default=None, help="Path to exported flood label GeoTIFF")
    parser.add_argument("--experiment", default="B_vv_vh", choices=["A_vv_only", "B_vv_vh", "C_vv_vh_derived"])
    parser.add_argument("--debug", action="store_true", help="Fast smoke-test run on a small subset")
    args = parser.parse_args()

    train(config_path=args.config, debug=args.debug,
          image_path=args.image_path, label_path=args.label_path, experiment=args.experiment)
