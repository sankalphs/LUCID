"""
Full training script with tqdm progress bars.
Trains U-Net for 100 epochs on mixed PSR patches.
Saves best model checkpoint.
"""
import sys
sys.path.insert(0, '.')

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import numpy as np
import torch
import torch.nn as nn
import yaml
import json
import time
from pathlib import Path
from tqdm import tqdm, trange

from src.data.dataset import PSRDataset
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.mask_generator import generate_mask, clean_mask
from src.models.losses import WeightedBCEDiceLoss
from src.train import MetricTracker
import segmentation_models_pytorch as smp


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_pbar):
    model.train()
    tracker = MetricTracker()
    tracker.reset()

    batch_pbar = tqdm(loader, desc=f"  Train", leave=False, ncols=80)
    for patches, masks in batch_pbar:
        patches = patches.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        pred = torch.sigmoid(logits)
        tracker.update(pred.detach(), masks.detach(), loss.item())
        batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

    return tracker.compute_metrics()


def validate(model, loader, criterion, device):
    model.eval()
    tracker = MetricTracker()
    tracker.reset()

    with torch.no_grad():
        for patches, masks in loader:
            patches = patches.to(device)
            masks = masks.to(device)
            logits = model(patches)
            loss = criterion(logits, masks)
            pred = torch.sigmoid(logits)
            tracker.update(pred, masks, loss.item())

    return tracker.compute_metrics()


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")

    patches_dir = str(Path(config["data"]["base_dir"]) / "patches")

    print("\nLoading datasets...")
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    train_ds = PSRDataset(patches_dir, "mixed", "train", transform=train_transform)
    val_ds = PSRDataset(patches_dir, "mixed", "val", transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False
    )

    print(f"Train: {len(train_ds)} patches ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} patches ({len(val_loader)} batches)")

    model = smp.Unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=None,
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"],
        activation=None,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet18 U-Net, {n_params:,} parameters")

    criterion = WeightedBCEDiceLoss(
        pos_weight=config["loss"]["pos_weight"],
        dice_weight=config["loss"]["dice_weight"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=config["training"]["scheduler"]["factor"],
        patience=config["training"]["scheduler"]["patience"],
        min_lr=config["training"]["scheduler"]["min_lr"]
    )

    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping"]["patience"]
    )

    n_epochs = config["training"]["epochs"]
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [],
               "val_dice": [], "val_accuracy": [], "learning_rate": []}

    print(f"\nStarting training: {n_epochs} epochs, batch_size={config['training']['batch_size']}")
    print("=" * 70)

    total_start = time.time()

    epoch_pbar = trange(1, n_epochs + 1, desc="Epochs", ncols=100)

    for epoch in epoch_pbar:
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch_pbar)
        val_m = validate(model, val_loader, criterion, device)

        scheduler.step(val_m["iou"])
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_iou"].append(train_m["iou"])
        history["val_iou"].append(val_m["iou"])
        history["val_dice"].append(val_m["dice"])
        history["val_accuracy"].append(val_m.get("accuracy", 0))
        history["learning_rate"].append(lr)

        epoch_pbar.set_postfix({
            "trL": f"{train_m['loss']:.4f}",
            "trI": f"{train_m['iou']:.4f}",
            "vlL": f"{val_m['loss']:.4f}",
            "vlI": f"{val_m['iou']:.4f}",
            "vlD": f"{val_m['dice']:.4f}",
        })

        marker = ""
        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            marker = " *"

        if (epoch) % 10 == 0 or epoch == 1:
            tqdm.write(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {train_m['loss']:.4f} IoU: {train_m['iou']:.4f} | "
                f"Val Loss: {val_m['loss']:.4f} IoU: {val_m['iou']:.4f} "
                f"Dice: {val_m['dice']:.4f} Acc: {val_m.get('accuracy',0):.4f} | "
                f"LR: {lr:.6f} | {elapsed:.1f}s{marker}"
            )

        if early_stopping(val_m["iou"]):
            tqdm.write(f"\nEarly stopping at epoch {epoch}")
            break

    total_time = time.time() - total_start
    epoch_pbar.close()

    torch.save(model.state_dict(), ckpt_dir / "last_model.pth")

    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Epochs completed: {len(history['val_iou'])}")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final val Dice: {history['val_dice'][-1]:.4f}")
    print(f"Checkpoint saved: {ckpt_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
