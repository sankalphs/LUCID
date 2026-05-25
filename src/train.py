"""
Training pipeline for PSR Shadow Boundary Segmentation.

Implements:
- SMP U-Net with ResNet18 encoder
- Weighted BCE + Dice loss
- Adam optimizer with ReduceLROnPlateau scheduler
- Early stopping on validation IoU
- TensorBoard logging
- Checkpoint saving/loading
"""

import os
import sys
import random
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import segmentation_models_pytorch as smp
from src.data.dataset import PSRDataset, CombinedPSRDataset
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.models.losses import WeightedBCEDiceLoss, compute_pos_weight


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class EarlyStopping:
    def __init__(self, patience: int = 15, metric: str = 'val_iou',
                 mode: str = 'max', min_delta: float = 1e-4):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total_loss = 0.0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, loss: float):
        pred_binary = (pred > 0.5).float()
        self.tp += ((pred_binary == 1) & (target == 1)).sum().item()
        self.fp += ((pred_binary == 1) & (target == 0)).sum().item()
        self.fn += ((pred_binary == 0) & (target == 1)).sum().item()
        self.tn += ((pred_binary == 0) & (target == 0)).sum().item()
        self.total_loss += loss * target.size(0)
        self.count += target.size(0)

    def compute_metrics(self) -> dict:
        eps = 1e-7
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        avg_loss = self.total_loss / max(self.count, 1)

        return {
            'loss': avg_loss,
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        }


def build_model(config: dict, device: torch.device) -> nn.Module:
    model_cfg = config['model']
    model = smp.Unet(
        encoder_name=model_cfg['encoder'],
        encoder_weights=model_cfg.get('encoder_weights', None),
        in_channels=model_cfg['in_channels'],
        classes=model_cfg['classes'],
        activation=model_cfg.get('activation', None),
    )
    return model.to(device)


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = config['data']
    train_cfg = config['training']
    aug_cfg = config.get('augmentation', {})

    patches_dir = str(Path(data_cfg['base_dir']) / 'patches')

    train_transform = get_train_transforms() if aug_cfg else None
    val_transform = get_val_transforms()

    train_classes = ['psr', 'sunlit', 'mixed']
    val_strips = data_cfg.get('splits', {}).get('val_strips', ['shackleton_02'])

    train_ds = CombinedPSRDataset(
        patches_dir, split='train',
        classes=train_classes, transform=train_transform
    )

    val_ds = PSRDataset(
        patches_dir, class_label='mixed', split='val',
        transform=val_transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg['batch_size'],
        shuffle=True, num_workers=0, pin_memory=False
    )

    val_loader = DataLoader(
        val_ds, batch_size=train_cfg['batch_size'],
        shuffle=False, num_workers=0, pin_memory=False
    )

    return train_loader, val_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> dict:
    model.train()
    tracker = MetricTracker()
    tracker.reset()

    for patches, masks in loader:
        patches = patches.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        pred = model(patches)
        loss = criterion(pred, masks)
        loss.backward()
        optimizer.step()

        tracker.update(pred.detach(), masks.detach(), loss.item())

    return tracker.compute_metrics()


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> dict:
    model.eval()
    tracker = MetricTracker()
    tracker.reset()

    with torch.no_grad():
        for patches, masks in loader:
            patches = patches.to(device)
            masks = masks.to(device)

            pred = model(patches)
            loss = criterion(pred, masks)

            tracker.update(pred, masks, loss.item())

    return tracker.compute_metrics()


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def train(config_path: str, resume: Optional[str] = None):
    config = load_config(config_path)
    project_cfg = config['project']
    train_cfg = config['training']
    loss_cfg = config['loss']
    output_cfg = config['outputs']

    set_seed(project_cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(PROJECT_ROOT) / output_cfg['checkpoint_dir']
    log_dir = Path(PROJECT_ROOT) / output_cfg['log_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir / f"run_{timestamp}")

    model = build_model(config, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, val_loader = build_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    criterion = WeightedBCEDiceLoss(
        pos_weight=loss_cfg['pos_weight'],
        dice_weight=loss_cfg['dice_weight']
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg.get('weight_decay', 1e-4)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=train_cfg['scheduler']['factor'],
        patience=train_cfg['scheduler']['patience'],
        min_lr=train_cfg['scheduler']['min_lr']
    )

    early_stopping = EarlyStopping(
        patience=train_cfg['early_stopping']['patience'],
        metric=train_cfg['early_stopping']['metric']
    )

    start_epoch = 0
    best_iou = 0.0

    if resume:
        start_epoch, prev_metrics = load_checkpoint(resume, model, optimizer, scheduler)
        best_iou = prev_metrics.get('iou', 0.0)
        print(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")

    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print("-" * 60)

    for epoch in range(start_epoch, train_cfg['epochs']):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['iou'])

        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalars('loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss'],
        }, epoch)
        writer.add_scalars('iou', {
            'train': train_metrics['iou'],
            'val': val_metrics['iou'],
        }, epoch)
        writer.add_scalars('dice', {
            'train': train_metrics['dice'],
            'val': val_metrics['dice'],
        }, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{train_cfg['epochs']} | "
                  f"Train Loss: {train_metrics['loss']:.4f} IoU: {train_metrics['iou']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} IoU: {val_metrics['iou']:.4f} "
                  f"Dice: {val_metrics['dice']:.4f} | LR: {current_lr:.6f}")

        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_metrics,
                output_dir / 'best_model.pth'
            )
            print(f"  -> New best IoU: {best_iou:.4f}, saved checkpoint")

        if early_stopping(val_metrics['iou']):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    save_checkpoint(
        model, optimizer, scheduler, epoch + 1, val_metrics,
        output_dir / 'last_model.pth'
    )

    writer.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"Logs saved to: {log_dir}")
    print("=" * 60)

    return model, best_iou


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PSR Shadow Segmentation Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    train(args.config, args.resume)
