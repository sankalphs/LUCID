"""
Ablation studies for PSR Shadow Boundary Segmentation.

Implements controlled ablations to validate design decisions:
1. No augmentation (baseline: with augmentation)
2. No CLAHE (baseline: with CLAHE)
3. Unweighted loss (baseline: weighted loss)

Each ablation trains a model from scratch and evaluates on the same val set.
"""

import sys
import json
import yaml
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import set_seed, build_model, build_dataloaders, train_one_epoch, validate
from src.evaluate import Evaluator, evaluate_model, compute_all_metrics
from src.models.losses import WeightedBCEDiceLoss
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


ABLATION_CONFIGS = {
    'no_augmentation': {
        'description': 'Remove all augmentations',
        'modify': lambda config: {
            **config,
            'augmentation': {}
        },
    },
    'no_clahe': {
        'description': 'Remove CLAHE augmentation only',
        'modify': lambda config: _remove_clahe(config),
    },
    'unweighted_loss': {
        'description': 'Set pos_weight=1.0 (no class balancing)',
        'modify': lambda config: {
            **config,
            'loss': {
                **config['loss'],
                'pos_weight': 1.0,
            }
        },
    },
}


def _remove_clahe(config: dict) -> dict:
    config = copy.deepcopy(config)
    if 'augmentation' in config and 'clahe' in config['augmentation']:
        config['augmentation']['clahe']['p'] = 0.0
    return config


def run_single_ablation(ablation_name: str, config: dict, 
                        device: torch.device) -> dict:
    """
    Run a single ablation study.
    
    Args:
        ablation_name: Name of the ablation configuration
        config: Base configuration
        device: Torch device
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'=' * 60}")
    print(f"ABLATION: {ablation_name}")
    print(f"{'=' * 60}")
    
    set_seed(config['project']['seed'])
    
    model = build_model(config, device)
    
    train_loader, val_loader = build_dataloaders(config)
    
    train_cfg = config['training']
    loss_cfg = config['loss']
    
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
        optimizer, mode='max',
        factor=train_cfg['scheduler']['factor'],
        patience=train_cfg['scheduler']['patience'],
        min_lr=train_cfg['scheduler']['min_lr']
    )
    
    best_iou = 0.0
    epochs = train_cfg['epochs']
    
    output_dir = Path(PROJECT_ROOT) / config['outputs']['checkpoint_dir'] / f"ablation_{ablation_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(PROJECT_ROOT) / config['outputs']['log_dir'] / f"ablation_{ablation_name}"
    writer = SummaryWriter(log_dir)
    
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
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
        writer.add_scalar('learning_rate', current_lr, epoch)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} IoU: {train_metrics['iou']:.4f} | "
                  f"Val IoU: {val_metrics['iou']:.4f} Dice: {val_metrics['dice']:.4f}")
        
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
    
    writer.close()
    
    model.load_state_dict(torch.load(output_dir / 'best_model.pth', map_location=device))
    
    evaluator = evaluate_model(model, val_loader, device, config)
    results = evaluator.compute_aggregate()
    
    print(f"\nResults for {ablation_name}:")
    evaluator.print_summary(f"Ablation: {ablation_name}")
    
    return results


def run_all_ablations(base_config: dict) -> dict:
    """
    Run all ablation studies and compare results.
    
    Args:
        base_config: Base configuration dictionary
    
    Returns:
        Dictionary with all ablation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for ablation_name, ablation_config in ABLATION_CONFIGS.items():
        modified_config = ablation_config['modify'](copy.deepcopy(base_config))
        results[ablation_name] = run_single_ablation(ablation_name, modified_config, device)
    
    output_path = Path(PROJECT_ROOT) / 'outputs' / 'ablation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAblation results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"{'Ablation':<20} {'IoU':>8} {'Dice':>8} {'HD95':>8} {'BoundF1':>8}")
    print("-" * 60)
    
    for name, metrics in results.items():
        iou = metrics.get('iou', {}).get('mean', 0)
        dice = metrics.get('dice', {}).get('mean', 0)
        hd95 = metrics.get('hd95', {}).get('mean', 0)
        bf1 = metrics.get('boundary_f1', {}).get('mean', 0)
        print(f"{name:<20} {iou:>8.4f} {dice:>8.4f} {hd95:>8.4f} {bf1:>8.4f}")
    
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--ablation', type=str, default=None,
                        help='Specific ablation to run (None = all)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.ablation:
        if args.ablation not in ABLATION_CONFIGS:
            print(f"Unknown ablation: {args.ablation}")
            print(f"Available: {list(ABLATION_CONFIGS.keys())}")
            sys.exit(1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        result = run_single_ablation(args.ablation, config, device)
        print(f"\nResult: {result}")
    else:
        results = run_all_ablations(config)
