"""
Evaluation metrics for PSR Shadow Boundary Segmentation.

Implements comprehensive metrics including IoU, Dice, HD95, and Boundary F1
for comparing model predictions against ground truth masks.
"""

import numpy as np
import torch
from typing import Optional
from pathlib import Path
import json
import yaml

from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Intersection over Union (Jaccard Index)."""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice coefficient (F1 score)."""
    intersection = np.logical_and(pred, target).sum()
    return float(2 * intersection / (pred.sum() + target.sum() + 1e-7))


def compute_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute pixel-level accuracy."""
    return float(np.mean(pred == target))


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute 95th percentile Hausdorff Distance.
    
    Uses distance transform for efficient computation.
    HD95 is more robust to outliers than full Hausdorff Distance.
    """
    if not pred.any() and not target.any():
        return 0.0
    
    if not pred.any() or not target.any():
        max_dim = max(pred.shape)
        return float(max_dim)
    
    pred_binary = pred.astype(bool)
    target_binary = target.astype(bool)
    
    if not pred_binary.any() or not target_binary.any():
        return float(max(pred_binary.shape))
    
    dt_target = distance_transform_edt(~target_binary)
    dt_pred = distance_transform_edt(~pred_binary)
    
    dist_pred_to_target = dt_target[pred_binary]
    dist_target_to_pred = dt_pred[target_binary]
    
    if len(dist_pred_to_target) == 0 or len(dist_target_to_pred) == 0:
        return float(max(pred_binary.shape))
    
    all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_boundary_f1(pred: np.ndarray, target: np.ndarray,
                        tolerance: int = 2) -> float:
    """
    Compute F1 score on boundary pixels within a tolerance distance.
    
    This metric evaluates how well predicted boundaries align with
    ground truth boundaries, allowing for small spatial offsets.
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        tolerance: Maximum distance (in pixels) for a prediction to count as correct
    """
    pred_boundary = find_boundaries(pred.astype(np.uint8), mode='outer')
    target_boundary = find_boundaries(target.astype(np.uint8), mode='outer')
    
    if not pred_boundary.any() and not target_boundary.any():
        return 1.0
    
    if not pred_boundary.any() or not target_boundary.any():
        return 0.0
    
    dt_target_boundary = distance_transform_edt(~target_boundary)
    dt_pred_boundary = distance_transform_edt(~pred_boundary)
    
    pred_correct = np.logical_and(pred_boundary, dt_target_boundary <= tolerance)
    target_correct = np.logical_and(target_boundary, dt_pred_boundary <= tolerance)
    
    tp = pred_correct.sum()
    fp = pred_boundary.sum() - tp
    fn = target_boundary.sum() - tp
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    return float(f1)


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute all evaluation metrics."""
    pred_binary = (pred > 0.5).astype(np.uint8)
    target_binary = (target > 0.5).astype(np.uint8)
    
    return {
        'iou': compute_iou(pred_binary, target_binary),
        'dice': compute_dice(pred_binary, target_binary),
        'pixel_accuracy': compute_pixel_accuracy(pred_binary, target_binary),
        'hd95': compute_hd95(pred_binary, target_binary),
        'boundary_f1': compute_boundary_f1(pred_binary, target_binary),
    }


class Evaluator:
    """
    Comprehensive evaluator for shadow segmentation models.
    
    Collects predictions and computes aggregate metrics across batches.
    
    Args:
        tolerance: Pixel tolerance for boundary F1 metric
    """
    
    def __init__(self, tolerance: int = 2):
        self.tolerance = tolerance
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []
        self.per_patch_metrics = []
    
    def update(self, probs: np.ndarray, targets: np.ndarray):
        """
        Update evaluator with batch predictions.
        
        Args:
            probs: (B, 1, H, W) probability maps
            targets: (B, 1, H, W) binary masks
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        probs = probs.squeeze(1)
        targets = targets.squeeze(1)
        
        for i in range(len(probs)):
            pred_binary = (probs[i] > 0.5).astype(np.uint8)
            target_binary = (targets[i] > 0.5).astype(np.uint8)
            
            metrics = compute_all_metrics(pred_binary, target_binary)
            self.per_patch_metrics.append(metrics)
            
            self.all_probs.append(probs[i])
            self.all_preds.append(pred_binary)
            self.all_targets.append(target_binary)
    
    def compute_aggregate(self) -> dict:
        """Compute aggregate metrics across all patches."""
        if not self.per_patch_metrics:
            return {}
        
        all_keys = self.per_patch_metrics[0].keys()
        aggregates = {}
        
        for key in all_keys:
            values = [m[key] for m in self.per_patch_metrics]
            aggregates[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        
        return aggregates
    
    def compute_per_class(self, class_labels: list[str]) -> dict:
        """Compute metrics grouped by class label."""
        per_class = {}
        
        for i, label in enumerate(class_labels):
            if label not in per_class:
                per_class[label] = []
            if i < len(self.per_patch_metrics):
                per_class[label].append(self.per_patch_metrics[i])
        
        class_aggregates = {}
        for label, metrics_list in per_class.items():
            if not metrics_list:
                continue
            agg = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list]
                agg[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'count': len(values),
                }
            class_aggregates[label] = agg
        
        return class_aggregates
    
    def evaluate_baseline(self, baseline_fn, patches: np.ndarray,
                          targets: np.ndarray) -> dict:
        """
        Evaluate a baseline method against the same targets.
        
        Args:
            baseline_fn: Function that takes a patch and returns a mask
            patches: (N, H, W) array of patches
            targets: (N, H, W) array of target masks
        """
        baseline_metrics = []
        
        for i in range(len(patches)):
            pred = baseline_fn(patches[i])
            target = targets[i].astype(np.uint8)
            metrics = compute_all_metrics(pred, target)
            baseline_metrics.append(metrics)
        
        agg = {}
        for key in baseline_metrics[0].keys():
            values = [m[key] for m in baseline_metrics]
            agg[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
        
        return agg
    
    def save_results(self, path: str, config: Optional[dict] = None):
        """Save evaluation results to JSON file."""
        results = {
            'aggregate': self.compute_aggregate(),
            'n_patches': len(self.per_patch_metrics),
            'per_patch_count': len(self.per_patch_metrics),
        }
        
        if config:
            results['config'] = {
                'model': config.get('model', {}),
                'training': config.get('training', {}),
            }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {path}")
    
    def print_summary(self, title: str = "Evaluation Results"):
        """Print formatted evaluation summary."""
        agg = self.compute_aggregate()
        
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print(f"{'=' * 60}")
        print(f"{'Metric':<20} {'Mean':>8} {'Std':>8} {'Median':>8}")
        print(f"{'-' * 60}")
        
        for key, stats in agg.items():
            print(f"{key:<20} {stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['median']:>8.4f}")
        
        print(f"{'=' * 60}")
        print(f"Total patches evaluated: {len(self.per_patch_metrics)}")
        print(f"{'=' * 60}\n")


def evaluate_model(model, loader, device, config=None) -> Evaluator:
    """
    Evaluate a model on a data loader.
    
    Args:
        model: Trained model
        loader: DataLoader for evaluation
        device: Torch device
        config: Optional config dict for saving
    
    Returns:
        Evaluator with results
    """
    model.eval()
    evaluator = Evaluator()
    
    with torch.no_grad():
        for patches, masks in loader:
            patches = patches.to(device)
            probs = torch.sigmoid(model(patches))
            evaluator.update(probs.cpu().numpy(), masks.numpy())
    
    return evaluator


def compare_with_baseline(evaluator: Evaluator, patches: np.ndarray,
                          targets: np.ndarray, 
                          threshold: float = 0.1) -> dict:
    """Compare model results with intensity baseline."""
    from src.baselines.intensity_threshold import intensity_baseline
    
    baseline_agg = evaluator.evaluate_baseline(
        lambda p: intensity_baseline(p, threshold),
        patches, targets
    )
    
    model_agg = evaluator.compute_aggregate()
    
    comparison = {}
    for key in model_agg:
        comparison[key] = {
            'model': model_agg[key]['mean'],
            'baseline': baseline_agg.get(key, {}).get('mean', 0),
            'improvement': model_agg[key]['mean'] - baseline_agg.get(key, {}).get('mean', 0),
        }
    
    return comparison


if __name__ == '__main__':
    print("Evaluation module loaded. Use via train.py or standalone scripts.")
