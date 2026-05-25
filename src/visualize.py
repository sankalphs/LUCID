"""
Visualization module for PSR Shadow Boundary Segmentation.

Generates publication-quality figures including:
- Mask validation overlays
- Training curves
- Qualitative results grid
- Confusion matrices
- Cross-crater comparison charts
- Full-strip segmentation overlays
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional
import json

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_mask_validation(patches: np.ndarray, masks: np.ndarray,
                         methods: list[str], n_samples: int = 10,
                         output_path: Optional[str] = None):
    """
    Plot mask validation: raw patches with overlaid masks.
    
    Args:
        patches: (N, H, W) float32 patches
        masks: (N, H, W) float32 masks
        methods: List of method strings per patch
        n_samples: Number of samples to show
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    indices = np.random.choice(len(patches), min(n_samples, len(patches)), replace=False)
    
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(patches[idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Raw Patch')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masks[idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Generated Mask ({methods[idx]})')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(patches[idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].imshow(masks[idx], cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
    plt.suptitle('Mask Validation: Multi-Otsu Thresholding', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        print(f"Saved mask validation figure: {output_path}")
    plt.close()


def plot_training_curves(log_dir: str, output_path: Optional[str] = None):
    """
    Plot training curves from TensorBoard logs.
    
    Args:
        log_dir: Path to tensorboard log directory
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = {}
    for json_file in Path(log_dir).glob("**/metrics.json"):
        with open(json_file) as f:
            metrics.update(json.load(f))
    
    if not metrics:
        axes[0].text(0.5, 0.5, 'No training data available', 
                    ha='center', va='center', fontsize=14)
        axes[0].set_axis_off()
    else:
        for key, values in metrics.items():
            if 'loss' in key.lower():
                axes[0].plot(values, label=key)
            elif 'iou' in key.lower():
                axes[1].plot(values, label=key)
            elif 'dice' in key.lower():
                axes[2].plot(values, label=key)
    
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('IoU (Jaccard)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Dice (F1)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Training Curves', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        print(f"Saved training curves: {output_path}")
    plt.close()


def plot_qualitative_results(patches: np.ndarray, ground_truth: np.ndarray,
                             predictions: np.ndarray, n_samples: int = 6,
                             output_path: Optional[str] = None):
    """
    Plot qualitative results grid: input / ground truth / prediction.
    
    Args:
        patches: (N, H, W) input patches
        ground_truth: (N, H, W) ground truth masks
        predictions: (N, H, W) predicted probability maps
        n_samples: Number of samples to show
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    indices = np.random.choice(len(patches), min(n_samples, len(patches)), replace=False)
    
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(patches[idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(ground_truth[idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(predictions[idx], cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        pred_binary = (predictions[idx] > 0.5).astype(np.uint8)
        axes[i, 3].imshow(patches[idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 3].imshow(pred_binary, cmap='Reds', alpha=0.4)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    plt.suptitle('Qualitative Segmentation Results', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        print(f"Saved qualitative results: {output_path}")
    plt.close()


def plot_confusion_matrix(pred_flat: np.ndarray, target_flat: np.ndarray,
                          output_path: Optional[str] = None):
    """
    Plot confusion matrix for binary segmentation.
    
    Args:
        pred_flat: Flattened binary predictions
        target_flat: Flattened binary targets
        output_path: Path to save figure
    """
    tp = ((pred_flat == 1) & (target_flat == 1)).sum()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum()
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Shadow', 'Illuminated'])
    ax.set_yticklabels(['Shadow', 'Illuminated'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                   color=color, fontsize=14)
    
    plt.colorbar(im, ax=ax)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        print(f"Saved confusion matrix: {output_path}")
    plt.close()


def plot_baseline_comparison(model_metrics: dict, baseline_metrics: dict,
                             output_path: Optional[str] = None):
    """
    Plot comparison bar chart: model vs. baseline.
    
    Args:
        model_metrics: Dictionary of model metric means
        baseline_metrics: Dictionary of baseline metric means
        output_path: Path to save figure
    """
    metrics = ['iou', 'dice', 'pixel_accuracy']
    model_vals = [model_metrics.get(m, {}).get('mean', 0) for m in metrics]
    baseline_vals = [baseline_metrics.get(m, {}).get('mean', 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, model_vals, width, label='U-Net Model',
                   color='#2196F3', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, baseline_vals, width, label='Intensity Baseline',
                   color='#FF9800', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Score')
    ax.set_title('Model vs. Intensity Baseline Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['IoU', 'Dice', 'Pixel Accuracy'])
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        print(f"Saved baseline comparison: {output_path}")
    plt.close()


def plot_cross_crater_comparison(per_crater_metrics: dict,
                                 output_path: Optional[str] = None):
    """
    Plot per-crater performance comparison bar chart.
    
    Args:
        per_crater_metrics: Dictionary mapping crater names to metric dicts
        output_path: Path to save figure
    """
    craters = list(per_crater_metrics.keys())
    metrics = ['iou', 'dice', 'hd95', 'boundary_f1']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    
    for ax, metric in zip(axes, metrics):
        values = [per_crater_metrics[c].get(metric, {}).get('mean', 0) for c in craters]
        colors = ['#2196F3', '#4CAF50', '#FF9800'][:len(craters)]
        bars = ax.bar(craters, values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_title(metric.upper().replace('_', ' '))
        ax.set_ylabel('Score' if metric != 'hd95' else 'Pixels')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                       fontsize=10)
    
    plt.suptitle('Cross-Crater Performance Comparison', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        print(f"Saved cross-crater comparison: {output_path}")
    plt.close()


def plot_full_strip_overlay(strip_image: np.ndarray, boundary_map: np.ndarray,
                            output_path: Optional[str] = None,
                            figsize: tuple = (20, 8)):
    """
    Plot full-strip segmentation overlay with boundary.
    
    Args:
        strip_image: 2D float32 array of strip image
        boundary_map: 2D binary boundary map
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(strip_image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('OHRC Strip')
    axes[0].axis('off')
    
    axes[1].imshow(boundary_map, cmap='hot')
    axes[1].set_title('Extracted Boundaries')
    axes[1].axis('off')
    
    axes[2].imshow(strip_image, cmap='gray', vmin=0, vmax=1)
    overlay = np.ma.masked_where(boundary_map == 0, boundary_map)
    axes[2].imshow(overlay, cmap='autumn', alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle('Full-Strip Segmentation Result', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        print(f"Saved full-strip overlay: {output_path}")
    plt.close()


def plot_ablation_comparison(ablation_results: dict,
                             output_path: Optional[str] = None):
    """
    Plot ablation study comparison.
    
    Args:
        ablation_results: Dictionary mapping ablation name to metrics
        output_path: Path to save figure
    """
    names = list(ablation_results.keys())
    metrics = ['iou', 'dice', 'hd95', 'boundary_f1']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    
    for ax, metric in zip(axes, metrics):
        values = [ablation_results[n].get(metric, {}).get('mean', 0) for n in names]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_title(metric.upper().replace('_', ' '))
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.3f}', xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(3, 0), textcoords="offset points", ha='left', va='center',
                       fontsize=10)
    
    plt.suptitle('Ablation Study Results', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        print(f"Saved ablation comparison: {output_path}")
    plt.close()
