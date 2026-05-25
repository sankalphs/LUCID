"""
PSR Shadow Boundary Segmentation Pipeline

Main orchestrator that runs the complete pipeline:
1. Validate masks
2. Train model
3. Evaluate against baseline
4. Run ablation studies
5. Generate paper figures

Usage:
    python run_pipeline.py --phase all
    python run_pipeline.py --phase train
    python run_pipeline.py --phase evaluate
    python run_pipeline.py --phase ablation
    python run_pipeline.py --phase figures
"""

import sys
import argparse
import yaml
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.train import set_seed, build_model, build_dataloaders, train, load_checkpoint
from src.evaluate import evaluate_model, compare_with_baseline, Evaluator
from src.data.splits import print_split_summary, load_split
from src.baselines.intensity_threshold import intensity_baseline
from src.visualize import (
    plot_mask_validation, plot_training_curves,
    plot_qualitative_results, plot_confusion_matrix,
    plot_baseline_comparison, plot_cross_crater_comparison,
    plot_ablation_comparison, plot_full_strip_overlay
)
from src.data.mask_generator import generate_masks_batch


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def phase_validate_masks(config: dict):
    """Phase 1: Validate mask generation quality."""
    print("\n" + "=" * 60)
    print("PHASE: MASK VALIDATION")
    print("=" * 60)
    
    patches_dir = str(Path(config['data']['base_dir']) / 'patches')
    print_split_summary(patches_dir)
    
    for cls in ['psr', 'sunlit', 'mixed']:
        try:
            data = load_split(patches_dir, 'train', [cls])
            patches = data['patches']
            
            masks, methods, stats = generate_masks_batch(
                patches[:100], data['class_labels'][:100]
            )
            
            print(f"\n{cls.upper()} class:")
            print(f"  Total patches: {stats['total_patches']}")
            print(f"  Method distribution: {stats['method_counts']}")
            print(f"  Fallback rate: {stats['fallback_rate']:.2%}")
            print(f"  Mean illuminated fraction: {stats['mean_illuminated_fraction']:.4f}")
            
            if stats['warning_low_std']:
                print(f"  WARNING: High fallback rate ({stats['fallback_rate']:.2%})")
            
            output_dir = Path(config['outputs']['figure_dir']) / 'mask_validation'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            indices = np.random.choice(len(patches), min(10, len(patches)), replace=False)
            plot_mask_validation(
                patches[indices], masks[indices], [methods[i] for i in indices],
                n_samples=10,
                output_path=str(output_dir / f'{cls}_mask_validation.png')
            )
        except Exception as e:
            print(f"  Error processing {cls}: {e}")


def phase_train(config: dict, resume: str = None):
    """Phase 2: Train the model."""
    print("\n" + "=" * 60)
    print("PHASE: TRAINING")
    print("=" * 60)
    
    model, best_iou = train('configs/config.yaml', resume)
    return model


def phase_evaluate(config: dict):
    """Phase 3: Evaluate model and compare with baseline."""
    print("\n" + "=" * 60)
    print("PHASE: EVALUATION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(config, device)
    checkpoint_path = Path(PROJECT_ROOT) / config['outputs']['checkpoint_dir'] / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}. Run training first.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    patches_dir = str(Path(config['data']['base_dir']) / 'patches')
    _, val_loader = build_dataloaders(config)
    
    evaluator = Evaluator()
    
    model.eval()
    with torch.no_grad():
        for patches, masks in val_loader:
            patches = patches.to(device)
            probs = torch.sigmoid(model(patches))
            evaluator.update(probs.cpu().numpy(), masks.numpy())
    
    model_results = evaluator.compute_aggregate()
    
    evaluator.print_summary("U-Net Model Results")
    
    val_data = load_split(patches_dir, 'train', ['mixed'])
    val_patches = val_data['patches'][:min(200, len(val_data['patches']))]
    
    val_masks_list = []
    for p in val_patches:
        mask = intensity_baseline(p, threshold=0.1)
        val_masks_list.append(mask)
    val_masks = np.array(val_masks_list)
    
    baseline_eval = Evaluator()
    for i in range(len(val_patches)):
        baseline_eval.update(
            val_masks[i:i+1, np.newaxis, :, :],
            (val_patches[i:i+1] > 0.05).astype(np.float32)[:, np.newaxis, :, :]
        )
    baseline_results = baseline_eval.compute_aggregate()
    
    baseline_eval.print_summary("Intensity Baseline Results")
    
    comparison = compare_with_baseline(evaluator, val_patches, val_masks)
    
    print("\nModel vs Baseline Comparison:")
    print(f"{'Metric':<20} {'Model':>10} {'Baseline':>10} {'Improvement':>12}")
    print("-" * 55)
    for metric, values in comparison.items():
        print(f"{metric:<20} {values['model']:>10.4f} {values['baseline']:>10.4f} {values['improvement']:>12.4f}")
    
    results_path = Path(PROJECT_ROOT) / 'outputs' / 'results.json'
    results = {
        'model_results': model_results,
        'baseline_results': baseline_results,
        'comparison': comparison,
    }
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return model_results, baseline_results


def phase_ablation(config: dict):
    """Phase 4: Run ablation studies."""
    print("\n" + "=" * 60)
    print("PHASE: ABLATION STUDIES")
    print("=" * 60)
    
    from src.ablation import run_all_ablations
    results = run_all_ablations(config)
    return results


def phase_figures(config: dict):
    """Phase 5: Generate paper-quality figures."""
    print("\n" + "=" * 60)
    print("PHASE: GENERATE FIGURES")
    print("=" * 60)
    
    figure_dir = Path(PROJECT_ROOT) / config['outputs']['figure_dir']
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(PROJECT_ROOT) / config['outputs']['log_dir']
    if log_dir.exists():
        plot_training_curves(str(log_dir), output_path=str(figure_dir / 'training_curves.png'))
    
    results_path = Path(PROJECT_ROOT) / 'outputs' / 'results.json'
    if results_path.exists():
        import json
        with open(results_path) as f:
            results = json.load(f)
        
        if 'model_results' in results and 'baseline_results' in results:
            plot_baseline_comparison(
                results['model_results'],
                results['baseline_results'],
                output_path=str(figure_dir / 'baseline_comparison.png')
            )
    
    ablation_path = Path(PROJECT_ROOT) / 'outputs' / 'ablation_results.json'
    if ablation_path.exists():
        import json
        with open(ablation_path) as f:
            ablation_results = json.load(f)
        plot_ablation_comparison(ablation_results, output_path=str(figure_dir / 'ablation_comparison.png'))
    
    print(f"\nFigures saved to: {figure_dir}")


def main():
    parser = argparse.ArgumentParser(description='PSR Shadow Segmentation Pipeline')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['all', 'validate', 'train', 'evaluate', 'ablation', 'figures'],
                        help='Pipeline phase to run')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume training from')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Config file path')
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['project']['seed'])
    
    if args.phase in ['all', 'validate']:
        phase_validate_masks(config)
    
    if args.phase in ['all', 'train']:
        phase_train(config, args.resume)
    
    if args.phase in ['all', 'evaluate']:
        phase_evaluate(config)
    
    if args.phase in ['all', 'ablation']:
        phase_ablation(config)
    
    if args.phase in ['all', 'figures']:
        phase_figures(config)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
