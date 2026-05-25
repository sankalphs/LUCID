# PSR Shadow Boundary Segmentation

Self-supervised pixel-level segmentation of shadow-illumination boundaries in lunar Permanently Shadowed Regions (PSRs) from Chandrayaan-2 OHRC imagery.

**Paper**: *"Delineation of Shadow-Illumination Boundaries in Lunar Permanently Shadowed Regions via Self-Supervised Segmentation from Chandrayaan-2 OHRC Imagery"*

## Overview

This project implements a binary pixel segmentation system that classifies each pixel in Chandrayaan-2 OHRC lunar imagery as either **shadow** or **illuminated**, then extracts the shadow-illumination boundary as a post-processing step.

### Key Features

- **Self-supervised mask generation**: Multi-Otsu thresholding for automatic training labels
- **U-Net segmentation**: SMP-based U-Net with ResNet18 encoder for binary segmentation
- **Comprehensive evaluation**: IoU, Dice, HD95, Boundary F1 metrics
- **Full-strip inference**: Sliding window with Gaussian-weighted overlap blending
- **Ablation studies**: Controlled experiments validating design decisions
- **Paper-quality figures**: Publication-ready visualizations

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── mask_generator.py    # Multi-Otsu mask generation
│   │   ├── dataset.py           # PyTorch Dataset classes
│   │   ├── splits.py            # Crater-based data splits
│   │   └── augmentations.py     # Albumentations transforms
│   ├── models/
│   │   └── losses.py            # Weighted BCE + Dice loss
│   ├── baselines/
│   │   └── intensity_threshold.py  # Intensity threshold baseline
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation metrics
│   ├── inference.py             # Full-strip inference
│   ├── visualize.py             # Paper-quality figures
│   └── ablation.py              # Ablation studies
├── configs/
│   └── config.yaml              # Experiment configuration
├── run_pipeline.py              # Main orchestrator
└── requirements.txt             # Dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python run_pipeline.py --phase all
```

### 3. Run Individual Phases

```bash
# Validate mask generation
python run_pipeline.py --phase validate

# Train model
python run_pipeline.py --phase train

# Evaluate model
python run_pipeline.py --phase evaluate

# Run ablation studies
python run_pipeline.py --phase ablation

# Generate paper figures
python run_pipeline.py --phase figures
```

### 4. Resume Training

```bash
python run_pipeline.py --phase train --resume outputs/checkpoints/best_model.pth
```

## Dataset

Uses Chandrayaan-2 OHRC lunar imagery from Kaggle:
- **PSR patches**: 8,663 training, 1,529 validation (mean intensity: 0.028)
- **Sunlit patches**: 12,750 training, 2,250 validation (mean intensity: 0.437)
- **Mixed patches**: 11,372 training, 2,007 validation (contains boundaries)

### Data Splits (Crater-Based)

- **Train**: Shackleton-01 + Cabeus-01 (all classes)
- **Val**: Shackleton-02 (held-out crater, sun_elevation ≈ 0°)

## Model Architecture

- **Architecture**: U-Net (SMP)
- **Encoder**: ResNet18 (no pretrained weights)
- **Parameters**: ~12M
- **Input**: 64×64 grayscale patches
- **Output**: Binary segmentation masks

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| IoU (Jaccard) | > 0.70 | Intersection over Union |
| Dice (F1) | > 0.80 | F1 score |
| Pixel Accuracy | > 0.90 | Overall accuracy |
| HD95 | < 5 pixels | 95th percentile Hausdorff Distance |
| Boundary F1 | > 0.50 | F1 on boundary pixels (2px tolerance) |

## Ablation Studies

| Ablation | Change |
|----------|--------|
| No augmentation | Remove all augmentations |
| No CLAHE | Remove CLAHE only |
| Unweighted loss | Set pos_weight=1.0 |

## Citation

```bibtex
@article{psr_shadow_boundary_2026,
  title={Delineation of Shadow-Illumination Boundaries in Lunar Permanently 
         Shadowed Regions via Self-Supervised Segmentation from 
         Chandrayaan-2 OHRC Imagery},
  year={2026}
}
```

## License

ISRO Open Data Policy — free for research and educational use.
