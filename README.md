# LUCID - Lunar Unsupervised Classification of Illumination and Darkness

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-ee4c2c.svg)](https://pytorch.org/)
[![Status: Research Code](https://img.shields.io/badge/status-research--code-orange.svg)](#status)

Self-supervised segmentation of shadow vs illuminated terrain in lunar PSRs from Chandrayaan-2 OHRC imagery. Masks are auto-generated with Multi-Otsu thresholding. A U-Net (ResNet-18) learns the boundary.

## Status

Open-source research code under MIT. Original journal plan was paused after ablations did not support all draft claims. Code is released as-is with full transparency.

| What | Detail |
|------|--------|
| Core result (reproducible) | U-Net IoU **0.9202** on legacy mixed split (11,372 train / 2,007 val, seed 42). 5-seed mean **0.9217 +/- 0.0035**. Measures pseudo-label agreement, not geological truth |
| Honest splits | `train.npy` contains ~4,250 dupe patches from `shackleton_02`. Use rebuilt per-strip splits for generalization (IoU drops ~1.1 pts, see `docs/audit/FINAL_FINDINGS.md`) |
| Ablations | Full-data: no-CLAHE -5.1 pts, no-augmentation -6.9 pts, `pos_weight=3` hurts vs 1.0 |
| Fallback 0.0484 | Never fires on real mixed patches (dead code) |
| Labels | No expert annotations. 80-patch blinded export ready in `results/expert_evaluation/` |

## Key Features

| Feature | Description |
|---------|-------------|
| Self-supervised labels | Multi-Otsu (3 classes) on mixed patches, fixed masks for PSR/sunlit |
| Model | SMP U-Net, ResNet-18, 1-channel, ~14M params, no pretraining |
| Evaluation | IoU, Dice, pixel accuracy, HD95, Boundary F1 (2px) |
| Inference | 64x64 sliding window, stride 32, Gaussian blending, banded for CPU |
| Experiments | 5-seed, cross-region, U-Net vs U-Net++ vs DeepLabV3+, classical baselines |

## Dataset

| Item | Detail |
|------|--------|
| Source | ISRO PRADAN via Kaggle `flickeringtubelight/chandrayaan-2-ohrc-lunar-psrs` (OHRC 0.22-0.27 m/px) |
| Location | `dataset/kaggle_dataset/` (not tracked, download separately) |
| Patches (train/val) | PSR 8,663/1,529, Sunlit 12,750/2,250, Mixed 11,372/2,007 |
| Mean intensity | PSR 0.028, Sunlit 0.437, Mixed 0.083 |
| Layout | `patches/{psr,sunlit,mixed}/{train,val,shackleton_01,shackleton_02,cabeus_01}.npy` and `raw/{strip}/{image.img,label.xml,geometry.csv}` |

Download and place as shown above. Config `configs/config.yaml:data.base_dir` defaults to `dataset/kaggle_dataset` (relative).

## Installation

| Step | Command |
|------|---------|
| Clone | `git clone https://github.com/sankalphs/ch-2_OHRC_PSRs.git && cd ch-2_OHRC_PSRs` |
| PyTorch CPU | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| Deps | `pip install -r requirements.txt` |
| Exact repro | `pip install -r requirements-lock.txt` (Python 3.13.15, torch 2.13.0+cpu) |

Requires Python 3.11+.

## Quick Start

| Task | Command |
|------|---------|
| Full pipeline | `python run_pipeline.py --phase all` |
| Validate masks | `python run_pipeline.py --phase validate` |
| Train | `python run_pipeline.py --phase train` |
| Evaluate | `python run_pipeline.py --phase evaluate` |
| Ablations | `python run_pipeline.py --phase ablation` |
| Figures | `python run_pipeline.py --phase figures` |
| Resume | `python run_pipeline.py --phase train --resume outputs/checkpoints/best_model.pth` |

Standalone (in `scripts/`):

| Script | Purpose |
|--------|---------|
| `python scripts/train_full.py` | Training with progress bars |
| `python scripts/evaluate_full.py` | Full evaluation |
| `python scripts/run_baselines.py` | Otsu / Adaptive / RF baselines |
| `python scripts/run_ablations.py` | Ablation matrix |

Reproducibility suite:

| Task | Command |
|------|---------|
| List experiments | `python experiments/run_all_publication_experiments.py --list` |
| Full suite (days) | `python experiments/run_all_publication_experiments.py` |
| Subset | `python experiments/run_all_publication_experiments.py --only reproduction` |

See `docs/REPRODUCIBILITY.md` for full command table.

## Model and Training

| Component | Setting |
|-----------|---------|
| Architecture | `smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=1, activation=None)` |
| Loss | `(1-a)*WeightedBCE(pos_weight) + a*Dice`, default `pos_weight=3.0, dice_weight=0.5` |
| Augmentation | HorizontalFlip, VerticalFlip, RandomRotate90 (p=0.5), CLAHE (1-4, p=0.5), RandomBrightnessContrast, RandomGamma, CoarseDropout |
| Optimizer | Adam lr 1e-3, weight_decay 1e-4, ReduceLROnPlateau (factor 0.5, patience 7), early stop patience 15 on val IoU |
| Training | Batch 32, max 100 epochs, seed 42, CPU-only |

## Results

| Split | Model | IoU | Dice | Acc | HD95 | BF1 |
|-------|-------|-----|------|-----|------|-----|
| legacy mixed (leaky) | U-Net R18 seed 42 | 0.9202 | 0.9574 | 0.9636 | 0.345 | 0.9391 |
| legacy mixed | U-Net R18 5-seed mean | 0.9217 +/- 0.0035 | 0.9583 | 0.9646 | 0.338 | 0.9418 |
| rebuilt B (sh01+sh02 -> cabeus) | U-Net R18 | 0.9092 | 0.9513 | 0.9592 | 0.478 | 0.9087 |
| rebuilt C (cabeus -> sh01+sh02) | U-Net R18 | 0.9098 | 0.9516 | 0.9579 | 0.424 | 0.9190 |
| legacy mixed | Random Forest | 0.7964 | 0.8787 | 0.9034 | 1.648 | 0.8189 |

All metrics are agreement with Multi-Otsu pseudo-labels. Architecture ranking (mixed-only): U-Net++ 0.9228 > U-Net 0.9202 > DeepLabV3+ 0.8904.

## Reproducibility

| Item | Location |
|------|----------|
| Provenance | `docs/REPRODUCIBILITY.md` |
| Audit | `docs/audit/FINAL_FINDINGS.md` (8 issues, corrected tables) |
| Ablations | `docs/audit/ABLATION_FINDINGS.md` |
| Checksums | `results/data_checksums.sha256` |
| Env | `requirements-lock.txt` |

Expect small IoU jitter across machines (oneDNN, stochastic CLAHE).

## License

MIT - see [LICENSE](LICENSE). Copyright (c) 2026 Sankalp H S. Imagery is ISRO Open Data Policy and not redistributed; obtain via PRADAN/Kaggle.
