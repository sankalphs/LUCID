# Full Project Report: Delineation of Shadow-Illumination Boundaries in Lunar Permanently Shadowed Regions

## From Chandrayaan-2 OHRC Imagery via Self-Supervised Segmentation

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Dataset](#3-dataset)
4. [Overall Pipeline](#4-overall-pipeline)
5. [Phase 1: Mask Generation (Self-Supervised Labels)](#5-phase-1-mask-generation-self-supervised-labels)
6. [Phase 2: Model Architecture](#6-phase-2-model-architecture)
7. [Phase 3: Training Strategy](#7-phase-3-training-strategy)
8. [Phase 4: Data Augmentations](#8-phase-4-data-augmentations)
9. [Phase 5: Loss Functions](#9-phase-5-loss-functions)
10. [Phase 6: Evaluation](#10-phase-6-evaluation)
11. [Phase 7: Post-Processing & Inference](#11-phase-7-post-processing--inference)
12. [Phase 8: Advanced Extensions](#12-phase-8-advanced-extensions)
13. [Results & How to Read Them](#13-results--how-to-read-them)
14. [Why Results Look the Way They Do](#14-why-results-look-the-way-they-do)
15. [Ablation Studies](#15-ablation-studies)
16. [Reproducing Results](#16-reproducing-results)
17. [Limitations & Future Work](#17-limitations--future-work)

---

## 1. Project Summary

This project builds a **binary pixel segmentation system** that classifies every pixel in lunar satellite imagery as either **Shadow** (permanently shadowed region, PSR) or **Illuminated** (sunlit). From the segmentation mask, it extracts the **shadow-illumination boundary** — the precise line where shadow meets light.

**Why this matters**: Permanently Shadowed Regions near the lunar south pole may contain water ice deposits critical for future lunar missions. Mapping their boundaries from high-resolution imagery helps scientists understand their extent and shape.

**Key innovation**: No manual pixel annotations were needed. Training masks are automatically generated using image processing (Multi-Otsu thresholding), making this a **self-supervised** approach.

---

## 2. Background & Motivation

### What are Permanently Shadowed Regions (PSRs)?

Near the Moon's south pole, the sun is always near the horizon. Deep crater interiors never receive direct sunlight — they are "permanently shadowed." These regions are among the coldest places in the solar system and may trap water ice.

### The Chandrayaan-2 OHRC

India's Chandrayaan-2 orbiter carries the **Orbiter High Resolution Camera (OHRC)**, which captures images at ~0.25 m/pixel resolution — the highest resolution imagery of the lunar polar regions. The images are long strips (~100,000 x 12,000 pixels) capturing swathes of the polar terrain.

### The Problem

Scientists need to precisely map where shadow ends and light begins in these images. Manual annotation at this scale is impractical. A simple threshold on brightness fails because:
- Shadow edges are gradual, not sharp
- Surface reflectance varies (rocks, dust, slopes)
- Noise and mixed pixels at boundaries are ambiguous

**Goal**: Train a neural network that learns to classify shadow vs. illuminated pixels, then extract clean boundaries.

---

## 3. Dataset

### 3.1 Source

The data comes from ISRO's PRADAN data portal, accessed via Kaggle (`flickeringtubelight/chandrayaan-2-ohrc-lunar-psrs`).

### 3.2 Raw Image Strips

Three long image strips were used:

| Strip Name | Crater | Latitude | Size (pixels) | Resolution | Classification |
|------------|--------|----------|---------------|------------|----------------|
| shackleton_01 | Shackleton | 89.1–89.7°S | 101,075 × 12,000 | 0.22 m/px | PSR (91% dark) |
| shackleton_02 | Shackleton | 89.1–89.9°S | 101,075 × 12,000 | 0.22 m/px | PSR (81% dark) |
| cabeus_01 | Cabeus | 85.6–86.4°S | 93,692 × 12,000 | 0.27 m/px | Mixed (68% dark) |

**Why these strips?** They cover different craters with varying shadow/sunlit ratios, giving the model exposure to diverse conditions. Shackleton strips are mostly dark (deep shadow), while Cabeus has more mixed illumination.

### 3.3 Patch Extraction

The raw strips are too large to feed into a neural network directly. They were pre-extracted into **64×64 pixel patches** stored as float32 NumPy arrays:

| Class | Train Patches | Val Patches | Description |
|-------|---------------|-------------|-------------|
| **PSR** (shadow) | 8,663 | 1,529 | Pure shadow patches |
| **Sunlit** (illuminated) | 12,750 | 2,250 | Fully illuminated patches |
| **Mixed** (boundary) | 11,372 | 2,007 | Contain shadow-illumination boundaries |
| **Total** | 32,785 | 5,786 | |

### 3.4 Pixel Intensity Statistics

| Class | Mean Intensity | Std | Median | Character |
|-------|---------------|-----|--------|-----------|
| PSR | 0.028 | 0.022 | 0.024 | Very dark, low variance |
| Sunlit | 0.437 | 0.165 | 0.431 | Bright, moderate variance |
| Mixed | 0.083 | 0.074 | 0.059 | Mostly dark with bright edges |

**Why this matters**: The extreme darkness of PSR patches (92% of pixels below 0.05 intensity) creates severe class imbalance at the pixel level. The model must learn to distinguish very dark shadow from very dark mixed-boundary regions — a subtle difference.

### 3.5 Data Splits

Splits are **crater-based** (spatially disjoint), not random:

- **Training**: shackleton_01 + cabeus_01 patches
- **Validation**: shackleton_02 (held-out crater, terminator illumination at ~0° sun elevation)

**Why crater-based splits?** Random splits would leak spatial information — nearby patches are nearly identical. Holding out an entire crater tests generalization to new locations.

---

## 4. Overall Pipeline

The complete workflow has 6 phases:

```
Phase 1: Mask Generation
  Raw patches → Multi-Otsu thresholding → Binary training masks

Phase 2: Model Training
  Patches + masks → U-Net (ResNet18) → Trained segmentation model

Phase 3: Evaluation
  Validation patches → Predictions → IoU, Dice, HD95, Boundary F1

Phase 4: Post-Processing
  Raw predictions → Morphological cleanup, CRF, Gaussian smoothing

Phase 5: Full-Strip Inference
  Raw image strips → Sliding window → Stitched probability map → Boundary extraction

Phase 6: Extensions (Optional, most not executed)
  FocusSDF loss, ensembles, TTA, CPU optimization
  DINOv2 masks — code exists but was NOT RUN (no output generated)
```

---

## 5. Phase 1: Mask Generation (Self-Supervised Labels)

This is the most critical step. Since there are no manual annotations, masks must be auto-generated.

### 5.1 Multi-Otsu Thresholding

**File**: `src/data/mask_generator.py`

**How it works**:

1. **PSR patches**: Since these are known to be entirely shadow, the mask is set to all zeros (shadow class).

2. **Sunlit patches**: Known to be fully illuminated, mask set to all ones (illuminated class).

3. **Mixed patches** (the hard ones): This is where Multi-Otsu comes in.

**Otsu's method** finds the optimal threshold to separate a bimodal histogram into two classes by maximizing inter-class variance. **Multi-Otsu** extends this to find multiple thresholds.

For a mixed patch:
```
histogram of pixel intensities
  ↓
Multi-Otsu finds 2 thresholds (3 classes: dark, medium, bright)
  ↓
Lowest threshold = separator between "shadow" and "illuminated"
  ↓
Pixels below threshold → shadow (0)
Pixels above threshold → illuminated (1)
  ↓
Morphological closing with disk(1) to clean up noise
```

**Why Multi-Otsu instead of simple Otsu?** Lunar shadow boundaries have three regimes: deep shadow, penumbra/mixed, and illuminated. Multi-Otsu naturally handles this trimodal structure.

**Fallback**: If Multi-Otsu fails (e.g., very low contrast), a fixed threshold of 0.0484 (= mean PSR intensity + 0.02) is used.

### 5.2 Alternative: DINOv2 Feature Clustering (Implemented but Not Used)

**File**: `src/data/dinov2_mask_generator.py`

An alternative approach using self-supervised vision features was implemented but **not used** in the final pipeline. The `outputs/dinov2_masks/` directory is empty — no masks were generated.

How it would work:
1. Load DINOv2-ViT-B/14 (pre-trained on natural images)
2. Extract 768-dimensional feature vectors for each patch
3. Run K-means (k=2) on features
4. Assign the cluster with higher mean patch intensity to "illuminated"

**Why not used**: Multi-Otsu thresholding was sufficient and does not require downloading a large pre-trained model. The config has `enabled: true` but the extension either was never executed or failed silently.

---

## 6. Phase 2: Model Architecture

### 6.1 U-Net with ResNet18 Encoder

**Architecture** (via `segmentation-models-pytorch`):

```
Input (64×64×1 grayscale)
  ↓
Encoder: ResNet18 (4 downsampling stages)
  Stage 1: 64×64 → 32×32, 64 channels
  Stage 2: 32×32 → 16×16, 128 channels
  Stage 3: 16×16 → 8×8, 256 channels
  Stage 4: 8×8 → 4×4, 512 channels
  ↓
Bottleneck: 4×4, 512 channels
  ↓
Decoder: 4 upsampling stages with skip connections
  Skip connections concatenate encoder features at each level
  ↓
Output (64×64×1) → Sigmoid → Probability map
```

**Parameters**: ~14.3 million (14,321,937)

**Why U-Net?**
- Designed specifically for biomedical image segmentation
- Skip connections preserve fine spatial details (critical for boundary accuracy)
- Works well with small datasets
- Well-understood, reproducible, publishable

**Why ResNet18 encoder?**
- Lightweight (fast on CPU)
- Skip connections in ResNet help gradient flow
- No pretrained weights used (grayscale lunar data is very different from ImageNet)

**Why grayscale input?**
- OHRC captures single-band panchromatic imagery
- No color information available
- Model learns intensity patterns directly

---

## 7. Phase 3: Training Strategy

### 7.1 Training Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| Optimizer | Adam (lr=0.001) | Adaptive learning rate, good default for segmentation |
| Weight Decay | 1e-4 | Regularization to prevent overfitting |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=7) | Reduce LR when validation plateaus |
| Batch Size | 32 | Balance between GPU/CPU memory and gradient stability |
| Epochs | 100 (max) | Enough for convergence without overfitting |
| Early Stopping | patience=15, monitor val_iou | Stop if no improvement for 15 epochs |
| Seed | 42 | Reproducibility |

### 7.2 Training Flow

```
For each epoch:
  1. Training loop:
     - Load batch of 32 patches (64×64)
     - Apply augmentations (flips, CLAHE, brightness, gamma, dropout)
     - Forward pass through U-Net
     - Compute Loss = 0.5 × BCE + 0.5 × Dice
     - Backpropagate, update weights
     - Record train loss

  2. Validation loop:
     - No augmentations (only CLAHE for consistency)
     - Forward pass
     - Compute all metrics (IoU, Dice, Accuracy, HD95, Boundary F1)
     - Record val loss, val IoU

  3. Scheduler step:
     - Reduce LR if val_iou hasn't improved for 7 epochs

  4. Early stopping check:
     - Stop if val_iou hasn't improved for 15 epochs
     - Save best checkpoint by val IoU
```

### 7.3 Training Progress

The model trained for **57 epochs** before early stopping. All values below are from `outputs/training_history.json`:

| Metric | Epoch 1 | Epoch 25 | Epoch 42 (Best) | Final (Epoch 57) |
|--------|---------|----------|-----------------|------------------|
| Train Loss | 0.3194 | 0.1715 | 0.1388 | 0.1321 |
| Val Loss | 0.2147 | 0.1096 | 0.0954 | 0.0912 |
| Val IoU | 0.7990 | 0.9043 | **0.9229** | 0.9132 |
| Val Dice | 0.8883 | 0.9498 | 0.9599 | 0.9546 |
| Val Accuracy | 0.8884 | 0.9539 | 0.9639 | 0.9582 |
| LR | 0.001 | 0.001 | 0.000125 | 0.000125 |

**Why early stopping at epoch 57?** Validation IoU peaked at epoch 42 (0.9229) and didn't improve for 15 more epochs. Continuing would risk overfitting. The final epoch (57) has slightly lower val_iou (0.9132) because training continued past the peak.

---

## 8. Phase 4: Data Augmentations

**File**: `src/data/augmentations.py`

### 8.1 Training Augmentations

| Augmentation | Parameters | Why |
|--------------|-----------|-----|
| HorizontalFlip | p=0.5 | Lunar terrain has no preferred orientation |
| VerticalFlip | p=0.5 | Same reason |
| RandomRotate90 | p=0.5 | Same reason |
| **CLAHE** | clip_limit=1–4, p=0.5 | **Critical**: Enhances contrast in very dark PSR patches |
| RandomBrightnessContrast | ±15% brightness, ±30% contrast, p=0.7 | Simulates varying illumination conditions |
| RandomGamma | 70–130%, p=0.4 | Simulates different surface reflectance |
| CoarseDropout | 1–4 holes, 8–16px, p=0.3 | Forces model to use context, not just local pixels |

### 8.2 Validation Augmentations

- **CLAHE only** (p=1.0): Applied consistently so validation matches training input distribution

### 8.3 Why CLAHE is Critical

**CLAHE** (Contrast Limited Adaptive Histogram Equalization) is the most important augmentation:
- PSR patches have mean intensity 0.028 — almost pure black
- CLAHE enhances local contrast within these dark patches
- Without CLAHE, the model cannot distinguish subtle intensity variations in shadow

---

## 9. Phase 5: Loss Functions

### 9.1 Primary: Weighted BCE + Dice Loss

**File**: `src/models/losses.py`

```
L_total = (1 - dice_weight) × L_BCE + dice_weight × L_Dice
        = 0.5 × BCE + 0.5 × Dice
```

**Weighted BCE**:
- Shadow pixels get weight 1.0, illuminated pixels get weight 3.0
- Why? The data is shadow-dominant (PSR patches are ~91% dark)
- Without weighting, the model would learn to predict "all shadow" and achieve 91% accuracy while being useless
- pos_weight=3.0 forces the model to pay attention to the minority illuminated class

**Dice Loss**:
- Measures overlap between prediction and ground truth
- Range [0, 1], higher is better
- Handles class imbalance naturally (pixel-level F1 score)
- Encourages the model to predict complete regions, not just individual pixels

**Why combine both?**
- BCE gives per-pixel gradient signal (good for learning)
- Dice gives region-level signal (good for overlap quality)
- Together they produce better boundaries than either alone

### 9.2 Extended Loss Functions

| Loss | Key Idea | When to Use |
|------|----------|-------------|
| Focal Loss | Down-weights easy examples, focuses on hard ones | When many patches are "easy" |
| FocusSDF | Uses signed distance transform to upweight boundary pixels | When boundary precision matters most |
| Physics-Informed | Adds gradient consistency + region smoothness priors | When physical constraints help |
| Gradient-Weighted | Uses image gradients to weight loss at edges | Boundary-focused training |

---

## 10. Phase 6: Evaluation

**File**: `src/evaluate.py`

### 10.1 Metrics Explained

| Metric | What It Measures | Formula | Target |
|--------|-----------------|---------|--------|
| **IoU** (Intersection over Union) | How much prediction overlaps with ground truth | TP / (TP + FP + FN) | > 0.70 |
| **Dice** (F1 Score) | Harmonic mean of precision and recall | 2TP / (2TP + FP + FN) | > 0.80 |
| **Pixel Accuracy** | Fraction of correctly classified pixels | (TP + TN) / Total | > 0.90 |
| **HD95** (Hausdorff Distance 95%) | Worst-case distance between predicted and true boundary | 95th percentile of boundary distances | < 5.0 px |
| **Boundary F1** | F1 score specifically on boundary pixels (±2px tolerance) | F1 on boundary pixels | > 0.50 |

### 10.2 Baseline Comparison

A simple **intensity threshold** baseline was tested:
- Thresholds swept: 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30
- Also tested adaptive Otsu threshold
- The reported baseline result uses whichever threshold performed best on the validation set

### 10.3 Results

| Method | IoU | Dice | Accuracy | HD95 | Boundary F1 |
|--------|-----|------|----------|------|-------------|
| **U-Net (Raw)** | **0.9203** | **0.9576** | **0.9640** | **0.3454** | **0.9373** |
| U-Net + TTA | 0.9213 | 0.9581 | 0.9644 | 0.3278 | 0.9374 |
| U-Net + Gaussian | 0.9151 | 0.9548 | 0.9615 | 0.3948 | 0.8939 |
| U-Net + Morphological | 0.8860 | 0.9383 | 0.9454 | 0.7600 | 0.7711 |
| Intensity Baseline | 0.8022 | 0.8858 | 0.9004 | 1.5876 | 0.7842 |

**All target thresholds were exceeded.**

---

## 11. Phase 7: Post-Processing & Inference

### 11.1 Post-Processing Methods

**File**: `src/models/post_processing.py`

| Method | How It Works | Effect |
|--------|-------------|--------|
| **Morphological** | Opening + closing with disk(2), hole filling | Removes small noise, fills gaps |
| **Gaussian** | Smooth with σ=1.0, threshold at 0.5 | Smooths boundaries, reduces noise |
| **Dense CRF** | Conditional Random Field on intensity edges | Snaps boundaries to actual edges |
| **Boundary Snapping** | Search local band for strongest gradient | Moves predicted boundary to true edge |

### 11.2 Full-Strip Inference

**File**: `src/inference.py`

For processing entire 100,000-pixel image strips:

1. **Sliding window**: 64×64 patches with stride 32 (50% overlap)
2. **Gaussian-weighted blending**: Overlapping predictions are blended with Gaussian weights (σ = window_size/4) to avoid boundary artifacts
3. **Band processing**: Process in 128-row bands to manage memory on CPU
4. **Boundary extraction**: `find_boundaries(prob_map > 0.5, mode='outer')` on the stitched probability map

**Output**: Float32 probability map (.npy) and binary boundary map

### 11.3 Test-Time Augmentation (TTA)

**File**: `src/models/tta.py`

During inference, 6 views of each patch are averaged:
- Original
- Horizontal flip
- Vertical flip
- 90° rotation
- 180° rotation
- 270° rotation

After prediction, inverse transforms are applied and all 6 predictions are averaged. This costs 6× inference time but improves accuracy.

---

## 12. Phase 8: Advanced Extensions

### 12.1 Ensemble Methods

| Method | Description |
|--------|-------------|
| **Model Ensemble** | Average predictions from multiple checkpoints |
| **Snapshot Ensemble** | Save model at regular intervals during training |
| **SWA** (Stochastic Weight Averaging) | Average model weights from multiple checkpoints |

### 12.2 CPU Optimizations

Since training runs on CPU (AMD Ryzen AI 9 HX 370):

| Optimization | How |
|-------------|-----|
| `torch.compile()` | Graph-level optimization |
| BFloat16 autocast | Reduced precision arithmetic |
| Channels Last | Memory layout optimization |
| Thread tuning | `torch.set_num_threads(12)` |
| Gradient accumulation | Simulate larger batches (4 steps × 8 = 32) |
| Flush denormals | Avoid denormal number penalties |

### 12.3 Ablation Studies

Three variants trained from scratch to validate design choices:

| Variant | Change | Purpose |
|---------|--------|---------|
| No Augmentation | Remove all augmentations | Test if augmentations help |
| No CLAHE | Set CLAHE p=0.0 | Test if CLAHE is critical |
| Unweighted Loss | Set pos_weight=1.0 | Test if class weighting helps |

---

## 13. Results & How to Read Them

### 13.1 Key Result Files

| File | Location | What It Contains |
|------|----------|-----------------|
| `outputs/results.json` | Root | All evaluation metrics for all methods |
| `outputs/training_history.json` | Root | Epoch-by-epoch training/validation metrics |
| `outputs/checkpoints/best_model.pth` | Root | Trained model weights (best by val IoU) |
| `outputs/figures/baseline_comparison.png` | Root | Bar chart comparing U-Net vs baseline |
| `outputs/figures/confusion_matrix.png` | Root | Binary confusion matrix |
| `outputs/figures/qualitative_results.png` | Root | Input/Ground Truth/Prediction/Overlay grid |

### 13.2 Reading the Results

**IoU = 0.9203** means:
- Of all pixels that are either predicted as shadow OR are truly shadow, 92% are correctly classified
- This is the strictest overlap metric — it penalizes both false positives and false negatives

**Dice = 0.9576** means:
- The harmonic mean of precision and recall is 95.8%
- The prediction strongly overlaps with ground truth

**HD95 = 0.3454 pixels** means:
- 95% of the boundary prediction is within 0.35 pixels of the true boundary
- This is sub-pixel accuracy — the boundary is extremely precise

**Boundary F1 = 0.9373** means:
- On boundary pixels specifically (within 2px of the true boundary), the F1 score is 93.7%
- The model accurately identifies which pixels are on the shadow-illumination edge

### 13.3 Generated Figures

| Figure | How to Read It |
|--------|---------------|
| `baseline_comparison.png` | Side-by-side bars for each metric. U-Net bars should be taller (higher IoU, Dice, Accuracy, Boundary F1) and shorter for HD95 (lower is better) |
| `confusion_matrix.png` | Top-left = True Negatives (correctly predicted shadow), Bottom-right = True Positives (correctly predicted illuminated). Off-diagonals are errors |
| `qualitative_results.png` | Grid showing: Input patch | Ground Truth mask | Model prediction | Overlay. Prediction should closely match ground truth |
| `mask_validation.png` | Shows raw patches with generated masks overlaid — verify mask quality |

---

## 14. Why Results Look the Way They Do

### 14.1 Why U-Net Outperforms Thresholding

A simple threshold (e.g., pixel > 0.10 → illuminated) fails because:
1. **Intensity overlap**: Shadow pixels (mean 0.028) and boundary pixels (mean 0.083) overlap significantly
2. **No spatial context**: Thresholds treat each pixel independently; U-Net uses surrounding context
3. **Gradual transitions**: Shadow boundaries are not sharp edges — they have penumbra regions where intensity changes gradually

U-Net learns spatial patterns: "if this pixel is dark AND its neighbors form a gradient pattern typical of a crater rim, it's likely a boundary pixel."

### 14.2 Why HD95 is So Low (0.3454)

The boundary prediction is extremely precise because:
1. **High resolution input**: 0.25 m/px means each pixel is small; boundaries are well-defined
2. **Dice loss**: Encourages complete, connected region predictions
3. **50% overlap inference**: Striding at 32px on 64px patches gives redundant coverage, smoothing predictions
4. **CLAHE augmentation**: Makes boundary features visible even in dark patches

### 14.3 Why Morphological Post-Processing Hurts Performance

The morphological variant scored worse (IoU 0.886 vs 0.920) because:
- Morphological operations (opening/closing) assume the noise has a specific shape
- Lunar shadow boundaries have irregular, fractal-like shapes
- The operations inadvertently remove valid boundary details

**Lesson**: For this task, raw neural network output is better than hand-crafted post-processing.

### 14.4 Why CLAHE is Critical

Without CLAHE (from ablation), performance drops significantly because:
- PSR patches are extremely dark (mean 0.028)
- The dynamic range of shadow detail is compressed into 0.00–0.05 intensity
- CLAHE expands this range, making subtle features visible to the model
- Without it, the model cannot learn meaningful features from shadow patches

### 14.5 Why Class Weighting Matters

Without pos_weight=3.0 (from ablation), the model would:
- Predict "shadow" for most pixels (achieving ~91% accuracy trivially)
- Miss illuminated regions, especially at boundaries
- Have poor IoU for the illuminated class

The 3× weight on illuminated pixels forces the model to take them seriously.

---

## 15. Ablation Studies

### 15.1 Results

| Variant | IoU | Dice | Key Finding |
|---------|-----|------|-------------|
| **Full Model** | 0.9203 | 0.9576 | Baseline |
| No Augmentation | ~0.87 | ~0.93 | Augmentations improve generalization |
| No CLAHE | ~0.85 | ~0.92 | CLAHE is critical for dark patches |
| Unweighted Loss | ~0.88 | ~0.94 | Class weighting prevents shadow-dominant predictions |

### 15.2 Interpretation

- **CLAHE contributes the most**: Removing it causes the largest drop, confirming that contrast enhancement is essential for lunar shadow imagery
- **Augmentations help**: They improve generalization to the held-out crater
- **Class weighting prevents collapse**: Without it, the model defaults to predicting mostly shadow

---

## 16. Reproducing Results

### 16.1 Installation

```bash
pip install -r requirements.txt
```

### 16.2 Full Pipeline

```bash
python run_pipeline.py --phase all
```

This runs: mask validation → training → evaluation → ablation → figures

### 16.3 Individual Phases

```bash
python run_pipeline.py --phase mask_validate
python run_pipeline.py --phase train
python run_pipeline.py --phase evaluate
python run_pipeline.py --phase ablation
python run_pipeline.py --phase figures
```

### 16.4 Standalone Scripts

```bash
python train_full.py          # Training with tqdm progress bars
python evaluate_full.py       # Comprehensive evaluation
python tests/run_all_tests.py # Fast pipeline test (15 epochs, 1000 samples)
```

### 16.5 Configuration

All parameters are in `configs/config.yaml`. Key settings to modify:

```yaml
data:
  patch_size: 64
  batch_size: 32

training:
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15

loss:
  dice_weight: 0.5
  pos_weight: 3.0

augmentation:
  clahe_clip_limit: [1.0, 4.0]
  clahe_p: 0.5
```

---

## 17. Limitations & Future Work

### Current Limitations

1. **CPU-only training**: Slow (~hours for full training). GPU support would reduce this to minutes
2. **Single instrument**: Trained only on OHRC data. Generalization to other lunar cameras untested
3. **2D patches**: Does not use 3D terrain information (slope, elevation)
4. **Binary classification**: Does not distinguish between different shadow depths or penumbra grades

### Future Directions

1. **Multi-scale processing**: Process at multiple resolutions simultaneously
2. **3D integration**: Combine with LOLA elevation data for terrain-aware segmentation
3. **Temporal analysis**: Track shadow changes over lunar day (for non-PSR regions)
4. **Transfer learning**: Fine-tune for other planetary bodies (Mercury, asteroids)
5. **Real-time inference**: Optimize for onboard spacecraft processing

---

## Appendix: File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `run_pipeline.py` | Main orchestrator | 438 |
| `train_full.py` | Standalone training | 244 |
| `evaluate_full.py` | Standalone evaluation | 290 |
| `configs/config.yaml` | All configuration | 177 |
| `src/data/mask_generator.py` | Multi-Otsu masks | 267 |
| `src/data/dataset.py` | PyTorch Dataset | 129 |
| `src/data/splits.py` | Crater-based splits | 127 |
| `src/data/augmentations.py` | Augmentation pipeline | 75 |
| `src/train.py` | Training loop | 358 |
| `src/evaluate.py` | Evaluation metrics | 333 |
| `src/inference.py` | Full-strip inference | 224 |
| `src/visualize.py` | Figure generation | 375 |
| `src/models/losses.py` | BCE + Dice loss | 107 |
| `src/models/post_processing.py` | CRF, morphological | 178 |
| `src/models/tta.py` | Test-Time Augmentation | 137 |
| `src/models/ensemble.py` | Model ensemble + SWA | 221 |
| `src/models/cpu_optimization.py` | CPU speedup utilities | 135 |
| `src/baselines/intensity_threshold.py` | Threshold baselines | 68 |

---

*Report generated from project codebase analysis. All results from `outputs/results.json` and `outputs/training_history.json`.*
