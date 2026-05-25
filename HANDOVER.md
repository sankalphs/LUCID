# Handover Document: PSR Shadow Boundary Segmentation Project

## Project Overview

**Goal**: Build a publishable binary segmentation system that classifies each pixel in Chandrayaan-2 OHRC lunar imagery as either **shadow** or **illuminated**, then extract the shadow-illumination boundary as a post-processing step.

**Paper Title**: "Delineation of Shadow-Illumination Boundaries in Lunar Permanently Shadowed Regions via Self-Supervised Segmentation from Chandrayaan-2 OHRC Imagery"

**Framing**: The system performs binary pixel segmentation (shadow vs. illuminated). The "boundary" is defined as the morphological edge between predicted shadow and illuminated regions (using `skimage.segmentation.find_boundaries`). This is consistent with the self-supervised labeling strategy.

---

## Novelty Status

**As of May 2026, we found no directly comparable prior work** in our search of Google Scholar, arXiv, and GitHub. Specifically, we found no published method that performs pixel-level segmentation of shadow-illumination boundaries in lunar PSR imagery using deep learning on orbital data.

**Search protocol**:
- Google Scholar: queries including `"permanently shadowed region" segmentation deep learning`, `Chandrayaan OHRC shadow segmentation`, `lunar shadow boundary detection`
- arXiv: queries including `lunar permanently shadowed region segmentation`, `shadow detection deep learning 2025 2026`
- GitHub: `chandrayaan OHRC PSR segmentation`, `lunar PSR shadow segmentation`
- Date range: 2023–2026, with focus on 2025–2026
- Date of search: May 2026

**Closest adjacent work:**

| Paper | Year | What They Do | Why We're Different |
|-------|------|--------------|---------------------|
| LunarS2O (Xia et al., Icarus) | 2026 | SAR-to-optical translation for PSRs | Generates images, doesn't segment boundaries |
| Vijayan et al. (Nature) | 2026 | Maps PSR crater populations using OHRC | Uses modeled PSR boundaries (LOLA/Diviner), not learned from OHRC |
| Pan et al. (IEEE) | 2025 | PSR image denoising | Enhancement, not segmentation |
| Cloud et al. (IEEE) | 2025 | Instance segmentation for hazards (rocks/craters) | Object detection, not boundary delineation |
| Gaur et al. (IEEE) | 2024 | Crater detection on OHRC | Object detection, not segmentation |

**Paper framing**:
> "Existing deep learning approaches for PSR imagery focus on image enhancement and object detection. We address the complementary problem of spatially delineating the shadow-illumination boundary using a self-supervised pipeline that requires no manual pixel annotations."

---

## Dataset Location & Structure

```
D:\Coding\ch-2_OHRC_PSRs\dataset\kaggle_dataset\
├── patches/                         # Pre-extracted 64×64 float32 patches
│   ├── psr/
│   │   ├── train.npy                # (8663, 64, 64)
│   │   ├── val.npy                  # (1529, 64, 64)
│   │   ├── shackleton_01.npy        # (3497, 64, 64)
│   │   ├── shackleton_02.npy        # (5000, 64, 64)
│   │   └── cabeus_01.npy            # (1695, 64, 64)
│   ├── sunlit/
│   │   ├── train.npy                # (12750, 64, 64)
│   │   ├── val.npy                  # (2250, 64, 64)
│   │   ├── shackleton_01.npy        # (5000, 64, 64)
│   │   ├── shackleton_02.npy        # (5000, 64, 64)
│   │   └── cabeus_01.npy            # (5000, 64, 64)
│   └── mixed/
│       ├── train.npy                # (11372, 64, 64)
│       ├── val.npy                  # (2007, 64, 64)
│       ├── shackleton_01.npy        # (5000, 64, 64)
│       ├── shackleton_02.npy        # (5000, 64, 64)
│       └── cabeus_01.npy            # (3379, 64, 64)
└── raw/                             # Full-resolution image strips
    ├── shackleton_01/               # 101075×12000, uint8, 0.22 m/px
    ├── shackleton_02/               # 101075×12000, uint8, 0.22 m/px
    └── cabeus_01/                   # 93692×12000, uint8, 0.27 m/px
```

Each raw directory contains: `image.img`, `label.xml`, `geometry.csv`, `browse.png`.

---

## Data Characteristics

### Pixel Intensity Statistics

| Class   | Train Count | Mean   | Std    | Median |
|---------|-------------|--------|--------|--------|
| PSR     | 8,663       | 0.0284 | 0.0215 | 0.024  |
| Sunlit  | 12,750      | 0.4365 | 0.1645 | 0.431  |
| Mixed   | 11,372      | 0.0832 | 0.0743 | 0.059  |

- PSR patches are extremely dark (mean=0.028, ~97% of pixels below 0.05)
- Mixed patches are predominantly dark but with higher variance; only ~8% have std > 0.1
- Mixed patches contain both shadow and illuminated pixels — the boundary transition zone

### Caveats

- **shackleton_02**: sun_elevation ≈ 0° (terminator). Use as **test strip only**; report metrics separately.
- **geometry.csv**: sampled every 100px (22m spacing). If reporting lat/lon of boundaries, interpolate and acknowledge ±11m error.

---

## Hardware

```
CPU:    AMD Ryzen AI 9 HX 370 (12 cores / 24 threads)
GPU:    AMD Radeon 890M (integrated) — no CUDA on Windows
RAM:    ~24 GB
```

---

## CORE PIPELINE

This is the minimum viable pipeline. Implement and validate everything here before considering optional extensions.

### Step 1: Mask Generation (Multi-Otsu)

Auto-generate pixel-level binary masks from patch-level labels.

```python
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.morphology import closing, disk

PSR_MEAN = 0.0284
FALLBACK_THRESHOLD = PSR_MEAN + 0.02  # = 0.0484, data-derived

def generate_mask(patch, class_label):
    if class_label == 'psr':
        return np.zeros_like(patch, dtype=np.float32), 'psr_fixed'
    elif class_label == 'sunlit':
        return np.ones_like(patch, dtype=np.float32), 'sunlit_fixed'
    else:
        try:
            thresholds = threshold_multiotsu(patch, classes=3)
            mask = (patch > thresholds[0]).astype(np.float32)
            return mask, 'multi_otsu'
        except ValueError:
            return (patch > FALLBACK_THRESHOLD).astype(np.float32), 'fallback'

def clean_mask(mask):
    return closing(mask, disk(1)).astype(np.float32)
```

### Step 2: Mask Quality Validation

Before training, validate the auto-generated masks:

1. Visualize 50 random masks overlaid on raw patches → save to `outputs/figures/mask_validation/`
2. Log fallback frequency (how often `threshold_multiotsu` raises `ValueError`)
3. Log threshold distribution across mixed patches
4. If fallback rate > 20%, filter out mixed patches with std < 0.03

### Step 3: Data Split (Crater-Based)

Spatially disjoint split to prevent leakage:

- **Train**: `shackleton_01` + `cabeus_01` patches (all 3 classes)
- **Val**: `shackleton_02` patches (held-out crater)
- **Test**: Per-crater metrics reported separately

### Step 4: Model (SMP U-Net)

Use `segmentation-models-pytorch` for a clean, well-maintained implementation:

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,  # no pretrained weights for grayscale lunar data
    in_channels=1,
    classes=1,
    activation="sigmoid"
)
```

This gives ~12M parameters with ResNet18 encoder. If too large, use `"mobilenet_v2"` encoder (~2M params).

**Why SMP over custom U-Net**: well-tested, maintained, easy to swap encoders for ablation.

### Step 5: Loss Function (Weighted BCE + Dice)

Standard combination that handles class imbalance:

```python
import torch
import torch.nn.functional as F

class WeightedBCEDiceLoss:
    def __init__(self, pos_weight=3.0, dice_weight=0.5):
        self.pos_weight = pos_weight  # weight for illuminated class
        self.dice_weight = dice_weight

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy(
            pred, target,
            weight=torch.where(target == 1, self.pos_weight, 1.0)
        )
        smooth = 1.0
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return (1 - self.dice_weight) * bce + self.dice_weight * dice
```

Compute `pos_weight` from actual mask statistics after Step 2 (e.g., if 75% shadow, 25% illuminated → pos_weight = 3.0).

### Step 6: Augmentation (Albumentations)

```python
import albumentations as A

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.15, 0.15),
        contrast_limit=(-0.3, 0.3),
        p=0.7
    ),
    A.RandomGamma(gamma_limit=(70, 130), p=0.4),
    A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(8, 16),
        hole_width_range=(8, 16),
        fill_value=0,
        p=0.3
    ),
])
```

CLAHE is included because PSR patches have extremely low contrast (mean=0.028). The other augmentations are standard.

### Step 7: Intensity Baseline

Before training the U-Net, establish a trivial baseline:

```python
def intensity_baseline(patch, threshold=0.1):
    return (patch > threshold).astype(np.float32)
```

Evaluate on the same val set. The U-Net must beat this.

### Step 8: Training

```python
# Seed everything
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# Adam optimizer, ReduceLROnPlateau scheduler
# Early stopping on val IoU, patience=15
# Log train/val loss and IoU per epoch
# Save best checkpoint by val IoU
```

### Step 9: Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| IoU (Jaccard) | TP / (TP + FP + FN) | > 0.70 |
| Dice (F1) | 2TP / (2TP + FP + FN) | > 0.80 |
| Pixel Accuracy | (TP + TN) / Total | > 0.90 |
| HD95 | 95th percentile Hausdorff Distance | < 5 pixels |
| Boundary F1 | F1 on pixels within 2px of predicted boundary | > 0.50 |

Report per-crater metrics separately. Compare against intensity baseline.

### Step 10: Boundary Extraction

```python
from skimage.segmentation import find_boundaries

binary_mask = (prob_map > 0.5).astype(np.uint8)
boundary = find_boundaries(binary_mask, mode='outer')
```

### Step 11: Ablation

Run at least one ablation to demonstrate a component matters:

| Ablation | Change |
|----------|--------|
| No augmentation | Remove all augmentations |
| No CLAHE | Remove CLAHE only |
| Unweighted loss | Set pos_weight=1.0 |

### Step 12: Full-Strip Inference

Sliding window on raw images:

```
Window: 64×64, Stride: 32 (50% overlap)
Gaussian weighting for overlap blending
Process in 128-row bands (never load full image)
Output: float32 .npy probability map
Boundary: find_boundaries(prob_map > 0.5)
Georeferencing: geometry.csv (bilinearly interpolate, ±11m error)
```

---

## OPTIONAL EXTENSIONS

Implement these only after the core pipeline is working and evaluated. Each is independent — pick based on time and results.

### Extension A: DINOv2 Mask Generation

If Multi-Otsu masks look poor on visual inspection (Step 2), try DINOv2 feature clustering as an alternative:

- Extract DINOv2-ViT-B/14 features from each mixed patch (replicate grayscale to 3-channel)
- K-means clustering (k=2) on feature vectors
- Assign cluster with higher mean intensity as "illuminated"
- CPU-feasible: ~3-6 hours for 11K patches as one-time preprocessing
- Compare IoU between DINOv2 masks and Multi-Otsu masks — report as ablation

### Extension B: Boundary-Aware Loss (FocusSDF)

If the core model produces blurry or inaccurate boundaries, try adding FocusSDF (arXiv:2511.11864):

```python
# FocusSDF: adaptive weighting by signed distance to boundary
# L_total = 0.5 * Dice + 0.5 * FocusSDF
```

This is a drop-in loss replacement. No architectural changes needed. Test whether it improves HD95 over the core Weighted BCE + Dice.

### Extension C: Physics-Informed Loss

If boundaries are fragmented or noisy, try adding Physics-Informed Loss (arXiv:2511.20501, GitHub: irfantahir301/Physicsis_loss) as regularization:

```python
# L_total = 0.4 * Dice + 0.4 * BCE + 0.2 * PIL
```

### Extension D: Post-Processing (Dense CRF)

If predicted boundaries don't align well with intensity edges, try Dense CRF:

- Install: build from source at https://github.com/lucasb-eyer/pydensecrf (Windows may require Visual C++ build tools)
- Apply after model prediction, before boundary extraction
- Uses intensity edges in the original image to snap predicted boundaries

Note: Installation on Windows can be non-trivial. If build fails, skip this extension.

### Extension E: Test-Time Augmentation (TTA)

Apply geometric TTA during evaluation/inference for free quality improvement:

```python
def tta_predict(model, patch):
    # 6 views: original + flips + rotations
    # Average predictions after inverse transforms
    ...
```

Zero additional training cost. Only increases inference time.

### Extension F: Ensemble (K-Fold Snapshot + SWA)

If the core model is working well and you have unlimited training time:

- 5-fold cross-validation with cosine annealing warm restarts
- Collect 4 snapshots per fold + 1 SWA model = 25 diverse models
- Average predictions at inference
- Expected: smoother predictions, better calibration

### Extension G: Multi-Scale TTA

Run inference at multiple scales (0.75×, 1.0×, 1.25×) and average. Combined with geometric TTA, this gives 30 views per patch from a single trained model.

### Extension H: CPU Optimization

Apply if training is too slow:

| Technique | Speedup | Effort |
|-----------|---------|--------|
| `torch.compile(model)` | 2-4× | One line |
| BFloat16 autocast | 1.5-3× | Wrap forward pass |
| Channels Last format | 1.2-1.8× | Convert model + data |
| `torch.set_flush_denormal(True)` | 1.1-1.5× | One line |
| Thread tuning: `torch.set_num_threads(12)` | 1.5-3× | One line |
| Gradient accumulation | Effective batch scaling | Training loop change |

---

## File Structure

```
D:\Coding\ch-2_OHRC_PSRs\
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── augmentations.py
│   │   ├── mask_generator.py
│   │   └── splits.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── losses.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── intensity_threshold.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── visualize.py
├── configs/
│   └── config.yaml
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_mask_validation.ipynb
│   ├── 03_training.ipynb
│   ├── 04_ablation.ipynb
│   └── 05_results.ipynb
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── results.json
├── requirements.txt
├── README.md
└── HANDOVER.md
```

---

## Paper Figures

**Core figures** (must have):
1. Dataset overview — browse images with regions highlighted
2. Mask generation — raw patch → Otsu threshold → binary mask → cleaned
3. Qualitative results — input / ground truth / prediction grid
4. Confusion matrix
5. Baseline comparison — intensity threshold vs U-Net
6. Full-strip segmentation overlay with boundary
7. Cross-crater bar chart (IoU/Dice/HD95 per crater)
8. Ablation comparison table

**Optional figures** (if extensions implemented):
9. Post-processing comparison (if CRF/active contour used)
10. TTA/ensemble comparison (if implemented)
11. Boundary detail zoom with lat/lon

---

## Dependencies

```txt
# requirements.txt
torch>=2.12.0
torchvision>=0.27.0
numpy>=2.4.6
scikit-image>=0.26.0
scipy>=1.17.1
matplotlib>=3.10.9
albumentations>=2.0.8
tqdm>=4.67.3
pyyaml>=6.0.3
tensorboard>=2.20.0
segmentation-models-pytorch>=0.5.0
```

**Python**: >= 3.11

**PyTorch CPU-only install**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Optional** (for extensions only):
- `timm>=1.0.27` — if using timm encoders with SMP
- `pydensecrf` — for Dense CRF post-processing (build from GitHub; may require Visual C++ on Windows)

---

## Execution Order

### Core Pipeline (must complete)

| Step | Task | Depends On |
|------|------|------------|
| 1 | Create project structure | — |
| 2 | Install dependencies | — |
| 3 | Implement mask generator | — |
| 4 | Validate masks (visualize + log stats) | Step 3 |
| 5 | Implement dataset + splits | Steps 3-4 |
| 6 | Implement augmentations | — |
| 7 | Set up SMP model | — |
| 8 | Implement loss function | — |
| 9 | Implement intensity baseline | — |
| 10 | Run baseline evaluation | Steps 5, 9 |
| 11 | Implement training script | Steps 5-8 |
| 12 | Run training | Steps 4, 11 |
| 13 | Implement evaluation metrics | — |
| 14 | Evaluate model, compare to baseline | Steps 12-13 |
| 15 | Run ablation(s) | Step 12 |
| 16 | Implement full-strip inference | Step 12 |
| 17 | Run inference on raw strips | Step 16 |
| 18 | Generate figures | Steps 14, 15, 17 |
| 19 | Write paper | Step 18 |

### Optional Extensions (after core works)

| Extension | When to try |
|-----------|-------------|
| DINOv2 masks | If Multi-Otsu masks look poor |
| FocusSDF loss | If boundaries are blurry |
| Physics-Informed loss | If boundaries are fragmented |
| Dense CRF | If boundaries don't align with intensity edges |
| TTA | Always (zero training cost) |
| Ensemble | If single model works well and time permits |
| CPU optimization | If training is too slow |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Multi-Otsu fails on non-bimodal patches | Fallback threshold = 0.048; filter low-std patches; try DINOv2 (Extension A) |
| Model too large for CPU | Use MobileNet encoder (~2M params) |
| Training too slow | CPU optimization (Extension H) |
| U-Net doesn't beat intensity baseline | Reconsider approach; paper contribution weakened |
| Full-strip inference crashes | Process in 128-row bands |
| pydensecrf won't build on Windows | Skip CRF extension; focus on other post-processing |
| shackleton_02 ambiguity (sun_elevation ≈ 0°) | Report its metrics separately |

---

## Context

- **Dataset source**: Kaggle — `flickeringtubelight/chandrayaan-2-ohrc-lunar-psrs`
- **Original data**: ISRO PRADAN (pradan.issdc.gov.in)
- **License**: ISRO Open Data Policy — free for research and educational use
- **Working directory**: `D:\Coding\ch-2_OHRC_PSRs`
