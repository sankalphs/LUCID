# LUCID — Lunar Unsupervised Classification of Illumination and Darkness

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-ee4c2c.svg)](https://pytorch.org/)
[![Status: Research Code](https://img.shields.io/badge/status-research--code-orange.svg)](#status--limitations)

Self-supervised binary segmentation of shadow vs. illuminated terrain in lunar Permanently Shadowed Regions (PSRs) from **Chandrayaan-2 OHRC** imagery. Training masks are auto-generated with Multi-Otsu thresholding — no manual pixel annotations required. A U-Net (ResNet-18) learns the shadow–illumination boundary, which is extracted as a post-processing step with `skimage.segmentation.find_boundaries`.

> **Paper draft:** `paper/paper.tex` — *"Delineation of Shadow-Illumination Boundaries in Lunar Permanently Shadowed Regions via Self-Supervised Segmentation from Chandrayaan-2 OHRC Imagery"* (LUCID).

---

## Status & Limitations — Please Read Before Citing

This repository is released as **open-source research code under the MIT License** so others can reproduce, audit, and build on the work. The original plan was a journal submission, but ablation and audit experiments did not support all claims in the draft paper. **We are releasing the code and data pipeline as-is, with full transparency, rather than publishing unsupported results.**

What holds and what does not — summarized from [`FINAL_FINDINGS.md`](FINAL_FINDINGS.md) and [`ablation_results/findings_report.md`](ablation_results/findings_report.md):

- **Reproducible core result:** On the legacy `mixed` split (11,372 train / 2,007 val patches, seed 42, CPU), U-Net reaches **IoU 0.9202 / Dice 0.9574 / HD95 0.345 px** against the Multi-Otsu pseudo-labels. Five-seed mean is **IoU 0.9217 ± 0.0035**. This measures *pseudo-label consistency*, not verified geological truth.
- **Unsupported paper claims — do not cite as published findings:**
  - The crater-disjoint split claim is contaminated: `train.npy` contains ~4,250 patches byte-identical to `shackleton_02` per class. Honest generalization must use the rebuilt per-strip splits (`results/split_integrity/`), where IoU drops by ~1.1 pts.
  - The ablation table in `paper.tex` (IoU 0.44–0.55) used a 12% data subset / 30-epoch / double-sigmoid regime and is **not comparable** to the full-data baseline. Full-data ablations are in [`FINAL_FINDINGS.md` §5](FINAL_FINDINGS.md) (e.g., no-CLAHE −5.1 pts, no-augmentation −6.9 pts, `pos_weight=3.0` is harmful vs. `1.0` on full data).
  - The fallback threshold `0.0484` never fires on real mixed patches — it is dead code.
  - `src/train.py` double-sigmoid and stochastic val CLAHE inject determinism/noise; fixed in the `experiments/` reruns (`activation=None`, deterministic val eval).
- **No expert annotations exist.** Metrics above measure agreement with algorithmic pseudo-labels. A blinded 80-patch expert annotation export is prepared in `results/expert_evaluation/` but has not been labeled.

If you reuse this work, **cite the code, report the split you used, and state explicitly that metrics are pseudo-label agreement unless you add independent ground truth.**

---

## Key Features

- **Self-supervised labels** — Multi-Otsu (3 classes) on mixed patches; fixed masks for PSR/sunlit
- **SMP U-Net / ResNet-18** — 1-channel input, sigmoid output, ~14M params, no ImageNet pretraining
- **Rigorous evaluation** — IoU, Dice, pixel accuracy, HD95, Boundary F1 (2 px tolerance)
- **Full-strip inference** — 64×64 sliding window, stride 32, Gaussian-weighted overlap blending, banded for CPU memory
- **Reproducibility package** — pinned env, per-experiment dirs, `results/data_checksums.sha256`, `docs/REPRODUCIBILITY.md`
- **Extended experiments** — 5-seed, cross-region, architecture comparison (U-Net vs U-Net++ vs DeepLabV3+), classical baselines (Otsu, Adaptive, Random Forest)

---

## Repository Structure

```
├── configs/config.yaml           # single experiment config (portable, relative paths)
├── src/
│   ├── data/{mask_generator,dataset,splits,augmentations}.py
│   ├── models/{losses,tta,ensemble,post_processing,cpu_optimization}.py
│   ├── baselines/intensity_threshold.py
│   ├── train.py / evaluate.py / inference.py / visualize.py / ablation.py
├── experiments/                  # publication-grade runners (framework.py, train_experiment.py, queue_runner.py)
├── paper/paper.tex + figures/    # draft manuscript (see Status note above)
├── notebooks/01_eda … 05_results.ipynb
├── results/                      # all rerun outputs, checksums, plots (generated)
├── ablation_results/             # legacy ablations (12% regime — not comparable)
├── dataset/kaggle_dataset/       # NOT tracked — download separately (see below)
├── run_pipeline.py / train_full.py / evaluate_full.py / run_ablations.py / run_baselines.py
├── requirements.txt              # minimal deps
├── requirements-lock.txt         # full pip freeze from 2026-08-29 machine
└── LICENSE                       # MIT
```

---

## Installation

**Python 3.11+ required.** CPU-only PyTorch is sufficient.

```bash
git clone https://github.com/sankalphs/ch-2_OHRC_PSRs.git
cd ch-2_OHRC_PSRs

# CPU PyTorch (Windows/Linux)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# For exact reproduction of the audit machine:
# pip install -r requirements-lock.txt
```

---

## Dataset

Source: ISRO PRADAN via Kaggle `flickeringtubelight/chandrayaan-2-ohrc-lunar-psrs` (OHRC, 0.22–0.27 m/px).

The repository does **not** bundle imagery (too large, ISRO data policy). Download and place as:

```
dataset/kaggle_dataset/
├── metadata.json
├── patches/{psr,sunlit,mixed}/{train,val,shackleton_01,shackleton_02,cabeus_01}.npy
└── raw/{shackleton_01,shackleton_02,cabeus_01}/{image.img,label.xml,geometry.csv,browse.png}
```

Pre-extracted patch stats: PSR 8,663/1,529 · Sunlit 12,750/2,250 · Mixed 11,372/2,007 (train/val). Mean intensities: PSR 0.028, Sunlit 0.437, Mixed 0.083.

Verify integrity (if you have `results/data_checksums.sha256` from a prior run):

```bash
sha256sum -c results/data_checksums.sha256        # Linux
# PowerShell: Get-Content results/data_checksums.sha256 | ForEach-Object { ... }
```

**Config portability:** `configs/config.yaml:data.base_dir` defaults to a relative path `dataset/kaggle_dataset`. Override on the CLI or via env — no hard-coded `D:/...` paths remain (legacy absolute defaults in `src/data/dinov2_mask_generator.py` are kept as fallback but `config.yaml` is authoritative).

---

## Quick Start

### 1. Core pipeline (original `run_pipeline.py`)

```bash
# validate masks → train → evaluate → ablations → figures
python run_pipeline.py --phase all

# individual phases
python run_pipeline.py --phase validate
python run_pipeline.py --phase train
python run_pipeline.py --phase evaluate
python run_pipeline.py --phase ablation
python run_pipeline.py --phase figures

# resume training
python run_pipeline.py --phase train --resume outputs/checkpoints/best_model.pth
```

Standalone scripts:

```bash
python train_full.py          # tqdm training
python evaluate_full.py       # full evaluation + metrics
python run_baselines.py       # Otsu / adaptive / RF baselines
python run_ablations.py       # ablation matrix (see caveat in Status)
```

### 2. Full reproducibility suite (audit-grade, ~20 CPU-heavy runs)

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the canonical command table.

```bash
# list all 13 experiment IDs with cost estimates (no run)
python experiments/run_all_publication_experiments.py --list

# full suite (multiple CPU-days)
python experiments/run_all_publication_experiments.py

# subsets
python experiments/run_all_publication_experiments.py --only reproduction
python experiments/run_all_publication_experiments.py --only classical_A_strip,classical_B,classical_C
python experiments/run_all_publication_experiments.py --only multiseed --keep-going

# low-level
python experiments/train_experiment.py --exp-id repro_seed42 --seed 42
python experiments/run_classical_baselines.py --split-preset A_strip
```

Outputs land in `results/experiments/<exp_id>_<stamp>/` with `config.json`, `train.log`, `history.json`, `best.pth`, `eval_*.json`, `predictions_*.npz`, `training_curves.png`, plus central registries `results/all_results.csv|json`.

---

## Model & Training

- **Architecture:** `segmentation_models_pytorch.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=1)` — set `activation=None` and apply sigmoid in the loss (fixes legacy double-sigmoid).
- **Loss:** `(1-α)·WeightedBCE(pos_weight) + α·Dice`, default `pos_weight=3.0, dice_weight=0.5`. Full-data audit found `pos_weight=1.0` and Dice-only both beat the default.
- **Augmentation ( Albumentations ):** `HorizontalFlip/VerticalFlip/RandomRotate90 (0.5)`, `CLAHE (1–4, p=0.5)` — critical for dark PSR patches — `RandomBrightnessContrast`, `RandomGamma`, `CoarseDropout`. Val should be **deterministic** (no stochastic CLAHE) for reporting; stochastic CLAHE costs ~1.4 IoU pts.
- **Optim:** Adam `lr=1e-3, weight_decay=1e-4`, `ReduceLROnPlateau(factor=0.5, patience=7)`, early stopping `patience=15` on val IoU, batch 32, max 100 epochs, seed 42.

---

## Results (What We Stand Behind)

| Split | Model | IoU | Dice | Acc | HD95 | BF1 |
|-------|-------|-----|------|-----|------|-----|
| legacy mixed (leaky) | U-Net R18, seed 42 | 0.9202 | 0.9574 | 0.9636 | 0.345 | 0.9391 |
| legacy mixed | U-Net R18, 5-seed mean±std | **0.9217±0.0035** | 0.9583±0.0020 | 0.9646±0.0017 | 0.338±0.048 | 0.9418±0.0049 |
| rebuilt B (train sh01+sh02 → test cabeus) | U-Net R18 | 0.9092 | 0.9513 | 0.9592 | 0.478 | 0.9087 |
| rebuilt C (train cabeus → test sh01+sh02) | U-Net R18 | 0.9098 | 0.9516 | 0.9579 | 0.424 | 0.9190 |
| legacy mixed (independent baseline) | Random Forest | 0.7964 | 0.8787 | 0.9034 | 1.648 | 0.8189 |

> All numbers are agreement with **automatically generated Multi-Otsu masks**, not independent geological ground truth. Cross-region drops are ~1.0–1.1 pts. Architecture ranking on controlled mixed-only data: **U-Net++ 0.9228 > U-Net 0.9202 > DeepLabV3+ 0.8904**.

Detailed tables, plots, error analysis, and the expert-annotation export are in `results/` and documented in `FINAL_FINDINGS.md`.

---

## Reproducibility

- `docs/REPRODUCIBILITY.md` — dataset provenance, hardware (AMD Ryzen AI 9 HX 370, CPU-only), env table, seeds/determinism notes, per-experiment argv, directory contract.
- `requirements-lock.txt` — full `pip freeze` from the audit machine (2026-08-29, Python 3.13.15, `torch==2.13.0+cpu`).
- `FINAL_FINDINGS.md` — complete audit with 8 integrity issues and corrected tables to use if you fork the paper.

Nondeterminism is expected (oneDNN reduction order, stochastic CLAHE if enabled): expect IoU jitter of a few tenths of a percent across machines.

---

## Citation

If you use this code, please cite the repository and state your split/protocol. A paper citation will be added if a peer-reviewed version is published.

```bibtex
@software{lucid_ohrc_2026,
  title  = {LUCID: Lunar Unsupervised Classification of Illumination and Darkness from Chandrayaan-2 OHRC Imagery},
  author = {Sankalp H S},
  year   = {2026},
  url    = {https://github.com/sankalphs/ch-2_OHRC_PSRs},
  note   = {Research code; metrics report pseudo-label agreement unless otherwise noted}
}

% Draft manuscript (unpublished, claims in paper.tex not all supported — see Status note):
@unpublished{lucid_draft_2026,
  title  = {Delineation of Shadow-Illumination Boundaries in Lunar Permanently Shadowed Regions via Self-Supervised Segmentation from Chandrayaan-2 OHRC Imagery},
  author = {Sankalp H S},
  year   = {2026},
  note   = {Unpublished draft; see FINAL_FINDINGS.md for audit}
}
```

---

## License

**MIT** — see [LICENSE](LICENSE). Copyright (c) 2026 Sankalp H S.

OHRC imagery itself is subject to the **ISRO Open Data Policy** and is not redistributed here; you must obtain it via PRADAN/Kaggle under its own terms.

---

## Acknowledgments

- ISRO / Chandrayaan-2 OHRC team and PRADAN for the imagery.
- Kaggle dataset `flickeringtubelight/chandrayaan-2-ohrc-lunar-psrs` for the patch curation.
- `segmentation-models-pytorch`, `albumentations`, `scikit-image`, and the open-source Python ecosystem.
