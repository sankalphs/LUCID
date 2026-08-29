# LUCID Experiment Audit (Phase 0)

Date: 2026-08-26 · Repo: `D:/Coding/ch-2_OHRC_PSRs` @ commit `5ec6c75` (branch `main`, clean tree)

## 1. Repository structure

```
├── configs/config.yaml            # central config (seed 42, hyperparameters, paths)
├── src/
│   ├── data/{dataset,splits,mask_generator,augmentations,dinov2_mask_generator}.py
│   ├── models/{losses,tta,post_processing,ensemble,physics_informed_loss,focus_sdf_loss,cpu_optimization}.py
│   ├── train.py                   # library trainer (CombinedPSRDataset, sigmoid head)
│   ├── evaluate.py                # ALL final metrics live here (Evaluator)
│   ├── ablation.py, inference.py, visualize.py
│   └── baselines/intensity_threshold.py
├── train_full.py                  # <-- produced the paper's main result
├── evaluate_full.py               # <-- produced outputs/results.json
├── run_pipeline.py                # orchestrator (validate/train/evaluate/ablation/figures/extensions)
├── run_baselines.py               # U-Net++ / DeepLabV3+ architecture baselines -> baselines/
├── run_ablations.py               # published ablations -> ablation_results/
├── dataset/kaggle_dataset/        # patches/{psr,sunlit,mixed}/{train,val,<strip>}.npy + raw strips
├── outputs/                       # checkpoints, results.json, training_history.json, figures
├── ablation_results/              # recorded ablation JSONs + findings_report.md
├── baselines/                     # architecture-baseline metrics + checkpoints
├── notebooks/01..05_*.ipynb
├── paper/paper.tex + paper.pdf
└── tests/                         # pytest suites + run_all_tests.py smoke pipeline
```

Dataset: Kaggle `flickeringtubelight/chandrayaan-2-ohrc-lunar-psrs` (created 2026-05-17; ISRO PRADAN origin).
Strips: shackleton_01 (20251007), shackleton_02 (20251105), cabeus_01 (20250310); both Shackleton strips are the *same crater* on different dates.

## 2. Existing experiment pipeline (as executed for the paper)

1. **Pseudo-labels** (`src/data/mask_generator.py`): PSR→all-dark(0); Sunlit→all-bright(1);
   Mixed→3-class Multi-Otsu (`skimage.filters.threshold_multiotsu(patch, classes=3)`),
   binarize with lowest threshold `patch > thresholds[0]`;
   on `ValueError` → fallback `patch > FALLBACK_THRESHOLD` where `FALLBACK_THRESHOLD = 0.0284 + 0.02 = 0.0484`
   (`PSR_MEAN = 0.0284`, mask_generator.py:24-26). Masks cleaned by morphological **closing only**
   (`skimage.morphology.closing(mask, disk(1))`, default `disk_radius=1`). No preprocessing before thresholding.
2. **Data**: `PSRDataset` loads pre-extracted `<class>/<split>.npy` float32 [0,1] arrays (no stored labels;
   masks synthesized from folder name). Train augmentations (albumentations, hardcoded):
   HFlip .5, VFlip .5, Rot90 .5, CLAHE(clip 1–4, tile 8) p=.5, BrightnessContrast(±.15/±.3) p=.7,
   Gamma[70,130] p=.4, CoarseDropout(1–4 holes 8–16px) p=.3.
   Val transform = CLAHE-only **with random clip_limit drawn from (1.0, 4.0)** per sample.
3. **Training** (`train_full.py`): U-Net SMP ResNet18, `encoder_weights=None`, 1-ch input, `activation=None`
   (manual sigmoid). Loss `WeightedBCEDiceLoss` = `0.5·wBCE(pos_weight=3) + 0.5·Dice(batch-global, smooth=1)`
   (losses.py:46-56). Adam lr=1e-3 wd=1e-4; ReduceLROnPlateau(max, factor .5, patience 7, min_lr 1e-5)
   stepped on **global-confusion val IoU** (`MetricTracker`); EarlyStopping patience 15, min_delta 1e-4;
   batch 32; max 100 epochs; num_workers=0; seed 42 (`random/np/torch`; **no cudnn flags**).
   Best checkpoint by global val IoU → `outputs/checkpoints/best_model.pth`.
4. **Evaluation** (`evaluate_full.py` → `src/evaluate.py::Evaluator`): per-patch metrics then mean:
   IoU, Dice, pixel accuracy, HD95 = 95th pct of pooled bidirectional surface EDT distances (pixels,
   empty-side penalty 64), Boundary F1 = outer-boundary EDT matching tol 2 px.
   Also evaluates intensity-threshold sweep {0.02…0.20}, TTA(6 views), gaussian/morph post-processing.

Recorded artifacts: `outputs/training_history.json` (57 epochs, best val IoU 0.9229 @ epoch 42),
`outputs/results.json` (headline numbers), `baselines/*.json` (U-Net++, DeepLabV3+ full runs),
`ablation_results/*` (12 %-data regime).

## 3. Exact current implementation facts

| Item | Value | Source |
|---|---|---|
| Patch size / dtype | 64×64 float32 [0,1], no normalization | dataset.py:47-50 |
| Fallback threshold | 0.0484 (only when multi-Otsu raises ValueError) | mask_generator.py:25-26 |
| Morphology | closing, disk(1) | mask_generator.py:99-116 |
| Model | smp.Unet resnet18 no-pretrain 14,321,937 params, logits output | train_full.py:129-135 |
| Loss | 0.5·wBCE(pw=3)+0.5·Dice, internal sigmoid on logits | losses.py:28,46,56 |
| Optimizer/sched | Adam 1e-3/1e-4; RLROP(.5,7,min 1e-5) on val IoU | config.yaml:43-49 |
| Early stop | patience 15, min_delta 1e-4 | train_full.py:37 |
| Batch/workers | 32 / 0, shuffle=True | train_full.py:117-124 |
| Seed | 42 (random,np,torch only; no CUDA seed, no cudnn.deterministic) | train_full.py:30-33 |
| Metrics | per-patch mean; HD95 pooled-surface px; BF1 tol=2 | src/evaluate.py |
| Headline result | IoU .9203 Dice .9576 Acc .9640 HD95 .3454 BF1 .9373 | outputs/results.json |

## 4. Bugs and discrepancies found (ranked)

1. **[CRITICAL - data leakage] Split contamination.** Documented split "train = Shackleton-01+Cabeus-01,
   val/test = Shackleton-02" is false at patch level. Exact byte-hash analysis
   (`results/split_integrity/split_overlap_report.json`): train.npy contains **4,255 (psr) / 4,238 (sunlit) /
   4,255 (mixed)** patches byte-identical to shackleton_02.npy (~⅓ of train comes from the "held-out" strip).
   Strip files are pairwise disjoint (0 overlap), and train∩val has **zero** exact duplicates, but the
   same-strip contamination invalidates "held-out crater" claims. Both Shackleton strips are also the same
   physical crater (different dates). ⇒ All geographic-generalization experiments must rebuild splits from
   strip files only.
2. **[CRITICAL - provenance] Main model trained on mixed-class patches ONLY** (train_full.py:114 loads
   `mixed/train.npy`, 11,372 patches) while paper narrative reports 32,785 combined patches. The reported
   headline number corresponds to mixed-only training.
3. **[CRITICAL - integrity] Paper/report ablation table (≈0.85–0.88 IoU) is unsupported.** Recorded
   ablation runs used stratified 12 % data (4,099 patches), max 30 epochs, and the double-sigmoid path
   (below), scoring IoU 0.44–0.55 (ablation_results/_all_results.json). findings_report.md itself admits
   non-comparability. Phase 7 re-runs all ablations at full scale.
4. **[HIGH] Double sigmoid on `build_model` path.** `config model.activation: "sigmoid"` +
   `WeightedBCEDiceLoss` internally sigmoiding ⇒ `src/train.py`, `run_ablations.py`, `src/ablation.py`,
   `run_pipeline --phase train` optimize sigmoid∘sigmoid(logits). `train_full.py`/`evaluate_full.py`
   correctly use activation=None. Two silently different objectives coexist.
5. **[HIGH] Stochastic validation preprocessing.** `get_val_transforms()` draws a random CLAHE clip_limit
   from (1.0, 4.0) per val sample (augmentations.py:57) despite docstring claiming consistency.
   Checkpoint re-eval: paper-faithful protocol IoU .9203 vs no-transform IoU .9337 — val CLAHE costs ~1.3 IoU
   points and injects eval noise into every reported number.
6. **[MED] Val strip == test strip** (config.yaml:24-25): model selection (early stop/checkpoint) tuned on
   the same strip reported as test.
7. **[MED] Metric-history anomaly:** old `results.json` contains boundary_f1 max = 1.041 > 1 (impossible
   under current code); current implementation reproduces headline values exactly, so anomaly predates
   committed metric revision. Noted; raw file preserved as-is.
8. **[LOW] Decorative config:** yaml augmentation/mask sections largely unread (hardcoded pipelines);
   `run_pipeline.phase_train` ignores `--config`; `splits.py:61` records class name as strip label;
   CoarseDropout `fill_value` arg invalid under albumentations>=2.0.8 (warning observed in logs);
   `get_test_transforms()` dead code with random geometric ops mislabeled "TTA"; DINOv2 extension never
   produced outputs and misuses torchvision supervised ViT-B/14; ensemble/cpu_optimization modules unused.
9. **[LOW] Nondeterminism:** no cudnn determinism flags anywhere; oneDNN reduction-order variance observed;
   albumentations RNG not independently seeded. Bit-exact GPU/CPU reproduction not guaranteed (documented).

## 5. Reproducibility verification already performed

* **Checkpoint identity:** loading `outputs/checkpoints/best_model.pth` and running the repo's own evaluator
  reproduces the paper numbers exactly (see table below) ⇒ stored checkpoint ↔ results.json ↔ metric code are consistent.
* **Split integrity:** hash-overlap matrix computed and saved (see §4.1).
* **NPU assessment:** AMD XDNA NPU present but unusable for training: onnxruntime lacks VitisAI/DirectML EP,
  no torch-directml/OpenVINO/Ryzen-AI stack installed; NPUs support inference only. All training remains CPU
  (Ryzen AI 9 HX 370, 24 threads, 23 GB RAM).

| Eval protocol of stored checkpoint | IoU | Dice | Acc | HD95 | BF1 |
|---|---|---|---|---|---|
| paper-faithful (random-CLAHE val) | 0.9203 | 0.9576 | 0.9640 | 0.3454 | 0.9373 |
| no val transform (cleaner)        | 0.9337 | 0.9649 | 0.9709 | 0.2574 | 0.9603 |

## 6. Missing experiments required for publication

Full-data reproduction under seed control; classical baselines (global/Otsu/Multi-Otsu/+morph/adaptive/RF);
multi-seed mean±std; region-disjoint cross-region splits (rebuilt from strips); fair architecture comparison
(existing U-Net++/DeepLabV3+ runs reusable); full-scale component ablations; fallback-threshold sensitivity
(train-only selection); pseudo-label quality analysis; expert-annotation export (none exist in repo);
error-analysis figures; machine-readable experiment registry.

## 7. Execution plan

Phase 1 reproduce (train_full.py semantics, seed 42) → Phase 2 framework (`experiments/framework.py`,
registry `results/all_results.csv|json`) → Phase 3 classical baselines (cheap) → Phase 4 seeds {123,456,789,2026}
(each ≈2.5-4 h CPU) → Phase 5 splits A′/B/C from strips → Phase 6 reuse existing arch baselines + seed-42 rerun
if needed → Phase 7 full-scale ablations → Phase 8 threshold sweep (mask-level + short-budget probes) →
Phases 9-11 analyses/figures → Phases 12-13 packages/reports. Long runs execute sequentially in background;
all raw logs preserved.
