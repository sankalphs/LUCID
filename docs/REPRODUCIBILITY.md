# Reproducibility Guide

How to re-run every experiment in the Chandrayaan-2 OHRC Lunar PSR segmentation
study from a clean checkout of this repository.

## 1. Dataset provenance

- **Kaggle dataset id:** `flickeringtubelight/chandrayaan-2-ohrc-lunar-psrs`
- **Dataset metadata created:** `2026-05-17T17:06:28` (see `dataset/kaggle_dataset/metadata.json`)
- **Origin:** ISRO PRADAN (pradan.issdc.gov.in), Chandrayaan-2 Orbiter High
  Resolution Camera (OHRC), 0.25 m/pixel, PDS4 products.
- **Source products:**

| Region          | Product ID                                   | Acquired  |
|-----------------|----------------------------------------------|-----------|
| shackleton_01   | `ch2_ohr_ncp_20251007T0659402125_d_img_d18`  | 20251007  |
| shackleton_02   | `ch2_ohr_ncp_20251105T0652329256_d_img_d18`  | 20251105  |
| cabeus_01       | `ch2_ohr_ncp_20250310T0833447498_d_img_d18`  | 20250310  |

- **Data integrity:** every materialized array is checksummed in
  [`results/data_checksums.sha256`](../results/data_checksums.sha256)
  (SHA256, `<hex>  <relative/path>` format; covers all 15 `.npy` patch arrays
  under `dataset/kaggle_dataset/patches/{mixed,psr,sunlit}/` plus
  `dataset/kaggle_dataset/metadata.json`). Verify with:
  `sha256sum -c results/data_checksums.sha256` (or the PowerShell equivalent).

### CRITICAL data-integrity note (split overlap)

The materialized `train.npy` contains **~4,250 byte-identical patches from
shackleton_02 per class** (see
`results/split_integrity/split_overlap_report.json`): e.g. `psr:train ∩
shackleton_02 = 4255`, `mixed:train ∩ shackleton_02 = 4255`,
`sunlit:train ∩ shackleton_02 = 4238`. The **documented split claim does not
hold at patch level** — train/test leakage exists against the shackleton_02
region.

Consequently the publication experiments explicitly distinguish:

- **`legacy`** split — exactly as published (leaky; kept only to reproduce the
  published numbers), and
- **rebuilt region-disjoint splits `A_strip` / `B` / `C`**, constructed from the
  **per-strip arrays only** (`{cabeus_01,shackleton_01,shackleton_02}.npy`),
  which eliminate cross-region patch duplication.

All honest generalization claims must be based on the rebuilt splits.

## 2. Environment

| Component                    | Version                                    |
|------------------------------|--------------------------------------------|
| Python                       | 3.13.15                                    |
| torch                        | 2.13.0+cpu                                 |
| segmentation_models_pytorch  | 0.5.0                                      |
| numpy                        | 2.5.1                                      |
| albumentations               | 2.0.8                                      |

(smp and albumentations versions read from
`results/reproduction/checkpoint_reeval.json` → `versions` key.)

Full frozen environment: [`requirements-lock.txt`](../requirements-lock.txt)
(output of `pip freeze` on this machine).

**Hardware:** CPU-only training. AMD Ryzen AI 9 HX 370, 24 logical cores.
The NPU and iGPU are unusable for PyTorch training here — no VitisAI /
DirectML execution providers are installed; everything runs on CPU threads.

## 3. Seeds & determinism

- Default seed: **42**.
- `set_seed()` seeds `random`, `numpy`, and `torch` (+ `torch.cuda` when available).
- `torch.backends.cudnn.deterministic` is **NOT** set (irrelevant on CPU anyway);
  oneDNN reduction-order variance means results are **approximately reproducible,
  not bit-exact** across runs/machines. Expect IoU differences of a few tenths
  of a percent.
- The validation protocol includes **stochastic CLAHE** (`clip_limit` drawn from
  U(1, 4)) to match the published pipeline; metrics under this protocol are
  stored as `eval_test_valclahe.json`. A **no-transform eval variant** is stored
  alongside as `eval_test_notransform.json` (`*_clean` metrics) for comparison.

## 4. Canonical commands

Every experiment ID maps to exact argv vectors run by
`experiments/run_all_publication_experiments.py` (executed with `sys.executable`
from the repo root):

| ID                   | Exact command line                                                                                     |
|----------------------|--------------------------------------------------------------------------------------------------------|
| `reproduction`       | `python experiments/train_experiment.py --exp-id repro_seed42 --seed 42`                               |
| `classical_legacy`   | `python experiments/run_classical_baselines.py --split-preset legacy`                                  |
| `classical_A_strip`  | `python experiments/run_classical_baselines.py --split-preset A_strip`                                 |
| `classical_B`        | `python experiments/run_classical_baselines.py --split-preset B`                                       |
| `classical_C`        | `python experiments/run_classical_baselines.py --split-preset C`                                       |
| `multiseed`          | `python experiments/train_experiment.py --exp-id multiseed_seed<S> --seed <S>` for S in {123,456,789,2026} (seed 42 == reproduction) |
| `crossregion`        | `python experiments/train_experiment.py --exp-id crossB_seed42 --seed 42 --split-preset B` then `--exp-id crossC_seed42 --split-preset C` |
| `ablations`          | six runs with base seed 42: `--no-clahe` (`ablation_no_clahe`); `--no-aug` (`ablation_no_augmentation`); `--dice-weight 0.0` (`ablation_bce_only`); `--dice-weight 1.0` (`ablation_dice_only`); `--pos-weight 1.0` (`ablation_posweight1`); `--no-morphology` (`ablation_no_morphology`) |
| `architectures`      | `python experiments/train_experiment.py --exp-id arch_deeplabv3plus_seed42 --seed 42 --arch deeplabv3plus`; then `--exp-id arch_unetplusplus_seed42 --arch unetplusplus` |
| `fallback`           | `python experiments/fallback_sensitivity.py`                                                            |
| `pseudolabel_quality`| `python experiments/analyze_pseudo_label_quality.py`                                                    |
| `expert_export`      | `python experiments/build_annotation_export.py`                                                         |
| `aggregate`          | `python experiments/aggregate_results.py`                                                               |

## 5. Directory contract

- **Registry:** `results/all_results.csv` and `results/all_results.json`
  accumulate one row/record per finished experiment.
- **Per-experiment dirs:** `results/experiments/<exp_id>_<stamp>/` containing
  - `config.json` — full resolved configuration
  - `train.log` — training log
  - `history.json` — epoch history
  - `best.pth` — best checkpoint by validation IoU
  - `eval_test_valclahe.json` — test metrics, validation-style stochastic CLAHE protocol
  - `eval_test_notransform.json` — test metrics, no transform (`*_clean`)
  - `predictions_test_valclahe.npz` — raw test predictions
  - `training_curves.png` — loss/IoU curves
- **Classical baselines:** `results/classical/<method>/`.
- **Queue mechanics:** jobs can be appended as JSON lines (one
  `{"args": [...], "name": ...}` per line) to `results/queue/queue.txt`;
  `experiments/queue_runner.py` pops them sequentially and moves finished lines
  to `done.txt`. This decouples long CPU runs from interactive shells.

## 6. How to rerun everything

```bash
# Full suite (~20 training-heavy runs; MULTIPLE CPU-DAYS on this machine):
python experiments/run_all_publication_experiments.py

# List IDs + cost estimates without running anything:
python experiments/run_all_publication_experiments.py --list

# Individual experiments:
python experiments/run_all_publication_experiments.py --only reproduction
python experiments/run_all_publication_experiments.py --only classical_A_strip,classical_B,classical_C
python experiments/run_all_publication_experiments.py --only multiseed --keep-going
python experiments/run_all_publication_experiments.py --only ablations,crossregion

# Or invoke the underlying scripts directly, e.g.:
python experiments/train_experiment.py --exp-id repro_seed42 --seed 42
python experiments/run_classical_baselines.py --split-preset A_strip
```

The runner prints `[START <id>] ...` / `[DONE <id> rc=<n>]` per run, stops at
the first failure unless `--keep-going` is given, and exits non-zero if any run
failed.
