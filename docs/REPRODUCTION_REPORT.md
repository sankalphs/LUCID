# Reproduction Report — LUCID U-Net/ResNet18 (Phase 1)

Date: 2026-08-26 · Experiment ID: `repro_seed42` · Runtime: 211.7 min CPU (12 threads), 71 epochs (early stop), best epoch **56**

## 1. Protocol

Faithful re-execution of `train_full.py` semantics through the new standardized
runner (`experiments/train_experiment.py`): mixed-class patches only
(11,372 train / 2,007 val), SMP U-Net ResNet18 no-pretrain logits head,
`0.5·wBCE(pos_weight=3)+0.5·Dice`, Adam(1e-3, wd 1e-4),
ReduceLROnPlateau(max, .5, 7, min 1e-5) on global-confusion val IoU,
early stop patience 15 (min_delta 1e-4), batch 32, shuffle, num_workers 0,
seed 42, paper augmentation incl. stochastic val CLAHE.

## 2. Headline comparison (paper-faithful eval protocol)

| Metric | Paper (`outputs/results.json`) | Reproduced | abs diff |
|---|---|---|---|
| IoU   | 0.9203 | **0.9202** | 0.0001 |
| Dice  | 0.9576 | **0.9574** | 0.0002 |
| Accuracy | 0.9640 | **0.9636** | 0.0004 |
| HD95 (px) | 0.3454 | **0.3451** | 0.0003 |
| Boundary F1 | 0.9373 | **0.9391** | 0.0018 |

All five metrics agree to within 0.002 → the published numbers are confirmed as
genuine outputs of the committed pipeline + stored data.

Additional protocol variants for the same best checkpoint:

| Eval variant | IoU | Dice | Acc | HD95 | BF1 |
|---|---|---|---|---|---|
| Paper-faithful (stochastic val CLAHE) | 0.9202 | 0.9574 | 0.9636 | 0.3451 | 0.9391 |
| No val transform ("clean")           | 0.9338 | 0.9649 | 0.9707 | 0.2768 | 0.9599 |

The stochastic CLAHE in `get_val_transforms()` costs ≈1.4 IoU points and is a
source of run-to-run variance in every reported number; recommend reporting
clean-protocol values (or fixing clip_limit) in any revision.

## 3. Training-trajectory agreement

| Quantity | Original run | This rerun |
|---|---|---|
| Best global val IoU | 0.92292719 @ ep42 | **0.92292702 @ ep56** |
| LR schedule drops | 5e-4@26, 2.5e-4@39, 1.25e-4@50 | 5e-4@26, 2.5e-4@39, 1.25e-4@64* |
| Early stop epoch | 57 | 71 |

Peak metric matches to ~2e-7; the best-epoch offset reflects oneDNN reduction-order
and RNG-consumption nondeterminism (no cudnn-determinism flags in repo; documented
in EXPERIMENT_AUDIT.md §4.9). No substantive discrepancy remains to explain.

## 4. Checkpoint identity check (pre-training)

Before any new training, loading `outputs/checkpoints/best_model.pth` into the
current code reproduced results.json exactly (IoU 0.9203 / Dice 0.9576 / Acc
0.9640 / HD95 0.3454 / BF1 0.9373):
see `results/reproduction/checkpoint_reeval.json`. The archived artifact,
metric implementation and headline numbers are mutually consistent.
(The old file's anomalous `boundary_f1.max=1.041` could not be reproduced with
committed code and predates it; noted in audit §4.7.)

## 5. Artifacts

```
results/experiments/repro_seed42_20260826_091530/
├── config.json            # fully resolved experiment configuration
├── train.log              # per-epoch log (losses/IoU/LR/timing)
├── history.json           # full training history
├── training_curves.png    # loss/IoU/LR curves
├── best.pth               # best checkpoint (state_dict)
├── eval_test_valclahe.json / eval_test_notransform.json
├── predictions_test_valclahe.npz   # float16 probability maps, all 2007 patches
└── result_row.json        # registry row copy
results/reproduction/checkpoint_reeval.json + checkpoint_reeval_probs.npz
results/error_analysis/     # Phase 11 panels + summary (uses this run)
results/all_results.csv|.json  # central registry
```

## 6. Conclusion

Reproduction status: **successful**. Differences ≤ 0.002 on all metrics are
attributable to documented floating-point/scheduling nondeterminism. Any future
manuscript revision can safely cite these as independently re-executed results,
preferably alongside multi-seed aggregates (Phase 4, running).
