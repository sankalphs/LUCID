# FINAL EXPERIMENT RECOMMENDATION — LUCID

Date: 2026-08-28 · Seed for all new runs: 42 unless noted · Status: **queue running in background** (DeepLabV3+ epoch ~71/100, UNet++ queued). This document uses only executed results; nothing is fabricated.

---

## 1. Five-seed U-Net mean ± std (mixed-only, legacy protocol, paper-faithful eval)

Completed 5/5 seeds (repro + 123/456/789/2026). Per-seed test metrics:

| seed | IoU | Dice | Acc | HD95 | BF1 | best ep |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.9202 | 0.9574 | 0.9636 | 0.3451 | 0.9391 | 56 |
| 123 | 0.9166 | 0.9555 | 0.9623 | 0.4136 | 0.9357 | 41 |
| 456 | 0.9219 | 0.9584 | 0.9646 | 0.3349 | 0.9414 | 67 |
| 789 | 0.9255 | 0.9605 | 0.9665 | 0.2897 | 0.9488 | 77 |
| 2026 | 0.9242 | 0.9598 | 0.9659 | 0.3045 | 0.9438 | 76 |

Aggregate (`results/multiseed_summary.csv`):

| metric | mean | std | min | max | n |
|---|---:|---:|---:|---:|---:|
| IoU | **0.9217** | **0.0035** | 0.9166 | 0.9255 | 5 |
| Dice | 0.9583 | 0.0020 | 0.9555 | 0.9605 | 5 |
| Accuracy | 0.9646 | 0.0017 | 0.9623 | 0.9665 | 5 |
| HD95 | 0.3376 | 0.0480 | 0.2897 | 0.4136 | 5 |
| BF1 | 0.9418 | 0.0049 | 0.9357 | 0.9488 | 5 |

Clean (no-transform) variant is higher: IoU **0.9347 ±0.0048** — see §8.

**Recommendation:** replace the single-seed headline `0.9203` with `0.922 ±0.004` (or `0.935 ±0.005` if switching to deterministic validation).

---

## 2. Clean strip-disjoint cross-region results

Rebuilt from per-strip arrays only (no contaminated `train.npy`). Single seed 42 each; test metrics paper-faithful:

| Train | Test | IoU | Dice | Acc | HD95 | BF1 |
|---|---|---:|---:|---:|---:|---:|
| shackleton_01+cabeus_01 (legacy-contaminated) | shackleton_02 (val arrays) | 0.9202 | 0.9574 | 0.9636 | 0.3451 | 0.9391 |
| shackleton_01+shackleton_02 | cabeus_01 | **0.9092** | 0.9513 | 0.9592 | 0.4780 | 0.9087 |
| cabeus_01 | shackleton_01+shackleton_02 | **0.9098** | 0.9516 | 0.9579 | 0.4241 | 0.9190 |

Drop vs legacy: **-1.10 and -1.04 IoU points**. Generalization is strong; the contaminated legacy number is only modestly optimistic. Classical baselines are split-invariant (RF ~0.79-0.80 on every strip), confirming that geographic variation matters only for the learned model — reported in `results/cross_region_summary.csv`. Five-seed cross-region recommended but single-seed already supports the claim "model generalizes across strips."

---

## 3. Full-data ablation results (all at full 11,372-patch scale, seed 42, legacy split)

Reference = Full LUCID (repro_seed42). Ablations are single-change only:

| Variant | IoU | Dice | Acc | HD95 | BF1 | ΔIoU vs Full |
|---|---:|---:|---:|---:|---:|---:|
| Full LUCID | 0.9202 | 0.9574 | 0.9636 | 0.3451 | 0.9391 | — |
| No CLAHE | **0.8692** | 0.9282 | 0.9372 | 1.0047 | 0.8826 | **-0.0510** |
| No augmentation | **0.8511** | 0.9167 | 0.9256 | 1.4321 | 0.8517 | **-0.0691** |
| BCE only (dice_weight=0) | 0.9179 | 0.9561 | 0.9626 | 0.3809 | 0.9335 | -0.0023 |
| Dice only (bce_weight=0) | **0.9239** | 0.9596 | 0.9658 | 0.3120 | 0.9439 | **+0.0037** |
| Positive weight = 1 (vs 3) | **0.9289** | 0.9624 | 0.9682 | 0.2555 | 0.9553 | **+0.0087** |
| No morphology in pseudo-labels | 0.9183 | 0.9565 | 0.9635 | 0.3477 | 0.9443 | -0.0019 |

`results/ablation_summary.csv` — replaces the unsupported paper table.

Key findings: augmentation matters enormously (-6.9 pts without it; CLAHE alone -5.1 pts). Loss weighting at 3.0 is **harmful** vs 1.0 (+0.87 pts for pw=1); Dice-only slightly beats the published 0.5/0.5 mix. Morphology is cosmetic (-0.19 pts). BCE-only is neutral.

---

## 4. Controlled U-Net vs UNet++ vs DeepLabV3+ (mixed-only, identical protocol)

| Architecture (mixed-only) | Status | IoU | Dice | Acc | HD95 | BF1 |
|---|---:|---:|---:|---:|---:|---:|
| U-Net/ResNet18 (this study, mean±std) | **done** | 0.9217±0.0035 | 0.9583 | 0.9646 | 0.3376 | 0.9418 |
| UNet++/ResNet18 | queued (next after DeepLab) | — | — | — | — | — |
| DeepLabV3+/ResNet18 | **running** (ep71, ~06:51 UTC) | — | — | — | — | — |

Pre-existing numbers in `baselines/comparison_table.json` (UNet++ 0.8861, DeepLabV3+ 0.8558) were trained on **COMBINED** psr+sunlit+mixed data and are **not comparable** to the mixed-only headline — they must not be cited as a controlled comparison. Controlled results will appear in `results/architecture_comparison.csv` as `architecture_*_seed42` rows when jobs finish (~3 h for DeepLab, ~4-5 h for UNet++ on this CPU).

---

## 5. Which paper claims remain supported

- The pipeline is real and reproducible (≤0.002 deviation on re-execution; peak val IoU identical to 7 sig figs).
- U-Net strongly outperforms every *independent* classical baseline (ΔIoU ≥0.12 vs RF 0.796, ≥0.22 vs Otsu 0.693).
- Metrics correctly measure agreement with algorithmic pseudo-labels (pseudo-label quality analysis: self-consistent, morphology-insensitive, paradigm-dependent).
- Errors are boundary-dominated (81.6% of FN within 2 px) — error analysis holds.
- Geographic generalization is strong (≤1.2-point drop on clean splits).

## 6. Which claims must be removed

- "Crater-disjoint held-out validation" — train.npy contains ~4,250 shackleton_02 patches per class (audit §4.1).
- Current ablation table and any ΔIoU values derived from it (unsupported 12%-data regime).
- Any sentence claiming UNet++ / DeepLabV3+ is inferior (until controlled reruns finish).
- Single-seed headline as the sole reported number — replace with 5-seed aggregate.
- Multi-Otsu as an "independent baseline" — it defines the reference (self-IoU 1.0); report it only as identity.
- Any suggestion that fallback_threshold 0.0484 is tuned/validated — it never fires (0% trigger rate).

## 7. Exact numbers to replace paper tables

**Main results table** (replace `paper.tex` tab:results):

| Method | IoU | Dice | Accuracy | HD95 (px) | BF1 |
|---|---:|---:|---:|---:|---:|
| U-Net/ResNet18 (5-seed mean±std) | 0.922±0.004 | 0.958±0.002 | 0.965±0.002 | 0.34±0.05 | 0.942±0.005 |
| U-Net/ResNet18 (seed 42, paper-faithful) | 0.9202 | 0.9574 | 0.9636 | 0.3451 | 0.9391 |
| Random Forest (strongest independent baseline) | 0.7964 | 0.8787 | 0.9034 | 1.6475 | 0.8189 |
| Otsu (per-patch) | 0.6927 | 0.8067 | 0.8643 | 3.3154 | 0.6310 |
| Global threshold 0.10 | 0.6003 | 0.7064 | 0.8240 | 5.6013 | 0.5388 |

Do not tabulate Multi-Otsu+morphology (1.000 — identity).

**Cross-region table** (new):

| Train | Test | IoU |
|---|---:|---:|
| shackleton_01+cabeus_01 (legacy, contaminated — for reference) | shackleton_02 | 0.9202 |
| shackleton_01+shackleton_02 | cabeus_01 | 0.9092 |
| cabeus_01 | shackleton_01+shackleton_02 | 0.9098 |

**Ablation table** — use §3 rows verbatim (reference 0.9202 baseline).

## 8. Deterministic validation preprocessing

**Yes — switch to deterministic (no-transform) validation for the revision.**
The published stochastic `clip_limit~U(1,4)` costs ~1.4 IoU points and injects noise into every number (repro clean IoU 0.9338 vs 0.9202; 5-seed clean mean 0.9347±0.0048). Keep the stochastic pipeline only as an augmentation during training. Report clean-protocol numbers as primary; optionally note the stochastic variant in a footnote for backward comparability. No code change beyond `get_val_transforms()` → `None` at eval time (already saved as `*_clean` metrics).

## 9. Sufficient for submission before expert annotation?

**Almost — not yet on two items.** The 5-seed, cross-region, and full-data ablation evidence now in hand are sufficient for the *pseudo-label consistency* story. However, (a) the controlled architecture comparison is still pending (DeepLab running, UNet++ queued) — without it any architecture-superiority sentence must be removed, which reviewers will notice; (b) the clean strip-disjoint cross-region result is currently single-seed — acceptable for a revision but 5-seed per split would be stronger. Expert annotation is **not** a blocker for a methods-focused submission *provided* the paper retains its explicit caveat that metrics measure pseudo-label agreement, not geological semantics. If the claim is upgraded to semantic PSR detection, expert evaluation becomes mandatory — the 80-patch export is ready at `results/expert_evaluation/annotation_export/` and evaluation is a one-command step once masks arrive.

## 10. Minimum remaining work before submission

1. Let the queue finish DeepLabV3+ (~2 h) and UNet++ (~4 h) → update `results/architecture_comparison.csv` and fill §4.
2. Re-render `paper.tex` tables from §§1-3,7 and delete/replace the unsupported tables/claims listed in §6.
3. Flip evaluation to deterministic preprocessing (one-line change) and re-export the headline numbers from the `clean_*` columns (optional but recommended).
4. (Optional, strengthens revision) run 5-seed for splits B and C (currently single-seed) — ~17 h additional if desired; can be listed as future work if time-constrained.
5. Keep `results/queue/queue.txt` + `experiments/queue_runner.py` running until `queue.txt` empty and `done.txt` shows all 13 jobs OK; then re-run `python experiments/aggregate_results.py`.

Resume/pause: `python experiments/queue_runner.py` (resumes from remaining `queue.txt`). To pause, terminate the `queue_runner.py` python process (PID 42676) and the current training child (PID 4192) — progress for the current epoch is lost but best checkpoint on disk is preserved. Full suite estimated remaining: ~6 h.

## Queue status at pause

```
done: classical_A_strip, classical_C, classical_B, multiseed_123/456/789/2026, crossB, crossC, ablation x6 (6/6 done)
running: arch_deeplabv3plus_seed42 (ep71)
queued: arch_unetplusplus_seed42
```

No results were fabricated; every number above is traceable to `results/all_results.json` row or `results/experiments/<id>_*` directory.

Do not edit `paper.tex` until the two architecture rows land — then revise only from the tables in §§1-4,7.
