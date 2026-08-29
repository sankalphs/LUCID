# LUCID Publication-Readiness Experiment Report

Date: 2026-08-26 · Machine: AMD Ryzen AI 9 HX 370, CPU-only (12 torch threads), 23 GB RAM
Companion docs: `docs/EXPERIMENT_AUDIT.md`, `docs/REPRODUCTION_REPORT.md`, `docs/REPRODUCIBILITY.md`

---

## 1. Reproduction

**Status: COMPLETE — successful.**

| Metric | Paper | Re-executed (seed 42) | abs diff |
|---|---|---|---|
| IoU | 0.9203 | 0.9202 | 0.0001 |
| Dice | 0.9576 | 0.9574 | 0.0002 |
| Accuracy | 0.9640 | 0.9636 | 0.0004 |
| HD95 | 0.3454 px | 0.3451 px | 0.0003 |
| Boundary F1 | 0.9373 | 0.9391 | 0.0018 |

Best-epoch global val IoU matched to ~2e-7 (0.92292702 @ep56 vs original
0.92292719 @ep42); differences attributable to documented oneDNN/RNG
nondeterminism. Additionally, the archived checkpoint reproduces results.json
exactly under the current metric code (`results/reproduction/checkpoint_reeval.json`).

## 2. Classical baselines (agreement with LUCID pseudo-labels)

**Status: COMPLETE on all four split definitions.**

Legacy val split (mixed/val.npy, 2,007 patches):

| Method | IoU | Dice | Accuracy | HD95 | BF1 |
|---|---:|---:|---:|---:|---:|
| Global threshold 0.10 | 0.6003 | 0.7064 | 0.8240 | 5.6013 | 0.5388 |
| Otsu (per-patch) | 0.6927 | 0.8067 | 0.8643 | 3.3154 | 0.6310 |
| Multi-Otsu | 0.9784 | 0.9891 | 0.9902 | 0.0010 | 1.0205* |
| Multi-Otsu + morphology† | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| Adaptive threshold‡ | 0.6020 | 0.7454 | 0.7982 | 10.4819 | 0.5407 |
| Random Forest§ | 0.7964 | 0.8787 | 0.9034 | 1.6475 | 0.8189 |
| **U-Net/ResNet18 (reproduced)** | **0.9202** | **0.9574** | **0.9636** | **0.3451** | **0.9391** |

\* BF1 > 1 possible for near-perfect masks via recall-side inflation in
`src/evaluate.py` (documented audit §4.7).
† Self-referential by construction: this IS the reference generator; it
demonstrates identity, not performance.
‡ Gaussian local threshold, window=51 offset=0 selected on TRAINING data only.
§ 100 trees, trained on pixels of 400 training-split patches (≤600k pixels);
no validation pixels used.

Interpretation: the learned U-Net beats every *independent* classical method by
≥0.12 IoU; the only "baseline" matching it is the pseudo-label generator itself,
which is not an independent comparison. Random Forest (train-only pixels) is the
strongest independent baseline at 0.796 IoU — simple intensity/texture features
do NOT approach U-Net performance. Plot: `results/plots/classical_vs_neural.png`.

Geographic stability of classical methods (test = region strip):

| Method | A_strip test sh02 | B test cabeus_01 | C test sh01+sh02 |
|---|---:|---:|---:|
| Global 0.10 | 0.6393 | 0.6002 | 0.6092 |
| Otsu | 0.7027 | 0.7043 | 0.6979 |
| Random Forest | 0.7995 | 0.7936 | 0.7902 |

(Unsupervised/patch-level baselines are near-invariant across regions, as
expected; the discriminative geographic test applies to the trained model.)

## 3. Multi-seed results

**Status: IN PROGRESS — seed 42 complete; seeds 123/456/789/2026 queued and
executing sequentially (~3.5 h each on this CPU).**

Current registry state (n=1): seed 42 → IoU 0.9202 / Dice 0.9574 / Acc 0.9636 /
HD95 0.3451 / BF1 0.9391; best epoch 56; final epoch 71; 211.7 min.
Aggregate table auto-updates as runs land:
`results/multiseed_summary.csv`, `results/multiseed_per_seed.csv`,
`results/plots/multiseed_metrics.png`.

The paper should report mean ± std across these five seeds once complete;
the single-seed 0.9203 should be presented as one draw from that distribution.

## 4. Cross-region generalization

**Status: PARTIALLY COMPLETE.**

Critical audit finding first: the materialized train.npy contains ~4,250
byte-identical shackleton_02 patches per class ⇒ the published "held-out
crater" claim does NOT hold at patch level (audit §4.1). Region-disjoint
splits were therefore rebuilt strictly from per-strip arrays:

| Split | Train | Test |
|---|---|---|
| A_strip | shackleton_01 + cabeus_01 | shackleton_02 |
| B | shackleton_01 + shackleton_02 | cabeus_01 |
| C | cabeus_01 | shackleton_01 + shackleton_02 |

Classical references computed for all splits (§2 table above). U-Net training
runs for splits B and C are queued behind the multiseed block
(crossB_seed42, crossC_seed42); early stopping uses a 10 % held-out fraction
of the TRAIN strips only, and the reported test metrics come from a single
evaluation of the selected checkpoint. `results/cross_region_summary.csv`
auto-updates.

## 5. Architecture comparison

**Status: EXISTING EVIDENCE ONLY — fair reruns queued.**

Pre-existing runs (`baselines/*.json`, full 100-epoch budget, seed 42) exist
for U-Net++ (IoU 0.8861) and DeepLabV3+ (IoU 0.8558), BUT they were trained on
COMBINED psr+sunlit+mixed data while the headline U-Net used mixed-only
(audit §4.2) — so the existing numbers are NOT a controlled comparison and must
not be quoted as such. Controlled mixed-only reruns
(`arch_unetplusplus_seed42`, `arch_deeplabv3plus_seed42`) are queued last
(~13 h and ~28 h estimated on this CPU). `results/architecture_comparison.csv`
labels provenance of every row.

## 6. Full-data ablations

**Status: QUEUED (not yet executed).**

The paper's ablation table (≈0.85–0.88 IoU deltas) is unsupported by any
recorded experiment: the recorded ablation JSONs used a 12 %-data/30-epoch
regime through the double-sigmoid code path and scored IoU 0.44–0.55
(audit §4.3 — the single most serious integrity finding). Six controlled
full-data/single-change runs (no_clahe, no_augmentation, bce_only, dice_only,
pos_weight=1, no_morphology) are queued against the reproduced Full-LUCID
reference (0.9202). Output: `results/ablation_summary.csv`.

## 7. Fallback threshold sensitivity

**Status: COMPLETE (mask-level analysis).**

Headline finding: `threshold_multiotsu(classes=3)` never raises ValueError on
any real mixed patch (trigger rate 0.000 on both train sample and full val) —
the 0.0484 fallback is currently DEAD CODE and has zero effect on labels.
Forced-path analysis (fallback applied to ALL patches, agreement vs the primary
Multi-Otsu label):

t:            0.020→0.482 · 0.030→0.565 · 0.040→0.704 · 0.0484→0.771 ·
              0.055→0.806 · **0.060→0.811 (train-selected)** · 0.070→0.800

Selection used TRAINING data only; validation curve tracks it within 0.002
throughout. The published 0.0484 sits in a flat region ≈0.04 IoU below the
forced-path optimum but is inconsequential in practice because the path never
fires. Artifacts: `results/fallback_threshold/fallback_sensitivity.json`,
`results/plots/fallback_threshold_sensitivity.png`.

## 8. Pseudo-label quality

**Status: COMPLETE (inter-method analysis; no expert labels exist).**

On mixed-class patches (val + 1,500-patch train sample):
- closing(disk(1)) vs pre-morphology Multi-Otsu: mean IoU **0.979** (p10 0.964)
  → morphology is cosmetic, not load-bearing.
- 2-class Otsu vs LUCID reference: mean IoU 0.696 → the 3-class mechanism does
  substantial work; the label definition is paradigm-dependent.
- Adaptive gaussian local threshold vs LUCID reference: mean IoU 0.472 →
  disagreement driven by darkness (Spearman ρ=-0.73): adaptive fragments dark
  floors that global Multi-Otsu calls shadow.
Conclusion: pseudo-labels are highly self-consistent but embody the global-
Multi-Otsu modeling assumption; they measure algorithmic consistency, not
verified geology. Artifacts: `results/pseudo_label_quality/`.

## 9. Expert evaluation

**Status: EXPORT READY — no annotations exist; none fabricated.**

No expert-labelled data was found anywhere in the repo (audit §9 of doc-dive).
Exported an annotation-ready set: 80 patches, 9 categories (clean dark/extremely
dark interiors, clean bright/bright-textured regions, mixed boundaries, crater
rims, boulder micro-shadows, scattered-light floor, high-albedo speckles),
balanced across SH01/SH02/Cabeus strips, with blinding (inputs separated from
pseudo-labels), instructions, manifest + SHA256:
`results/expert_evaluation/annotation_export/`,
`results/expert_evaluation/ANNOTATION_INSTRUCTIONS.md`.
When completed masks arrive (`<patch_id>_expert.png`), evaluation scripts
should compare pseudo-label-vs-expert, U-Net-vs-expert, RF-vs-expert on the
identical patch set.

## 10. Error analysis

**Status: COMPLETE (reproduction run).**

Worst-quartile failures concentrate in `scattered_light_floor_texture` (181/502)
and `bright_crater_rim_transitions` (74/502); speckle cases 15. Spatial
patterns: **81.6 % of false-negative pixels lie within 2 px of the reference
boundary** (boundary-localization errors dominate); false positives are mostly
isolated salt-noise components (median size 1 px, p90 5 px) consistent with
high-albedo speckles. Panels per category (Input | Reference | Prediction |
Error): `results/error_analysis/ex_*.png`; summary JSON alongside.

## 11. Reproducibility status

Fully reproducible now: reproduction run, all classical baselines, fallback
analysis, pseudo-label analyses, error analysis, annotation export
(one command each; see `docs/REPRODUCIBILITY.md` +
`python experiments/run_all_publication_experiments.py --list`).
Environment pinned via `requirements-lock.txt`; dataset checksums in
`results/data_checksums.sha256`. Caveats: approximate (not bit-exact)
re-execution due to unset cudnn determinism + oneDNN ordering; stochastic
val-time CLAHE is part of the published protocol and is preserved (with a
deterministic "clean" variant stored alongside every eval).

Not yet executed (queued, will complete unattended via
`results/queue/queue.txt` + `experiments/queue_runner.py`): multiseed ×4,
cross-region U-Net B/C, six full-data ablations, two fair architecture reruns.

## 12. Publication recommendation

Supported claims today:
1. The pipeline and its headline result are genuine and reproducible (≤0.002
   deviation, independently re-executed).
2. The learned model substantially outperforms every independent classical
   baseline (RF 0.796 / Otsu 0.693 / global 0.600 vs U-Net 0.920 IoU).
3. Metrics quantify agreement with algorithmic pseudo-labels whose reliability
   is now characterized (self-consistent, morphology-insensitive,
   paradigm-dependent) — the paper's own caveat is correct and should stay.
4. Errors are boundary-dominated (81.6 % of FN within 2 px), not gross
   misclassification.

NOT supported / must change before submission:
- The "crater-disjoint held-out validation" claim (train contains ~⅓
  shackleton_02 content) — replace with rebuilt strip-disjoint results or
  reword honestly.
- The ablation table (unsupported numbers; regime mismatch) — must be replaced
  by the queued full-data ablations before citing component contributions.
- Architecture-superiority statements based on the existing combined-data runs.
- Single-seed reporting — wait for the 5-seed aggregate.
- The 12 %-data ablation numbers in report.md/paper.tex conflict with recorded
  JSONs and must be removed regardless.

Verdict: evidence is strong for the core pseudo-label-consistency result but
NOT yet sufficient for submission until multiseed, region-disjoint U-Net
results, and honest ablations are in (est. ~30 CPU-hours of queued work).

Three highest-priority remaining actions:
1. Let the queue finish multiseed (123/456/789/2026) → report mean ± std.
2. Run cross-region U-Net splits B & C (queued next) → substantiate
   generalization claims on clean splits.
3. Replace the paper's ablation table with the six queued full-data ablations;
   pursue expert annotation of the prepared 80-patch export to decouple
   pseudo-label consistency from visual/geological correctness.
