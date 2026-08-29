"""Phase 8: fallback-threshold sensitivity analysis.

Reality check first: threshold_multiotsu(classes=3) never raises ValueError on
the real mixed patches (measured fallback rate 0.00 - see
results/pseudo_label_quality/agreement_stats.json), so through the intended
mechanism the 0.0484 constant currently has NO effect on any label. To still
characterize sensitivity we (a) verify trigger rates across a grid, and
(b) FORCE the fallback path (bypassing multi-Otsu) to measure how masks would
shift if degenerate inputs did trigger it.

Selection rule (training data ONLY): pick the grid value whose forced-fallback
mask best agrees (mean IoU) with the primary Multi-Otsu label ON TRAINING
patches. Validation numbers are reported for transparency, never used for
selection.

Downstream U-Net probes are run separately via train_experiment.py
(--fallback-threshold + --force-fallback) under an identical reduced budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.morphology import closing, disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

GRID = [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.0484, 0.050, 0.055,
        0.060, 0.070]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-sample", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def load_mixed(patches_dir: Path, name: str):
    return np.load(patches_dir / "mixed" / f"{name}.npy").astype(np.float32)


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / "results" / "fallback_threshold"
    out_dir.mkdir(parents=True, exist_ok=True)

    patches_dir = Path("dataset/kaggle_dataset/patches")
    val = load_mixed(patches_dir, "val")
    rng = np.random.default_rng(args.seed)
    tr_idx = rng.choice(11372, size=min(args.train_sample, 11372),
                        replace=False)
    train = load_mixed(patches_dir, "train")[tr_idx]

    results = {"grid": GRID, "seed": args.seed,
               "n_train_sample": int(len(train)), "n_val": int(len(val)),
               "per_threshold": []}

    # natural trigger rates (real pipeline behaviour)
    def trigger_rate(arr):
        n = 0
        for p in arr:
            try:
                threshold_multiotsu(p, classes=3)
            except ValueError:
                n += 1
        return n / len(arr)

    results["natural_fallback_rate_train"] = trigger_rate(train)
    results["natural_fallback_rate_val"] = trigger_rate(val)

    # LUCID reference labels (multi-Otsu + closing), precomputed once
    ref_tr = np.stack([closing((p > threshold_multiotsu(p, classes=3)[0])
                               .astype(np.float32), disk(1))
                       if _ok(p) else (p > 0.0484).astype(np.float32)
                       for p in train])
    ref_va = np.stack([closing((p > threshold_multiotsu(p, classes=3)[0])
                               .astype(np.float32), disk(1))
                       if _ok(p) else (p > 0.0484).astype(np.float32)
                       for p in val])

    for t in GRID:
        rec = {"threshold": t}
        fb_tr = np.stack([_forced(p, t) for p in train])
        fb_va = np.stack([_forced(p, t) for p in val])
        ious_tr = [iou(fb_tr[i], ref_tr[i]) for i in range(len(train))]
        ious_va = [iou(fb_va[i], ref_va[i]) for i in range(len(val))]
        # dark-fraction shift vs reference (label-balance distortion)
        rec.update({
            "train_mean_iou_vs_primary": float(np.mean(ious_tr)),
            "train_p10_iou": float(np.percentile(ious_tr, 10)),
            "val_mean_iou_vs_primary": float(np.mean(ious_va)),
            "val_p10_iou": float(np.percentile(ious_va, 10)),
            "mean_dark_fraction_forced": float(fb_tr.mean(axis=(1, 2)).mean()),
            "mean_dark_fraction_primary": float(ref_tr.mean(axis=(1, 2)).mean()),
        })
        results["per_threshold"].append(rec)
        print(f"t={t:.3f} | train IoU-vs-primary {rec['train_mean_iou_vs_primary']:.4f}"
              f" | val {rec['val_mean_iou_vs_primary']:.4f}")

    best = max(results["per_threshold"],
               key=lambda r: r["train_mean_iou_vs_primary"])
    results["selected_threshold_train_only"] = best["threshold"]
    results["selection_rule"] = ("argmax train-side mean IoU of forced-fallback "
                                 "mask vs primary multi-Otsu+closing label")

    with open(out_dir / "fallback_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    plot(results, out_dir / ".." / "plots" /
         "fallback_threshold_sensitivity.png")


def _ok(p):
    try:
        threshold_multiotsu(p, classes=3)
        return True
    except ValueError:
        return False


def _forced(p, t):
    return closing((p > t).astype(np.float32), disk(1))


def plot(results, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    ts = [r["threshold"] for r in results["per_threshold"]]
    tr = [r["train_mean_iou_vs_primary"] for r in results["per_threshold"]]
    va = [r["val_mean_iou_vs_primary"] for r in results["per_threshold"]]
    sel = results["selected_threshold_train_only"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, tr, "o-", label="TRAIN (selection set)")
    ax.plot(ts, va, "s--", label="validation (report only)")
    ax.axvline(sel, color="crimson", ls=":",
               label=f"selected (train-only) t={sel}")
    ax.set_xlabel("forced fallback threshold")
    ax.set_ylabel("mean IoU vs primary Multi-Otsu label")
    ax.set_title("Fallback threshold sensitivity (forced-path analysis)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
