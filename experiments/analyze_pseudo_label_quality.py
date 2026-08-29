"""Phase 9: Reliability analysis of algorithmic pseudo-labels (mixed class).

Compares alternative unsupervised labeling schemes against each other on
identical data. No method is treated as ground truth; V3 (Multi-Otsu-3
lowest threshold + morphological closing) is the *reference* actually used
by LUCID during training, so agreement with it quantifies how much the
pseudo-labels would change under a different thresholding choice.

Variants computed per patch:
    V1: Otsu (2-class), degenerate guard std < 1e-6 -> all-zero mask.
    V2: Multi-Otsu 3-class, lowest threshold (= LUCID pre-morphology,
        including its fixed-threshold fallback when Multi-Otsu fails).
    V3: V2 + closing(disk(1)) = exact LUCID training-time reference.
    V4: Adaptive Gaussian local threshold, window=31, offset=0.005.

Data: all mixed val.npy patches + a deterministic rng(42) sample of 1500
mixed train.npy patches.

Outputs (results/pseudo_label_quality/):
    agreement_stats.json
    disagreement_vs_patch_character.csv
    plots/hist_v2_vs_v3_iou.png
    plots/scatter_iou_v2v3_vs_std.png
    plots/panel_representative_patches.png
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from skimage.filters import threshold_local, threshold_otsu
from skimage.filters import sobel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.mask_generator import FALLBACK_THRESHOLD, clean_mask, generate_mask  # noqa: E402

PATCHES_DIR = REPO_ROOT / "dataset" / "kaggle_dataset" / "patches"
OUT_DIR = REPO_ROOT / "results" / "pseudo_label_quality"
PLOTS_DIR = OUT_DIR / "plots"

TRAIN_SAMPLE_N = 1500
SEED = 42
LOCAL_BLOCK = 31
LOCAL_OFFSET = 0.005


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    """IoU of two binary masks; 1.0 when both are empty (perfect agreement)."""
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    union = np.logical_or(pred_b, target_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_b, target_b).sum() / union)


def compute_v1(patch: np.ndarray) -> np.ndarray:
    if patch.std() < 1e-6:
        return np.zeros_like(patch)
    return (patch > threshold_otsu(patch)).astype(np.float32)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (patches, original_indices) for one split's mixed patches."""
    fname = "train.npy" if split == "train_sample" else f"{split}.npy"
    patches = np.load(PATCHES_DIR / "mixed" / fname)
    n = patches.shape[0]
    if split == "val":
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(SEED)
        take = min(TRAIN_SAMPLE_N, n)
        idx = np.sort(rng.choice(n, size=take, replace=False))
        patches = patches[idx]
    return patches, idx


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["val", "train_sample"]
    rows: list[dict] = []
    fallback_counts = {s: 0 for s in splits}

    for split in splits:
        patches, orig_idx = load_split(split)
        print(f"[{split}] processing {len(patches)} patches ...")
        for i, p in enumerate(patches):
            v1 = compute_v1(p)
            v2, method = generate_mask(p, "mixed", None)  # pre-morphology LUCID
            v3 = clean_mask(v2)  # exact LUCID reference used in training
            v4 = (p > threshold_local(p, LOCAL_BLOCK, method="gaussian",
                                      offset=LOCAL_OFFSET)).astype(np.float32)
            fallback_counts[split] += int(method == "fallback")

            rows.append({
                "split": split,
                "patch_index": int(orig_idx[i]),
                "iou_v1_v3": iou(v1, v3),
                "iou_v2_v3": iou(v2, v3),
                "iou_v4_v3": iou(v4, v3),
                "iou_v1_v2": iou(v1, v2),
                "iou_v2_v4": iou(v2, v4),
                "std": float(p.std()),
                "dark_fraction": float((p < FALLBACK_THRESHOLD).mean()),
                "edge_density": float((sobel(p) > 0.02).mean()),
                "used_fallback": int(method == "fallback"),
                "v2_illuminated_frac": float(v2.mean()),
            })

    n_total = len(rows)

    # ---- Agreement stats -------------------------------------------------
    def pair_stats(key: str) -> dict:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        return {
            "mean": round(float(vals.mean()), 6),
            "median": round(float(np.median(vals)), 6),
            "p10": round(float(np.percentile(vals, 10)), 6),
        }

    pairs = {k: pair_stats(k) for k in
             ("iou_v1_v3", "iou_v2_v3", "iou_v4_v3", "iou_v1_v2", "iou_v2_v4")}

    rates = {}
    counts = {}
    for s in splits:
        sub = [r["used_fallback"] for r in rows if r["split"] == s]
        counts[s] = len(sub)
        rates[s] = round(sum(sub) / len(sub), 6) if sub else 0.0
    total_fb = sum(r["used_fallback"] for r in rows)
    rates["overall"] = round(total_fb / n_total, 6)
    counts["n_val"] = counts["val"]
    counts["n_train_sample"] = counts["train_sample"]
    counts["total_patches"] = n_total
    counts["total_fallback"] = int(total_fb)

    # ---- Correlate disagreement with patch character ---------------------
    iou23 = np.array([r["iou_v2_v3"] for r in rows])
    correlations = {}
    for feat in ("std", "dark_fraction", "edge_density"):
        x = np.array([r[feat] for r in rows])
        pr = pearsonr(iou23, x)
        sr = spearmanr(iou23, x)
        correlations[feat] = {
            "pearson_r_vs_iou_v2_v3": round(float(pr.statistic), 4),
            "pearson_p": float(pr.pvalue),
            "spearman_rho_vs_iou_v2_v3": round(float(sr.statistic), 4),
        }
    # Same correlations for the adaptive-vs-reference disagreement.
    iou43 = np.array([r["iou_v4_v3"] for r in rows])
    for feat in ("std", "dark_fraction", "edge_density"):
        x = np.array([r[feat] for r in rows])
        correlations[f"{feat}__v4_vs_v3"] = {
            "pearson_r_vs_iou_v4_v3": round(float(pearsonr(iou43, x).statistic), 4),
            "spearman_rho_vs_iou_v4_v3": round(float(spearmanr(iou43, x).statistic), 4),
        }

    stats = {
        "description": ("Pseudo-label reliability: pairwise IoU between "
                        "alternative unsupervised labeling schemes. No method "
                        "is ground truth; V3 = exact LUCID training reference."),
        "variants": {
            "V1": "threshold_otsu, std<1e-6 guard -> zeros",
            "V2": "multiotsu(classes=3)[0] (=LUCID pre-morphology, w/ fallback)",
            "V3": "V2 + closing(disk(1)) (=exact LUCID training reference)",
            "V4": f"threshold_local gaussian block={LOCAL_BLOCK} offset={LOCAL_OFFSET}",
        },
        "both_empty_iou_convention": 1.0,
        "sampling": {"seed": SEED,
                     "train_sample_n": TRAIN_SAMPLE_N,
                     "train_pool_n": int(np.load(PATCHES_DIR / "mixed" / "train.npy").shape[0]),
                     "val_used_in_full": True},
        "counts": counts,
        "fallback_rates": rates,
        "pairs": pairs,
        "correlations_with_disagreement": correlations,
    }
    with open(OUT_DIR / "agreement_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ---- CSV -------------------------------------------------------------
    cols = ["split", "patch_index", "iou_v1_v3", "iou_v2_v3", "iou_v4_v3",
            "iou_v1_v2", "iou_v2_v4", "std", "dark_fraction",
            "edge_density", "used_fallback", "v2_illuminated_frac"]
    with open(OUT_DIR / "disagreement_vs_patch_character.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- Plots -----------------------------------------------------------
    # Histogram of V2 vs V3 IoU.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(iou23, bins=40, range=(0, 1), color="#4878a8", edgecolor="black",
            linewidth=0.4)
    ax.axvline(float(iou23.mean()), color="crimson", linestyle="--",
               label=f"mean={iou23.mean():.3f}")
    ax.set_xlabel("IoU(V2 multiotsu-lowest, V3 LUCID reference)")
    ax.set_ylabel("patch count")
    ax.set_title("Agreement of Multi-Otsu mask with exact LUCID pseudo-label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hist_v2_vs_v3_iou.png", dpi=150)
    plt.close(fig)

    # Scatter IoU(V2,V3) vs patch std.
    stds = np.array([r["std"] for r in rows])
    fb = np.array([r["used_fallback"] for r in rows], dtype=bool)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(stds[~fb], iou23[~fb], s=6, alpha=0.35, label="multi_otsu")
    ax.scatter(stds[fb], iou23[fb], s=10, alpha=0.5, marker="x",
               color="crimson", label="fallback")
    ax.set_xlabel("patch std")
    ax.set_ylabel("IoU(V2, V3)")
    ax.set_title("Mask disagreement vs patch contrast (std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scatter_iou_v2v3_vs_std.png", dpi=150)
    plt.close(fig)

    # Representative panel: 12 patches (4 lowest / 4 mid / 4 highest
    # agreement), rows = Input, V1, V2/V3 ref, |V2-V3| difference map.
    patches_by_row: dict[str, list[np.ndarray]] = {}
    masks_cache: dict[tuple, tuple] = {}
    for split in splits:
        patches, _ = load_split(split)
        patches_by_row[split] = patches
    order = np.argsort(iou23, kind="stable")
    sel = np.concatenate([order[:4],
                          order[(n_total // 2) - 2:(n_total // 2) + 2],
                          order[-4:]])

    def get_patch(row_idx: int) -> tuple[np.ndarray, str, int]:
        if row_idx < counts["val"]:
            return patches_by_row["val"][row_idx], "val", row_idx
        k = row_idx - counts["val"]
        return patches_by_row["train_sample"][k], "train_sample", k

    fig, axes = plt.subplots(4, 12, figsize=(19, 7))
    group_labels = ["lowest agreement", "median agreement", "highest agreement"]
    for col, ri in enumerate(sel):
        p, split, k = get_patch(int(ri))
        v1 = compute_v1(p)
        v2, _ = generate_mask(p, "mixed", None)
        v3 = clean_mask(v2)
        diff = np.abs(v2 - v3)
        views = [(p, "gray", 0.0, max(0.15, float(p.max()))),
                 (v1, "gray", 0, 1),
                 (v3, "gray", 0, 1),
                 (diff, "gray", 0, 1)]
        for r, (img, cmap, lo, hi) in enumerate(views):
            ax = axes[r, col]
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"IoU={iou23[ri]:.2f}\n{split}[{k}]", fontsize=7)
            if col == 0:
                ax.set_ylabel(["Input", "V1 Otsu2", "V3 LUCID ref",
                               "|V2-V3|"][r], fontsize=8)
        masks_cache[(split, k)] = (v1, v2, v3)
    for g, lab in enumerate(group_labels):
        axes[0, g * 4 + 1].annotate(lab, xy=(0.5, 1.45),
                                    xycoords="axes fraction", ha="center",
                                    fontsize=9, fontweight="bold",
                                    annotation_clip=False)
    fig.suptitle("Representative mixed patches by V2-vs-V3 agreement "
                 "(grayscale; diff shows where closing changed the mask)",
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS_DIR / "panel_representative_patches.png", dpi=150)
    plt.close(fig)

    # ---- Console summary --------------------------------------------------
    print("\n=== Pseudo-label quality summary ===")
    print(f"Patches analyzed: {counts['total_patches']} "
          f"(val={counts['n_val']}, train_sample={counts['n_train_sample']})")
    for key, s in pairs.items():
        print(f"  {key}: mean={s['mean']:.4f} median={s['median']:.4f} "
              f"p10={s['p10']:.4f}")
    print(f"Fallback rates (multiotsu ValueError): "
          f"val={rates['val']*100:.2f}%  "
          f"train_sample={rates['train_sample']*100:.2f}%  "
          f"overall={rates['overall']*100:.2f}%")
    print("Correlations vs IoU(V2,V3):")
    for feat in ("std", "dark_fraction", "edge_density"):
        c = correlations[feat]
        print(f"  {feat}: pearson_r={c['pearson_r_vs_iou_v2_v3']:.3f} "
              f"spearman_rho={c['spearman_rho_vs_iou_v2_v3']:.3f}")
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
