"""Phase 11: error analysis for the reproduction U-Net run on the legacy
mixed/val split. Categorizes failure patches into the paper's error modes,
saves Input | Reference | Prediction | Difference panels, and quantifies
FP/FN spatial patterns.

Run AFTER reproduction training has finished (needs predictions_test_valclahe.npz).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mask_generator import generate_mask, clean_mask  # noqa: E402


def find_repro_dir() -> Path:
    base = PROJECT_ROOT / "results" / "experiments"
    cands = sorted(base.glob("repro_seed42_*"))
    if not cands:
        raise SystemExit("no repro_seed42_* experiment directory found")
    return cands[-1]


def patch_stats(p: np.ndarray) -> dict:
    sx = ndimage.sobel(p, axis=1)
    sy = ndimage.sobel(p, axis=0)
    edge = np.hypot(sx, sy) > 0.02
    return {
        "mean": float(p.mean()),
        "std": float(p.std()),
        "p95": float(np.percentile(p, 95)),
        "p99": float(np.percentile(p, 99)),
        "edge_density": float(edge.mean()),
        "dark_frac": float((p < 0.05).mean()),
        "speckle": float(np.percentile(p, 99) - np.percentile(p, 95)),
        "micro_shadows": float((p < p.mean() - 1.5 * p.std()).mean()),
    }


def categorize(s: dict) -> str:
    if s["mean"] < 0.01:
        return "extremely_dark_interiors"
    if s["speckle"] > 0.15 and s["p99"] > 0.5:
        return "high_albedo_speckles"
    if s["edge_density"] > 0.10 and s["mean"] > 0.10:
        return "bright_crater_rim_transitions"
    if s["micro_shadows"] > 0.05:
        return "boulder_micro_shadows"
    if 0.02 <= s["mean"] <= 0.08 and s["std"] < 0.04:
        return "scattered_light_floor_texture"
    return "other"


def panel(imgs, ref, pred, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    axes[0].imshow(imgs, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")
    axes[1].imshow(ref, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reference (pseudo-label)")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    diff = np.zeros((64, 64, 3))
    fn = (ref > 0.5) & (pred < 0.5)
    fp = (ref < 0.5) & (pred > 0.5)
    diff[..., 0] = fp          # red   = false positive (predicted bright)
    diff[..., 1] = ref > .5    # green = reference shadow
    diff[..., 2] = 0
    axes[3].imshow(diff)
    axes[3].set_title("Error (red=FP, green=missed FN)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    out_dir = PROJECT_ROOT / "results" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_dir = find_repro_dir()
    data = np.load(exp_dir / "predictions_test_valclahe.npz")["probs"]
    probs = data[:, 0].astype(np.float32)
    preds = (probs > 0.5).astype(np.float32)

    patches = np.load(PROJECT_ROOT / "dataset/kaggle_dataset/patches/"
                      "mixed/val.npy").astype(np.float32)
    refs = np.stack([clean_mask(generate_mask(p, "mixed", None)[0])
                     for p in patches])

    ious, stats, cats = [], [], []
    for i in range(len(patches)):
        inter = np.logical_and(preds[i] > .5, refs[i] > .5).sum()
        union = np.logical_or(preds[i] > .5, refs[i] > .5).sum()
        ious.append(inter / union if union else 1.0)
        s = patch_stats(patches[i])
        stats.append(s)
        cats.append(categorize(s))
    ious = np.array(ious)

    summary = {
        "experiment": exp_dir.name,
        "n_patches": int(len(patches)),
        "iou_mean": round(float(ious.mean()), 4),
        "iou_p10": round(float(np.percentile(ious, 10)), 4),
        "category_counts": {c: int(cats.count(c))
                            for c in set(cats)},
        "failure_category_counts": {},
    }

    failed = ious < np.percentile(ious, 25)   # worst quartile = failures
    fail_cats = [c for c, f in zip(cats, failed) if f]
    summary["failure_category_counts"] = {c: int(fail_cats.count(c))
                                          for c in set(fail_cats)}

    # representative examples per category among failures
    order = np.argsort(ious)
    for category in ["bright_crater_rim_transitions",
                     "scattered_light_floor_texture",
                     "boulder_micro_shadows",
                     "extremely_dark_interiors",
                     "high_albedo_speckles"]:
        picks = [i for i in order if failed[i] and cats[i] == category][:3]
        if not picks:
            picks = [i for i in np.argsort(-np.array(
                [1 if cats[j] == category else 0 for j in range(len(patches))]))
                [:1]]
        for rank, i in enumerate(picks):
            panel(patches[i], refs[i], preds[i],
                  f"{category} | val#{i} | IoU={ious[i]:.3f}",
                  out_dir / f"ex_{category}_{rank}.png")

    # FP / FN spatial patterns (global)
    fp_total = fn_total = fn_near_boundary = 0
    fp_sizes = []
    dist_to_ref_boundary_all = []
    from skimage.segmentation import find_boundaries
    for i in range(len(patches)):
        r, p = refs[i] > 0.5, preds[i] > 0.5
        fnm, fpm = r & ~p, ~r & p
        fp_total += int(fpm.sum()); fn_total += int(fnm.sum())
        bnd = find_boundaries(r, mode="outer")
        dt = ndimage.distance_transform_edt(~bnd)
        if fnm.any():
            dist_to_ref_boundary_all.append(dt[fnm])
            fn_near_boundary += int((dt[fnm] <= 2).sum())
        lab, n = ndimage.label(fpm)
        if n:
            fp_sizes.extend(ndimage.sum(np.ones_like(lab), lab,
                                        range(1, n + 1)).tolist())
    all_fn = sum(int(d.size) for d in dist_to_ref_boundary_all)
    summary["fp_fn_patterns"] = {
        "total_fp_pixels": fp_total,
        "total_fn_pixels": fn_total,
        "fn_within_2px_of_reference_boundary": fn_near_boundary,
        "fn_within_2px_fraction": round(fn_near_boundary / max(all_fn, 1), 4),
        "fp_component_count": len(fp_sizes),
        "fp_component_size_median": float(np.median(fp_sizes)) if fp_sizes else 0,
        "fp_component_size_p90": float(np.percentile(fp_sizes, 90))
        if fp_sizes else 0,
    }
    hist = np.concatenate(dist_to_ref_boundary_all) \
        if dist_to_ref_boundary_all else np.array([0])
    np.save(out_dir / "fn_distance_hist.npy", hist)

    with open(out_dir / "error_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
