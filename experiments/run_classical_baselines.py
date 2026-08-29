"""Phase 3: strong classical baselines evaluated on exactly the same held-out
data as the neural model, against identical LUCID pseudo-label references.

Methods
  global_threshold   : patch > 0.10 (paper baseline)
  otsu               : per-patch 2-class Otsu, robust degenerate handling
  multi_otsu         : LUCID mixed mechanism (3-class MOtsu, lowest thr,
                       ValueError -> fallback 0.0484), no morphology
  multi_otsu_morph   : + closing(disk(1)) == exact LUCID mixed pseudo-label
  adaptive           : local threshold; params selected ON TRAINING DATA ONLY
  random_forest      : pixel-level RF on local intensity/texture features,
                       trained ONLY on training-split pixels

Usage:
  python experiments/run_classical_baselines.py                # legacy split
  python experiments/run_classical_baselines.py --split-preset B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_multiotsu, threshold_local
from skimage.morphology import closing, disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from framework import (RESULTS_DIR, SPLIT_PRESETS, load_config,           # noqa: E402
                       make_logger, register_result, stamp)
from src.data.mask_generator import FALLBACK_THRESHOLD, generate_mask, clean_mask  # noqa: E402
from src.evaluate import Evaluator                                        # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-preset", default="legacy",
                   choices=["legacy", "A_strip", "B", "C"])
    p.add_argument("--classes", default="mixed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-param-sample", type=int, default=1500)
    p.add_argument("--rf-train-patches", type=int, default=400)
    p.add_argument("--rf-max-pixels", type=int, default=600_000)
    p.add_argument("--skip-rf", action="store_true")
    p.add_argument("--methods", default=None,
                   help="comma list among global,otsu,multiotsu,morph,"
                        "adaptive,rf; default all")
    return p.parse_args()


# ---------------------------------------------------------------- references
def build_reference(patches_dir: Path, sources, clean: bool):
    """Returns (N,64,64) patches and matching LUCID pseudo-label masks."""
    parts_p, parts_m = [], []
    for cls, name in sources:
        arr = np.load(patches_dir / cls / f"{name}.npy").astype(np.float32)
        masks = np.zeros_like(arr, dtype=np.float32)
        for i in range(len(arr)):
            m, _ = generate_mask(arr[i], cls, None)
            if clean:
                m = clean_mask(m)
            masks[i] = m
        parts_p.append(arr)
        parts_m.append(masks)
    return np.concatenate(parts_p), np.concatenate(parts_m)


# ------------------------------------------------------------------- methods
def pred_global(patch: np.ndarray, t: float = 0.10) -> np.ndarray:
    return (patch > t).astype(np.float32)


def pred_otsu(patch: np.ndarray) -> np.ndarray:
    if patch.std() < 1e-6:                      # constant patch
        return np.zeros_like(patch, dtype=np.float32)
    try:
        t = threshold_otsu(patch)
    except ValueError:                          # pragma: no cover
        return np.zeros_like(patch, dtype=np.float32)
    if not (patch.min() <= t <= patch.max()):
        t = 0.5 * (patch.min() + patch.max())   # degenerate range guard
    return (patch > t).astype(np.float32)


def pred_multi_otsu(patch: np.ndarray, morph: bool) -> np.ndarray:
    m, _ = generate_mask(patch, "mixed", None)  # repo implementation verbatim
    if morph:
        m = clean_mask(m)
    return m.astype(np.float32)


def pred_adaptive(patch: np.ndarray, window: int, offset: float) -> np.ndarray:
    t = threshold_local(patch, block_size=window, method="gaussian",
                        offset=offset)
    return (patch > t).astype(np.float32)


FEATURE_NAMES = ["intensity", "local_mean7", "local_std7", "sobel_x",
                 "sobel_y", "grad_mag", "local_contrast7"]


def extract_features(patches: np.ndarray) -> np.ndarray:
    """(N,64,64) -> (N*4096, F) float32 features."""
    n = len(patches)
    feats = np.empty((n, 64, 64, len(FEATURE_NAMES)), dtype=np.float32)
    sx = ndimage.sobel(patches, axis=2)
    sy = ndimage.sobel(patches, axis=1)
    lm = ndimage.uniform_filter(patches, size=(1, 7, 7))
    sq = ndimage.uniform_filter(patches ** 2, size=(1, 7, 7))
    ls = np.sqrt(np.maximum(sq - lm ** 2, 0))
    lc = (ndimage.maximum_filter(patches, size=(1, 7, 7))
          - ndimage.minimum_filter(patches, size=(1, 7, 7)))
    feats[..., 0] = patches
    feats[..., 1] = lm
    feats[..., 2] = ls
    feats[..., 3] = sx
    feats[..., 4] = sy
    feats[..., 5] = np.hypot(sx, sy)
    feats[..., 6] = lc
    return feats.reshape(-1, len(FEATURE_NAMES))


def select_adaptive_params(train_patches, train_ref, sample_n, rng) -> dict:
    idx = rng.choice(len(train_patches), size=min(sample_n, len(train_patches)),
                     replace=False)
    grid = [(w, o) for w in (7, 15, 31, 51) for o in (0.0, 0.005, 0.01)]
    scores = {}
    for w, o in grid:
        ious = []
        for i in idx:
            p = train_patches[i]
            if p.std() < 1e-6:
                ious.append(float((train_ref[i].sum() == 0)))
                continue
            pred = pred_adaptive(p, w, o)
            inter = np.logical_and(pred > 0.5, train_ref[i] > 0.5).sum()
            union = np.logical_or(pred > 0.5, train_ref[i] > 0.5).sum()
            ious.append(inter / union if union else 1.0)
        scores[f"w{w}_o{o}"] = float(np.mean(ious))
    best = max(scores, key=scores.get)
    bw, bo = best.split("_")
    return {"selected": {"window": int(bw[1:]), "offset": float(bo[1:]),
                         "method": "gaussian"},
            "train_only_scores": scores}


def eval_masks(name: str, preds: np.ndarray, refs: np.ndarray, out_dir: Path,
               tag: str = "legacy"):
    ev = Evaluator()
    ev.update(torch_from(preds), torch_from(refs))
    agg = {k: {m: float(v) for m, v in d.items()}
           for k, d in ev.compute_aggregate().items()}
    flat = {f"{k}_mean": agg[k]["mean"] for k in
            ("iou", "dice", "pixel_accuracy", "hd95", "boundary_f1")}
    pooled = {
        "tp": int(np.logical_and(preds > .5, refs > .5).sum()),
        "fp": int(np.logical_and(preds > .5, refs < .5).sum()),
        "fn": int(np.logical_and(preds < .5, refs > .5).sum()),
        "tn": int(np.logical_and(preds < .5, refs < .5).sum()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "predictions_uint8.npz",
                        preds=preds.astype(np.uint8))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"method": name, "aggregate": agg,
                   "global_confusion_counts": pooled, **flat}, f, indent=2)
    try:
        from framework import register_result, stamp
        register_result({
            "exp_id": f"classical/{name}/{tag}",
            "timestamp": stamp(), "arch": "classical", "encoder": "-",
            "split_preset": tag, "classes": "mixed", "seed": 42,
            "pseudo_label_method": "multiotsu3_lowest+closing_disk1",
            "augmentation": "none", "loss": "-", "optimizer": "-",
            "iou": round(flat["iou_mean"], 4),
            "dice": round(flat["dice_mean"], 4),
            "accuracy": round(flat["pixel_accuracy_mean"], 4),
            "hd95": round(flat["hd95_mean"], 4),
            "bf1": round(flat["boundary_f1_mean"], 4),
        })
    except Exception:
        pass
    return flat


def torch_from(x):
    import torch
    return torch.from_numpy(x[:, np.newaxis].astype(np.float32))


def main():
    args = parse_args()
    cfg = load_config()
    patches_dir = Path(cfg["data"]["base_dir"]) / "patches"
    preset = SPLIT_PRESETS[args.split_preset]
    tag = args.split_preset
    base = RESULTS_DIR / "classical"
    logger = make_logger(base / f"run_{tag}_{stamp()}.log")

    rng = np.random.default_rng(args.seed)
    wanted = (set(args.methods.split(",")) if args.methods
              else {"global", "otsu", "multiotsu", "morph", "adaptive", "rf"})

    logger.info("building references for split %s ...", tag)
    train_sources = [(c.strip(), n) for c, n in preset["train_sources"]]
    test_sources = [(c.strip(), n) for c, n in preset["final_test_sources"]]

    # NOTE: parameter selection uses TRAINING data exclusively.
    tr_patches, tr_ref = build_reference(patches_dir, train_sources, clean=True)
    te_patches, te_ref = build_reference(patches_dir, test_sources, clean=True)
    logger.info("train %d patches | test %d patches",
                len(tr_patches), len(te_patches))

    results = {}

    def run_method(name, fn, morph=False, chunk=512):
        t0 = time.time()
        preds = np.empty_like(te_patches)
        for s in range(0, len(te_patches), chunk):
            for i in range(s, min(s + chunk, len(te_patches))):
                preds[i] = fn(te_patches[i])
        flat = eval_masks(f"{name}", preds, te_ref,
                          base / tag / name.replace(" ", "_"), tag=tag)
        flat["runtime_s"] = round(time.time() - t0, 1)
        results[name] = flat
        logger.info("%-24s IoU %.4f Dice %.4f Acc %.4f HD95 %.4f BF1 %.4f (%.1fs)",
                    name, flat["iou_mean"], flat["dice_mean"],
                    flat["pixel_accuracy_mean"], flat["hd95_mean"],
                    flat["boundary_f1_mean"], flat["runtime_s"])

    if "global" in wanted:
        run_method("Global threshold 0.10",
                   lambda p: pred_global(p, 0.10))
    if "otsu" in wanted:
        run_method("Otsu (per-patch)", pred_otsu)
    if "multiotsu" in wanted:
        run_method("Multi-Otsu",
                   lambda p: pred_multi_otsu(p, morph=False))
    if "morph" in wanted:
        run_method("Multi-Otsu + morphology",
                   lambda p: pred_multi_otsu(p, morph=True))

    # ---- adaptive: params chosen on TRAINING data only ---------------------
    if "adaptive" in wanted:
        sel = select_adaptive_params(tr_patches, tr_ref,
                                     args.train_param_sample, rng)
        (base / "adaptive" / "param_selection_trainonly.json").parent.mkdir(
            parents=True, exist_ok=True)
        with open(base / "adaptive" / "param_selection_trainonly.json", "w") as f:
            json.dump(sel, f, indent=2)
        w, o = sel["selected"]["window"], sel["selected"]["offset"]
        logger.info("adaptive params (train-only): window=%d offset=%.3f", w, o)

        def adaptive_fn(p):
            if p.std() < 1e-6:
                return np.zeros_like(p, dtype=np.float32)
            return pred_adaptive(p, w, o)

        run_method("Adaptive threshold", adaptive_fn)

    # ---- Random Forest ------------------------------------------------------
    if "rf" in wanted and not args.skip_rf:
        from sklearn.ensemble import RandomForestClassifier
        t0 = time.time()
        ridx = rng.choice(len(tr_patches),
                          size=min(args.rf_train_patches, len(tr_patches)),
                          replace=False)
        Xtr = extract_features(tr_patches[ridx])
        ytr = (tr_ref[ridx] > 0.5).reshape(-1)
        if len(ytr) > args.rf_max_pixels:
            keep = rng.choice(len(ytr), size=args.rf_max_pixels, replace=False)
            Xtr, ytr = Xtr[keep], ytr[keep]
        rf = RandomForestClassifier(n_estimators=100, n_jobs=8,
                                    random_state=args.seed)
        rf.fit(Xtr, ytr)
        n_rf_pixels = int(len(ytr))
        del Xtr, ytr

        def rf_fn(p):
            f = extract_features(p[None])
            return rf.predict(f).reshape(64, 64).astype(np.float32)

        preds = np.empty_like(te_patches)
        for i in range(len(te_patches)):
            preds[i] = rf_fn(te_patches[i])
        flat = eval_masks("Random Forest", preds, te_ref,
                          base / tag / "random_forest", tag=tag)
        flat["n_train_pixels"] = n_rf_pixels
        flat["runtime_s"] = round(time.time() - t0, 1)
        results["Random Forest"] = flat
        logger.info("%-24s IoU %.4f Dice %.4f Acc %.4f HD95 %.4f BF1 %.4f",
                    "Random Forest", flat["iou_mean"], flat["dice_mean"],
                    flat["pixel_accuracy_mean"], flat["hd95_mean"],
                    flat["boundary_f1_mean"])
        import joblib
        joblib.dump(rf, base / tag / "random_forest" / "rf_model.joblib")

    with open(base / f"classical_summary_{tag}.json", "w") as f:
        json.dump({"split": tag, "fallback_threshold_default":
                   FALLBACK_THRESHOLD, "results": results}, f, indent=2)

    for name, r in results.items():
        register_result({
            "exp_id": f"classical/{name}/{tag}", "timestamp": stamp(),
            "arch": "classical", "encoder": "-", "split_preset": tag,
            "classes": args.classes, "seed": args.seed,
            "pseudo_label_method": "multiotsu3_lowest+closing_disk1",
            "augmentation": "none", "loss": "-", "optimizer": "-",
            "notes": json.dumps({k: round(v, 4) for k, v in r.items()}),
            "iou": round(r["iou_mean"], 4), "dice": round(r["dice_mean"], 4),
            "accuracy": round(r["pixel_accuracy_mean"], 4),
            "hd95": round(r["hd95_mean"], 4),
            "bf1": round(r["boundary_f1_mean"], 4),
        })


if __name__ == "__main__":
    main()
