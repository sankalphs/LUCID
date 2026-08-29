"""Build the expert-annotation export for independent evaluation of
pseudo-label reliability (results/expert_evaluation/ANNOTATION_INSTRUCTIONS.md).

Candidates come ONLY from per-strip arrays (shackleton_01, shackleton_02,
cabeus_01) so strip provenance is clean; train.npy/val.npy mix strips and
are excluded. No human annotations exist: this script exports candidate
patches, ALGORITHMIC pseudo-labels, and instructions only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from skimage.filters import threshold_multiotsu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mask_generator import clean_mask, generate_mask  # noqa: E402

PATCHES_DIR = PROJECT_ROOT / "dataset" / "kaggle_dataset" / "patches"
DATASET_META_PATH = PROJECT_ROOT / "dataset" / "kaggle_dataset" / "metadata.json"
OUT_ROOT = PROJECT_ROOT / "results" / "expert_evaluation"
EXPORT_DIR = OUT_ROOT / "annotation_export"
IMAGES_DIR = EXPORT_DIR / "images"
MASKS_DIR = EXPORT_DIR / "masks"
BLINDED_DIR = EXPORT_DIR / "blinded"
SUBMISSIONS_DIR = OUT_ROOT / "submissions"

STRIPS = ("shackleton_01", "shackleton_02", "cabeus_01")
CLASSES = ("psr", "sunlit", "mixed")
STRIP_CODE = {"shackleton_01": "SH01", "shackleton_02": "SH02", "cabeus_01": "CB01"}
DATASET_ID = "flickeringtubelight/chandrayaan-2-ohrc-lunar-psrs"
SEED = 42
TOTAL_TARGET = 80
MIN_PER_CATEGORY = 8
EPS = 1e-6
EDGE_THRESH = 0.02
DARK_THRESH = 0.05
BRIGHT_THRESH = 0.3

CATEGORY_TARGETS = {
    "clean_dark_interior": 8,
    "extremely_dark_interior": 8,
    "clean_bright_region": 8,
    "bright_with_texture": 8,
    "mixed_boundary_clean": 10,
    "crater_rim_transition": 10,
    "boulder_micro_shadows": 10,
    "scattered_light_floor": 8,
    "high_albedo_speckles": 10,
}

CATEGORY_CRITERIA_TEXT = {
    "clean_dark_interior": "psr class; ranked by lowest intensity std.",
    "extremely_dark_interior": "psr class; ranked by lowest mean intensity.",
    "clean_bright_region": "sunlit class; ranked by mean - std (high mean, low variance).",
    "bright_with_texture": "sunlit class; ranked by highest std.",
    "mixed_boundary_clean": (
        "mixed class; finite otsu_gap, edge density within central quantiles, "
        "ranked by highest otsu_gap (strongest bimodality)."
    ),
    "crater_rim_transition": (
        "mixed/sunlit class; dark fraction in [0.05, 0.60]; top decile edge "
        "density, ranked descending."
    ),
    "boulder_micro_shadows": (
        "all classes; top decile of local-minima proxy "
        "(fraction of pixels < mean - 2*std), ranked descending."
    ),
    "scattered_light_floor": (
        "mixed class; low-to-mid mean, low otsu_gap (weak bimodality), "
        "ranked by lowest otsu_gap."
    ),
    "high_albedo_speckles": (
        "all classes; top decile of (p99 - p95) intensity spike, ranked descending."
    ),
}

STAT_COLS = [
    "mean", "std", "p5", "p50", "p95", "p99", "edge_density",
    "dark_fraction", "bright_fraction", "otsu_threshold", "otsu_gap",
    "local_min_proxy", "p99_minus_p95",
]

RAND_KEY: np.ndarray | None = None


def compute_stats(patches: np.ndarray) -> dict[str, np.ndarray]:
    n = patches.shape[0]
    flat = patches.reshape(n, -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    pct = np.percentile(flat, [5, 50, 95, 99], axis=1)
    p5, p50, p95, p99 = pct[0], pct[1], pct[2], pct[3]
    gx = sobel(patches, axis=1)
    gy = sobel(patches, axis=2)
    grad = np.hypot(gx, gy)
    edge = (grad > EDGE_THRESH).reshape(n, -1).mean(axis=1)
    dark = (flat < DARK_THRESH).mean(axis=1)
    bright = (flat > BRIGHT_THRESH).mean(axis=1)
    otsu_t = np.full(n, np.nan)
    for i in range(n):
        try:
            otsu_t[i] = threshold_multiotsu(patches[i], classes=3)[0]
        except ValueError:
            otsu_t[i] = np.nan
    span = p95 - p5
    gap = (otsu_t - p5) / (span + EPS)
    gap[~np.isfinite(gap)] = np.nan
    gap[span < 1e-3] = np.nan
    lm = (flat < (mean - 2.0 * std)[:, None]).mean(axis=1)
    spk = p99 - p95
    return {
        "mean": mean, "std": std, "p5": p5, "p50": p50, "p95": p95, "p99": p99,
        "edge_density": edge, "dark_fraction": dark, "bright_fraction": bright,
        "otsu_threshold": otsu_t, "otsu_gap": gap, "local_min_proxy": lm,
        "p99_minus_p95": spk,
    }


def load_pool() -> dict[str, np.ndarray]:
    cols: dict[str, list[np.ndarray]] = {k: [] for k in STAT_COLS}
    strips: list[str] = []
    clss: list[str] = []
    rows: list[int] = []
    pids: list[str] = []
    for c in CLASSES:
        for s in STRIPS:
            path = PATCHES_DIR / c / f"{s}.npy"
            arr = np.load(path, mmap_mode="r")[::1]
            st = compute_stats(np.asarray(arr, dtype=np.float32))
            n = st["mean"].shape[0]
            for k in STAT_COLS:
                cols[k].append(st[k])
            strips.extend([s] * n)
            clss.extend([c] * n)
            rows.extend(range(n))
            code = STRIP_CODE[s]
            pids.extend(f"{code}_{c}_{r:05d}" for r in range(n))
            print(f"  scanned {c}/{s}.npy: {n} patches")
    R: dict[str, np.ndarray] = {k: np.concatenate(v) for k, v in cols.items()}
    R["strip"] = np.array(strips)
    R["cls"] = np.array(clss)
    R["row"] = np.array(rows, dtype=np.int64)
    R["pid"] = np.array(pids)
    return R


def _pick(
    cand: list[int],
    score_of,
    group_of,
    target: int,
    taken: set[int],
) -> list[int]:
    avail = [i for i in cand if i not in taken]
    order = sorted(avail, key=lambda i: (-score_of(i), RAND_KEY[i]))
    groups: dict[object, list[int]] = {}
    for i in order:
        groups.setdefault(group_of(i), []).append(i)
    sel: list[int] = []
    keys = sorted(groups.keys(), key=str)
    changed = True
    while len(sel) < target and changed:
        changed = False
        for k in keys:
            g = groups[k]
            if g and len(sel) < target:
                sel.append(g.pop(0))
                changed = True
    taken.update(sel)
    return sel


def _sel_psr_low_std(R, pool, taken, target):
    cand = [i for i in pool if R["cls"][i] == "psr"]
    return _pick(cand, lambda i: -float(R["std"][i]), lambda i: R["strip"][i], target, taken)


def _sel_psr_low_mean(R, pool, taken, target):
    cand = [i for i in pool if R["cls"][i] == "psr"]
    return _pick(cand, lambda i: -float(R["mean"][i]), lambda i: R["strip"][i], target, taken)


def _sel_sunlit_clean(R, pool, taken, target):
    cand = [i for i in pool if R["cls"][i] == "sunlit"]
    return _pick(cand, lambda i: float(R["mean"][i] - R["std"][i]), lambda i: R["strip"][i], target, taken)


def _sel_sunlit_texture(R, pool, taken, target):
    cand = [i for i in pool if R["cls"][i] == "sunlit"]
    return _pick(cand, lambda i: float(R["std"][i]), lambda i: R["strip"][i], target, taken)


def _sel_mixed_boundary(R, pool, taken, target):
    mx = [i for i in pool if R["cls"][i] == "mixed" and np.isfinite(R["otsu_gap"][i])]
    if not mx:
        return []
    edges = np.array([R["edge_density"][i] for i in mx])
    gaps = np.array([R["otsu_gap"][i] for i in mx])
    cand: list[int] = []
    for elo, ehi, gq in ((0.25, 0.85, 0.60), (0.10, 0.95, 0.50), (0.00, 1.00, 0.40)):
        lo, hi = np.quantile(edges, [elo, ehi])
        gcut = np.quantile(gaps, gq)
        cand = [mx[j] for j in range(len(mx)) if lo <= edges[j] <= hi and gaps[j] >= gcut]
        if len(cand) >= target:
            break
    return _pick(cand, lambda i: float(R["otsu_gap"][i]), lambda i: R["strip"][i], target, taken)


def _sel_rim(R, pool, taken, target):
    base = [
        i for i in pool
        if R["cls"][i] in ("mixed", "sunlit")
        and DARK_THRESH <= R["dark_fraction"][i] <= 0.60
    ]
    if not base:
        return []
    edges = np.array([R["edge_density"][i] for i in base])
    cand: list[int] = []
    for q in (0.90, 0.80, 0.70, 0.60, 0.50):
        cut = np.quantile(edges, q)
        cand = [base[j] for j in range(len(base)) if edges[j] >= cut]
        if len(cand) >= target:
            break
    return _pick(cand, lambda i: float(R["edge_density"][i]),
                 lambda i: (R["strip"][i], R["cls"][i]), target, taken)


def _sel_boulder(R, pool, taken, target):
    vals = np.array([R["local_min_proxy"][i] for i in pool])
    cand: list[int] = []
    for q in (0.90, 0.80, 0.70, 0.60, 0.50):
        cut = np.quantile(vals, q)
        cand = [pool[j] for j in range(len(pool)) if vals[j] >= cut]
        if len(cand) >= target:
            break
    return _pick(cand, lambda i: float(R["local_min_proxy"][i]), lambda i: R["strip"][i], target, taken)


def _sel_scattered(R, pool, taken, target):
    mx = [i for i in pool if R["cls"][i] == "mixed" and np.isfinite(R["otsu_gap"][i])]
    if not mx:
        return []
    means = np.array([R["mean"][i] for i in mx])
    gaps = np.array([R["otsu_gap"][i] for i in mx])
    cand: list[int] = []
    for mq, gq in ((0.70, 0.25), (0.80, 0.30), (0.90, 0.40), (1.00, 0.50)):
        mcut = np.quantile(means, mq)
        gcut = np.quantile(gaps, gq)
        cand = [mx[j] for j in range(len(mx)) if means[j] <= mcut and gaps[j] <= gcut]
        if len(cand) >= target:
            break
    return _pick(cand, lambda i: -float(R["otsu_gap"][i]), lambda i: R["strip"][i], target, taken)


def _sel_speckles(R, pool, taken, target):
    vals = np.array([R["p99_minus_p95"][i] for i in pool])
    cand: list[int] = []
    for q in (0.90, 0.80, 0.70, 0.60, 0.50):
        cut = np.quantile(vals, q)
        cand = [pool[j] for j in range(len(pool)) if vals[j] >= cut]
        if len(cand) >= target:
            break
    return _pick(cand, lambda i: float(R["p99_minus_p95"][i]), lambda i: R["strip"][i], target, taken)


SELECTORS = {
    "clean_dark_interior": _sel_psr_low_std,
    "extremely_dark_interior": _sel_psr_low_mean,
    "clean_bright_region": _sel_sunlit_clean,
    "bright_with_texture": _sel_sunlit_texture,
    "mixed_boundary_clean": _sel_mixed_boundary,
    "crater_rim_transition": _sel_rim,
    "boulder_micro_shadows": _sel_boulder,
    "scattered_light_floor": _sel_scattered,
    "high_albedo_speckles": _sel_speckles,
}


def fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)) or v != v:
        return ""
    return f"{float(v):.6f}"


def save_png(arr: np.ndarray, path: Path) -> None:
    u8 = np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def contact_sheet(cat: str, idxs: list[int], R, patches_cache) -> None:
    n = len(idxs)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, 2 * ncols, figsize=(2 * ncols * 1.45, nrows * 1.65), squeeze=False
    )
    for r in range(nrows):
        for c in range(ncols):
            k = r * ncols + c
            ax_in = axes[r][2 * c]
            ax_mk = axes[r][2 * c + 1]
            for ax in (ax_in, ax_mk):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if k >= n:
                ax_in.axis("off")
                ax_mk.axis("off")
                continue
            i = idxs[k]
            patch, mask = patches_cache[i]
            ax_in.imshow(patch, cmap="gray", vmin=0.0, vmax=1.0)
            ax_in.set_title(R["pid"][i], fontsize=6)
            ax_mk.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
            ax_mk.set_title("pseudo", fontsize=6)
    fig.suptitle(
        f"{cat} (n={n}) - left: OHRC input, right: ALGORITHMIC pseudo-label "
        "(experts must NOT view before annotating)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(EXPORT_DIR / f"contact_sheet_{cat}.png", dpi=150)
    plt.close(fig)


def write_instructions(counts: dict[str, int]) -> None:
    lines = [
        "# Expert Annotation Instructions - LUCID Shadow Segmentation Evaluation",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} | "
        f"Dataset: {DATASET_ID}_",
        "",
        "## Purpose",
        "",
        "The project's current shadow/illuminated labels are **algorithmic pseudo-labels**",
        "(Multi-Otsu thresholding for `mixed` patches, fixed thresholds for `psr`/`sunlit`),",
        "**not** ground truth. Your task is an **independent expert visual interpretation**",
        "of a curated set of 64x64 Chandrayaan-2 OHRC patches so we can quantify how",
        "reliable the pseudo-labels are against human judgement. Label what YOU see;",
        "the quality of this evaluation depends on it.",
        "",
        "## BLINDING (important)",
        "",
        "- Annotate ONLY from the **blinded/** folder: `annotation_export/blinded/*.png`",
        "  (original input imagery, nothing else).",
        "- You **must NOT see the pseudo-labels** (`masks/`, the right-hand panels of the",
        "  `contact_sheet_*.png` files, or `manifest.csv`) before or during annotation.",
        "- Do not attempt to infer or reconstruct the algorithmic labels. Deviating from",
        "  them when your visual interpretation differs is exactly what we need.",
        "",
        "## Protocol",
        "",
        "1. Work through every PNG in `annotation_export/blinded/` (64x64 grayscale, [0,255]).",
        "2. Produce a **binary mask** marking EVERY pixel as:",
        "   - `SHADOW` = 0, `ILLUMINATED` = 255.",
        "3. Use any mask/paint editor, e.g. GIMP (paint black/white), labelme, ImageJ, or a",
        "   10-line Python/PIL script. Output must stay 64x64, single channel.",
        "4. Save as: `<patch_id>_expert.png` (same stem as the input file), e.g.",
        "   `SH02_mixed_00417.png` -> `SH02_mixed_00417_expert.png`.",
        "5. Place all files in `results/expert_evaluation/submissions/`.",
        "",
        "## Category-specific guidance",
        "",
        "The export was screened for visually tricky regimes. Whatever your instinct says",
        "for each case, **follow the visual appearance of the pixels - never the",
        "pseudo-label**:",
        "",
        "- **Penumbra / boundary transitions:** decide per pixel. A gradual falloff is not",
        "  automatically shadow; label a pixel illuminated if a reasonable eye reads direct",
        "  or indirect illumination. Do not snap your boundary to a convenient line.",
        "- **Boulder micro-shadows:** tiny dark cast shadows next to bright boulder tops are",
        "  genuinely mixed content. Label the individual shadow pixels as SHADOW and the",
        "  lit boulder surfaces as ILLUMINATED, even when fragments are 1-2 px wide.",
        "- **High-albedo speckles:** isolated bright specks inside dark terrain: label what",
        "  each pixel actually looks like; do not smooth speckles away because the region",
        "  'feels' dark.",
        "- **Very dark interiors:** if you cannot discern any illumination, everything is",
        "  SHADOW - that is a valid answer.",
        "- **Ambiguous pixels:** make your best single judgement; consistency matters more",
        "  than certainty.",
        "",
        "## File checklist per submission",
        "",
        "- Grayscale PNG, exactly 64x64, pixel values only 0 or 255.",
        "- Named `<patch_id>_expert.png` inside `results/expert_evaluation/submissions/`.",
        "- One file per blinded input; no extra markup, colour, or alpha channels.",
        "",
        f"Exported patch count: **{sum(counts.values())}** "
        f"(minimum {MIN_PER_CATEGORY} per category). See `annotation_export/manifest.csv` "
        "for identifiers (do not read its statistics before annotating).",
        "",
    ]
    (OUT_ROOT / "ANNOTATION_INSTRUCTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    global RAND_KEY
    for d in (IMAGES_DIR, MASKS_DIR, BLINDED_DIR, SUBMISSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/6] Scanning per-strip candidate pool (train.npy excluded by design)...")
    R = load_pool()
    n_total = len(R["pid"])
    rng = np.random.default_rng(SEED)
    RAND_KEY = rng.permutation(n_total)

    print("[2/6] Selecting candidates...")
    pool = list(range(n_total))
    taken: set[int] = set()
    selected: dict[str, list[int]] = {}
    for cat, fn in SELECTORS.items():
        sel = fn(R, pool, taken, CATEGORY_TARGETS[cat])
        selected[cat] = sel

    print("[3/6] Exporting images, pseudo-label masks, blinded copies...")
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    methods: dict[str, str] = {}
    arr_cache: dict[tuple[str, str], np.ndarray] = {}
    for cat, idxs in selected.items():
        for i in idxs:
            cls_i, strip_i = str(R["cls"][i]), str(R["strip"][i])
            key = (cls_i, strip_i)
            if key not in arr_cache:
                arr_cache[key] = np.load(
                    PATCHES_DIR / cls_i / f"{strip_i}.npy", mmap_mode="r"
                )
            patch = np.asarray(arr_cache[key][R["row"][i]], dtype=np.float32)
            mask, method = generate_mask(patch, cls_i, None)
            mask = clean_mask(mask)
            methods[str(R["pid"][i])] = method
            pid = str(R["pid"][i])
            save_png(patch, IMAGES_DIR / f"{pid}.png")
            save_png(mask, MASKS_DIR / f"{pid}_pseudo.png")
            shutil.copyfile(IMAGES_DIR / f"{pid}.png", BLINDED_DIR / f"{pid}.png")
            cache[i] = (patch, mask)

    print("[4/6] Writing manifest.csv and contact sheets...")
    manifest_path = EXPORT_DIR / "manifest.csv"
    header = ["patch_id", "source_strip", "source_class", "array_row_index", "category"] + STAT_COLS
    rows_out = []
    for cat, idxs in selected.items():
        for i in idxs:
            rows_out.append([
                str(R["pid"][i]), str(R["strip"][i]), str(R["cls"][i]), int(R["row"][i]), cat,
                *[fmt(float(R[c][i])) for c in STAT_COLS],
            ])
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)
    sha_manifest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    for cat, idxs in selected.items():
        contact_sheet(cat, idxs, R, cache)
    blinded_list = EXPORT_DIR / "blinded" / "blinded_list.txt"
    blinded_list.write_text("\n".join(str(R["pid"][i]) for _, idxs in selected.items() for i in idxs) + "\n", encoding="utf-8")

    print("[5/6] Writing metadata.json...")
    counts = {c: len(v) for c, v in selected.items()}
    strip_meta = {}
    if DATASET_META_PATH.exists():
        dm = json.loads(DATASET_META_PATH.read_text(encoding="utf-8"))
        strip_meta = {
            s: {k: dm.get("images", {}).get(s, {}).get(k) for k in ("product_id", "date", "crater", "lat_range", "classification")}
            for s in STRIPS
        }
    per_cell = {}
    for c in CLASSES:
        for s in STRIPS:
            per_cell[f"{c}/{s}"] = int(((R["cls"] == c) & (R["strip"] == s)).sum())
    meta = {
        "dataset_id": DATASET_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "purpose": (
            "Expert annotation export for independent evaluation of algorithmic "
            "pseudo-label reliability. Pseudo-labels are NOT ground truth; no human "
            "annotations existed at export time."
        ),
        "provenance_policy": "Only per-strip arrays used; train.npy/val.npy excluded (mixed-strip content).",
        "strip_products": strip_meta,
        "candidate_pool": {"total": int(n_total), "per_class_strip": per_cell},
        "selection": {
            "total_target": TOTAL_TARGET,
            "min_per_category": MIN_PER_CATEGORY,
            "tie_breaking": f"np.random.default_rng({SEED}) permutation key; stable sort on (-score, key)",
            "balance": "round-robin across strips (and strip x class for crater_rim_transition)",
            "categories": {
                c: {"target": CATEGORY_TARGETS[c], "selected": counts[c],
                    "criteria": CATEGORY_CRITERIA_TEXT[c]}
                for c in SELECTORS
            },
            "stat_definitions": {
                "edge_density": f"fraction(|scipy sobel gradient| > {EDGE_THRESH})",
                "dark_fraction": f"fraction(intensity < {DARK_THRESH})",
                "bright_fraction": f"fraction(intensity > {BRIGHT_THRESH})",
                "otsu_threshold": "lowest threshold_multiotsu(classes=3) value; NaN when it fails",
                "otsu_gap": "(t_otsu - p5) / (p95 - p5 + 1e-6); NaN when span < 1e-3",
                "local_min_proxy": "fraction(intensity < mean - 2*std)",
                "p99_minus_p95": "high-albedo speckle spike metric",
            },
        },
        "pseudo_label_methods_used": dict(sorted(((k, v) for k, v in methods.items()), key=lambda kv: kv[0])),
        "outputs": {
            "manifest": "annotation_export/manifest.csv",
            "inputs": "annotation_export/images/",
            "pseudo_masks": "annotation_export/masks/",
            "blinded_inputs": "annotation_export/blinded/",
            "contact_sheets": "annotation_export/contact_sheet_<category>.png",
            "instructions": "ANNOTATION_INSTRUCTIONS.md",
            "submissions_dir": "submissions/",
        },
        "blinding_note": "Experts annotate from annotation_export/blinded only; pseudo-labels must remain unseen during annotation.",
        "sha256_manifest_csv": sha_manifest,
    }
    (EXPORT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[6/6] Writing ANNOTATION_INSTRUCTIONS.md and verifying...")
    write_instructions(counts)
    (SUBMISSIONS_DIR / ".gitkeep").touch()

    print("\n================ SELECTION SUMMARY ================")
    print(f"{'category':<26}{'tgt':>5}{'sel':>5}{'SH01':>6}{'SH02':>6}{'CB01':>6}")
    for cat in SELECTORS:
        idxs = selected[cat]
        sc = {s: sum(1 for i in idxs if R["strip"][i] == s) for s in STRIPS}
        flag = "" if len(idxs) >= MIN_PER_CATEGORY else "  <-- SHORTFALL"
        print(f"{cat:<26}{CATEGORY_TARGETS[cat]:>5}{len(idxs):>5}"
              f"{sc['shackleton_01']:>6}{sc['shackleton_02']:>6}{sc['cabeus_01']:>6}{flag}")
    total_sel = sum(counts.values())
    print("-" * 54)
    print(f"{'TOTAL':<26}{TOTAL_TARGET:>5}{total_sel:>5}")
    if total_sel > TOTAL_TARGET:
        print("WARNING: total exceeds target cap of 80.")
    short = [c for c, v in counts.items() if v < MIN_PER_CATEGORY]
    if short:
        print(f"HONEST SHORTFALL REPORT: categories below minimum of {MIN_PER_CATEGORY}: {short}")
    else:
        print(f"All categories meet the minimum of {MIN_PER_CATEGORY} per category.")
    print(f"Done in {time.time() - t0:.1f}s. Export root: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
