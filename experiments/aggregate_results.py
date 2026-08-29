"""Aggregate all experiment outputs into publication tables/plots.

Idempotent; tolerates missing pieces (warns and skips). Primary metrics are
the paper-faithful protocol (val random-CLAHE); *_clean columns carry the
no-transform variant when available.

Outputs:
  results/multiseed_summary.csv          results/cross_region_summary.csv
  results/architecture_comparison.csv    results/ablation_summary.csv
  results/plots/multiseed_metrics.png    results/plots/classical_vs_neural.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
REG_JSON = RESULTS / "all_results.json"

METRICS = ["iou", "dice", "accuracy", "hd95", "bf1"]


def load_registry() -> list[dict]:
    if REG_JSON.exists():
        return json.loads(REG_JSON.read_text(encoding="utf-8"))
    return []


def rows_by_prefix(rows, prefix):
    return [r for r in rows if str(r.get("exp_id", "")).startswith(prefix)]


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def multiseed(rows):
    runs = rows_by_prefix(rows, "multiseed_seed")
    repro = [r for r in rows if r.get("exp_id") == "repro_seed42"]
    runs = sorted(runs + repro, key=lambda r: int(r.get("seed", 0)))
    if not runs:
        print("[aggregate] no multiseed rows yet"); return
    import numpy as np
    out_rows = []
    for m in METRICS:
        vals = np.array([float(r[m]) for r in runs if r.get(m) is not None])
        key = f"clean_{m}" if all(r.get(f"clean_{m}") is not None for r in runs) else None
        cvals = (np.array([float(r[f"clean_{m}"]) for r in runs])
                 if key else None)
        out_rows.append({
            "metric": m,
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0,
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "n_seeds": len(vals),
            "clean_mean": round(float(cvals.mean()), 4) if cvals is not None else "",
            "clean_std": round(float(cvals.std(ddof=1)), 4)
            if cvals is not None and len(cvals) > 1 else "",
        })
    write_csv(RESULTS / "multiseed_summary.csv",
              ["metric", "mean", "std", "min", "max", "n_seeds",
               "clean_mean", "clean_std"], out_rows)

    with open(RESULTS / "multiseed_per_seed.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exp_id", "seed", *METRICS,
                                          "best_epoch", "final_epoch",
                                          "train_minutes"],
                           extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({**r, "seed": r.get("seed"),
                        **{m: r.get(m) for m in METRICS}})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        labels = [str(r.get("seed")) for r in runs]
        for ax, m in zip(axes, METRICS):
            vals = [float(r[m]) for r in runs]
            ax.bar(labels, vals, color="#4477aa")
            mean = sum(vals) / len(vals)
            ax.axhline(mean, ls="--", c="crimson",
                       label=f"mean {mean:.4f}")
            ax.set_title(m.upper()); ax.legend(fontsize=8)
            ax.set_xlabel("seed")
        fig.suptitle("U-Net/ResNet18 across seeds (paper-faithful eval)")
        fig.tight_layout()
        fig.savefig(RESULTS / "plots" / "multiseed_metrics.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[aggregate] multiseed plot skipped: {e}")
    print(f"[aggregate] multiseed: {len(runs)} seed runs")


CLASSICAL_ORDER = [
    ("Global threshold 0.10", "Global threshold"),
    ("Otsu (per-patch)", "Otsu"),
    ("Multi-Otsu", "Multi-Otsu"),
    ("Multi-Otsu + morphology", "Multi-Otsu + morphology"),
    ("Adaptive threshold", "Adaptive threshold"),
    ("Random Forest", "Random Forest"),
]


def classical(rows):
    tag = "legacy"
    cls_rows = [r for r in rows if str(r.get("split_preset")) == tag
                and str(r.get("arch")) == "classical"]
    unet = [r for r in rows if r.get("exp_id") == "repro_seed42"]
    if not cls_rows:
        print("[aggregate] no legacy classical rows yet")
        cls_map = {}
    else:
        cls_map = {r["exp_id"].split("/")[1]: r for r in cls_rows}
    table = []
    for key, pretty in CLASSICAL_ORDER:
        r = cls_map.get(key)
        if r:
            table.append({"Method": pretty, "IoU": r.get("iou"),
                          "Dice": r.get("dice"), "Accuracy": r.get("accuracy"),
                          "HD95": r.get("hd95"), "BF1": r.get("bf1")})
        else:
            table.append({"Method": pretty})
    if unet:
        r = unet[0]
        table.append({"Method": "U-Net (ResNet18)", "IoU": r.get("iou"),
                      "Dice": r.get("dice"), "Accuracy": r.get("accuracy"),
                      "HD95": r.get("hd95"), "BF1": r.get("bf1")})
    write_csv(RESULTS / "classical_comparison_legacy.csv",
              ["Method", "IoU", "Dice", "Accuracy", "HD95", "BF1"], table)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have = [t for t in table if t.get("IoU") is not None]
        names = [t["Method"] for t in have]
        x = range(len(have))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, m, title in ((axes[0], "IoU", "IoU"),
                             (axes[1], "Dice", "Dice")):
            vals = [float(t[m]) for t in have]
            bars = ax.bar(x, vals,
                          color=["#77aadd"] * (len(have) - 1) + ["#cc4444"])
            ax.set_xticks(list(x))
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.set_title(title)
            ax.bar_label(bars, fmt="%.4f", fontsize=8)
            lo = min(vals) - 0.05
            ax.set_ylim(max(0.0, lo), 1.0)
        fig.suptitle("Classical baselines vs U-Net "
                     "(agreement with LUCID pseudo-labels, legacy val split)")
        fig.tight_layout()
        fig.savefig(RESULTS / "plots" / "classical_vs_neural.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[aggregate] classical plot skipped: {e}")


REGION_STRINGS = {
    "legacy": ("shackleton_01+cabeus_01+shackleton_02(contam)",
               "shackleton_02(val arrays)"),
    "A_strip": ("shackleton_01+cabeus_01", "shackleton_02"),
    "B": ("shackleton_01+shackleton_02", "cabeus_01"),
    "C": ("cabeus_01", "shackleton_01+shackleton_02"),
}


def cross_region(rows):
    out = []
    neural = [r for r in rows if str(r.get("arch")).startswith(("unet", "deeplab"))
              and str(r.get("split_preset")) in REGION_STRINGS
              and r.get("exp_id") != "repro_seed42"]
    neural += [r for r in rows if r.get("exp_id") == "repro_seed42"]
    for r in neural:
        tr, te = REGION_STRINGS[str(r.get("split_preset"))]
        out.append({"Train regions": tr, "Test region": te,
                    "Method": f"U-Net ({r.get('arch')})",
                    "seed": r.get("seed"), "IoU": r.get("iou"),
                    "Dice": r.get("dice"), "Accuracy": r.get("accuracy"),
                    "HD95": r.get("hd95"), "BF1": r.get("bf1")})
    for tag in ("legacy", "A_strip", "B", "C"):
        cls_rows = [r for r in rows if str(r.get("split_preset")) == tag
                    and str(r.get("arch")) == "classical"
                    and "morphology" in str(r.get("exp_id", ""))
                    and tag in str(r.get("exp_id", ""))]
        for r in cls_rows:
            tr, te = REGION_STRINGS[tag]
            out.append({"Train regions": tr, "Test region": te,
                        "Method": "Multi-Otsu+morphology (classical)",
                        "seed": "-", "IoU": r.get("iou"),
                        "Dice": r.get("dice"), "Accuracy": r.get("accuracy"),
                        "HD95": r.get("hd95"), "BF1": r.get("bf1")})
    if out:
        write_csv(RESULTS / "cross_region_summary.csv",
                  ["Train regions", "Test region", "Method", "seed",
                   "IoU", "Dice", "Accuracy", "HD95", "BF1"], out)
        print(f"[aggregate] cross-region rows: {len(out)}")
    else:
        print("[aggregate] no cross-region rows yet")


def architectures(rows):
    out = []
    repro = [r for r in rows if r.get("exp_id") == "repro_seed42"]
    for r in repro:
        out.append({"Architecture": "U-Net/ResNet18 (mixed-only, this study)",
                    "seed": r.get("seed"), "IoU": r.get("iou"),
                    "Dice": r.get("dice"), "Accuracy": r.get("accuracy"),
                    "HD95": r.get("hd95"), "BF1": r.get("bf1"),
                    "best_epoch": r.get("best_epoch"),
                    "note": "identical protocol"})
    fair = [r for r in rows if str(r.get("exp_id", "")).startswith("arch_")]
    for r in fair:
        out.append({"Architecture": f"{r.get('arch')}/ResNet18 (mixed-only)",
                    "seed": r.get("seed"), "IoU": r.get("iou"),
                    "Dice": r.get("dice"), "Accuracy": r.get("accuracy"),
                    "HD95": r.get("hd95"), "BF1": r.get("bf1"),
                    "best_epoch": r.get("best_epoch"),
                    "note": "identical protocol"})
    # pre-existing combined-data architecture baselines (NOT same training set
    # as headline U-Net: trained on psr+sunlit+mixed combined -> labelled)
    bt = PROJECT_ROOT / "baselines" / "comparison_table.json"
    if bt.exists():
        try:
            data = json.loads(bt.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("models", [])
            for d in items:
                name = d.get("model", "?")
                if "LUCID" in str(name):
                    note = "published reference (training-data scope differs)"
                else:
                    note = ("PREEXISTING run: trained on COMBINED psr+sunlit+"
                            "mixed data - NOT directly comparable to mixed-only"
                            " U-Net; documented in EXPERIMENT_AUDIT.md 4.2")
                out.append({"Architecture": name, "seed": 42,
                            "IoU": round(d.get("iou", float("nan")), 4),
                            "Dice": round(d.get("dice", float("nan")), 4),
                            "Accuracy": round(d.get("accuracy", float("nan")), 4),
                            "HD95": round(d.get("hd95", float("nan")), 4),
                            "BF1": round(d.get("bf1", float("nan")), 4),
                            "best_epoch": d.get("best_epoch"), "note": note})
        except Exception as e:
            print(f"[aggregate] could not parse comparison_table.json: {e}")
    if out:
        write_csv(RESULTS / "architecture_comparison.csv",
                  ["Architecture", "seed", "IoU", "Dice", "Accuracy", "HD95",
                   "BF1", "best_epoch", "note"], out)
        print(f"[aggregate] architecture rows: {len(out)}")


def ablations(rows):
    full = [r for r in rows if r.get("exp_id") == "repro_seed42"]
    abl = rows_by_prefix(rows, "ablation_")
    if not abl and not full:
        print("[aggregate] no ablation rows yet"); return
    out = [{"Variant": "Full LUCID (reference)", **{m: r.get(m) for m in METRICS}}
           for r in full]
    names = {
        "ablation_no_clahe": "No CLAHE augmentation",
        "ablation_no_augmentation": "No augmentation",
        "ablation_bce_only": "BCE only (dice_weight=0)",
        "ablation_dice_only": "Dice only (bce_weight=0)",
        "ablation_posweight1": "Positive weight = 1",
        "ablation_posweight3": "Positive weight = 3 (= full)",
        "ablation_no_morphology": "No morphology in pseudo-labels",
    }
    for r in sorted(abl, key=lambda r: r.get("exp_id", "")):
        out.append({"Variant": names.get(r.get("exp_id"), r.get("exp_id")),
                    **{m: r.get(m) for m in METRICS}})
    write_csv(RESULTS / "ablation_summary.csv", ["Variant", *METRICS], out)
    print(f"[aggregate] ablation rows: {len(out)}")


def main():
    rows = load_registry()
    print(f"[aggregate] registry rows: {len(rows)}")
    multiseed(rows)
    classical(rows)
    cross_region(rows)
    architectures(rows)
    ablations(rows)


if __name__ == "__main__":
    main()
