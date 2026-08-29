"""Backfill registry rows for classical methods whose metrics.json exists on
disk but which were never registered (pre-crash legacy run)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from framework import register_result, stamp, load_config          # noqa: E402

BASE = PROJECT_ROOT / "results" / "classical"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "legacy"


def main():
    reg = []
    rp = PROJECT_ROOT / "results" / "all_results.json"
    if rp.exists():
        reg = json.loads(rp.read_text(encoding="utf-8"))
    have = {r.get("exp_id") for r in reg}

    for mdir in sorted(BASE.iterdir()):
        mj = mdir / "metrics.json"
        if not mdir.is_dir() or not mj.exists():
            continue
        d = json.loads(mj.read_text(encoding="utf-8"))
        exp_id = f"classical/{d['method']}/{mdir.name}"
        if exp_id in have:
            continue
        register_result({
            "exp_id": exp_id, "timestamp": stamp(), "arch": "classical",
            "encoder": "-", "split_preset": mdir.name, "classes": "mixed",
            "seed": 42,
            "pseudo_label_method": "multiotsu3_lowest+closing_disk1",
            "augmentation": "none", "loss": "-", "optimizer": "-",
            "iou": round(d["iou_mean"], 4), "dice": round(d["dice_mean"], 4),
            "accuracy": round(d["pixel_accuracy_mean"], 4),
            "hd95": round(d["hd95_mean"], 4),
            "bf1": round(d["boundary_f1_mean"], 4),
        })
        print("registered", exp_id)


if __name__ == "__main__":
    main()
