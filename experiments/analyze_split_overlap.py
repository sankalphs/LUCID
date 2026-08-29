"""Split integrity analysis: detect exact-duplicate patch overlap between
train.npy / val.npy and per-strip arrays to verify the documented
strip-based split (train = shackleton_01 + cabeus_01, val = shackleton_02).

Outputs: results/split_integrity/split_overlap_report.json (+ .md)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PATCHES_DIR = PROJECT_ROOT / "dataset" / "kaggle_dataset" / "patches"
OUT_DIR = PROJECT_ROOT / "results" / "split_integrity"

CLASSES = ["psr", "sunlit", "mixed"]
FILES = ["train", "val", "cabeus_01", "shackleton_01", "shackleton_02"]


def hash_array(arr: np.ndarray) -> list[str]:
    # Hash raw bytes of each 64x64 float32 patch.
    return [
        hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
        for p in arr
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hashes: dict[tuple[str, str], list[str]] = {}
    counts: dict[str, dict[str, int]] = {}

    for cls in CLASSES:
        counts[cls] = {}
        for fname in FILES:
            f = PATCHES_DIR / cls / f"{fname}.npy"
            arr = np.load(f)
            hashes[(cls, fname)] = hash_array(arr)
            counts[cls][fname] = len(arr)

    report: dict = {"counts": counts, "pairwise_overlap": {}}

    def overlap(cls: str, a: str, b: str) -> int:
        sa = set(hashes[(cls, a)])
        sb = set(hashes[(cls, b)])
        return len(sa & sb)

    pairs = []
    for i, a in enumerate(FILES):
        for b in FILES[i + 1:]:
            pairs.append((a, b))

    for cls in CLASSES:
        for a, b in pairs:
            n = overlap(cls, a, b)
            report["pairwise_overlap"][f"{cls}:{a}|{b}"] = n

    # Key questions
    keys = {}
    for cls in CLASSES:
        n_train_in_sh02 = overlap(cls, "train", "shackleton_02")
        n_train_in_sh01 = overlap(cls, "train", "shackleton_01")
        n_train_in_cab = overlap(cls, "train", "cabeus_01")
        n_val_in_sh02 = overlap(cls, "val", "shackleton_02")
        n_train_in_val = overlap(cls, "train", "val")
        keys[cls] = {
            "train∩shackleton_01": n_train_in_sh01,
            "train∩cabeus_01": n_train_in_cab,
            "train∩shackleton_02": n_train_in_sh02,
            "val∩shackleton_02": n_val_in_sh02,
            "train∩val_duplicates": n_train_in_val,
            "train_count": counts[cls]["train"],
            "val_count": counts[cls]["val"],
            "sh01+cabeus": counts[cls]["shackleton_01"] + counts[cls]["cabeus_01"],
        }
    report["key_questions"] = keys

    with open(OUT_DIR / "split_overlap_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# Split Integrity Report", "",
             "| class | question | value |", "|---|---|---|"]
    for cls, d in keys.items():
        for k, v in d.items():
            lines.append(f"| {cls} | {k} | {v} |")
    lines += ["", "## Pairwise exact-duplicate overlaps", "",
              "| class | pair | shared unique patches |", "|---|---|---|"]
    for k, v in report["pairwise_overlap"].items():
        cls, pair = k.split(":")
        lines.append(f"| {cls} | {pair.replace('|', ' ∩ ')} | {v} |")
    (OUT_DIR / "split_overlap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(keys, indent=2))


if __name__ == "__main__":
    main()
