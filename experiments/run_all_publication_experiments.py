"""Master runner for every publication experiment.

Maps experiment IDs to exact argv vectors (relative to the repo root) and
executes them SEQUENTIALLY with sys.executable from the repo root.

WARNING: running the FULL suite launches ~20 full training runs; at CPU-only
speeds (no CUDA/NPU on this machine) this costs MULTIPLE CPU-DAYS. Use
`--list` first and `--only id[,id...]` to select individual experiments.

Usage:
  python experiments/run_all_publication_experiments.py --list
  python experiments/run_all_publication_experiments.py                 # ALL (days!)
  python experiments/run_all_publication_experiments.py --only classical_legacy,reproduction
  python experiments/run_all_publication_experiments.py --only ablations --keep-going
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HOURS_EACH = "HOURS-each on CPU"
MINUTES = "~minutes"

TRAIN = "experiments/train_experiment.py"


def train_argv(*extra: str, exp_id: str, seed: int = 42) -> list[str]:
    return [TRAIN, "--exp-id", exp_id, "--seed", str(seed), *extra]


# experiment ID -> (cost estimate, [argv vector, ...] run sequentially)
EXPERIMENTS: dict[str, tuple[str, list[list[str]]]] = {
    "reproduction": (
        MINUTES,
        [
            [TRAIN, "--exp-id", "repro_seed42", "--seed", "42"],
        ],
    ),
    "classical_legacy": (
        MINUTES,
        [
            ["experiments/run_classical_baselines.py", "--split-preset", "legacy"],
        ],
    ),
    "classical_A_strip": (
        MINUTES,
        [
            ["experiments/run_classical_baselines.py", "--split-preset", "A_strip"],
        ],
    ),
    "classical_B": (
        MINUTES,
        [
            ["experiments/run_classical_baselines.py", "--split-preset", "B"],
        ],
    ),
    "classical_C": (
        MINUTES,
        [
            ["experiments/run_classical_baselines.py", "--split-preset", "C"],
        ],
    ),
    "multiseed": (
        HOURS_EACH,
        [
            train_argv(exp_id=f"multiseed_seed{s}", seed=s)
            for s in [123, 456, 789, 2026]  # seed 42 == reproduction
        ],
    ),
    "crossregion": (
        HOURS_EACH,
        [
            train_argv("--split-preset", "B", exp_id="crossB_seed42"),
            train_argv("--split-preset", "C", exp_id="crossC_seed42"),
        ],
    ),
    "ablations": (
        HOURS_EACH,
        [
            train_argv("--no-clahe", exp_id="ablation_no_clahe"),
            train_argv("--no-aug", exp_id="ablation_no_augmentation"),
            train_argv("--dice-weight", "0.0", exp_id="ablation_bce_only"),
            train_argv("--dice-weight", "1.0", exp_id="ablation_dice_only"),
            train_argv("--pos-weight", "1.0", exp_id="ablation_posweight1"),
            train_argv("--no-morphology", exp_id="ablation_no_morphology"),
        ],
    ),
    "architectures": (
        HOURS_EACH,
        [
            train_argv("--arch", "deeplabv3plus", exp_id="arch_deeplabv3plus_seed42"),
            train_argv("--arch", "unetplusplus", exp_id="arch_unetplusplus_seed42"),
        ],
    ),
    "fallback": (
        MINUTES,
        [
            ["experiments/fallback_sensitivity.py"],
        ],
    ),
    "pseudolabel_quality": (
        MINUTES,
        [
            ["experiments/analyze_pseudo_label_quality.py"],
        ],
    ),
    "expert_export": (
        MINUTES,
        [
            ["experiments/build_annotation_export.py"],
        ],
    ),
    "aggregate": (
        MINUTES,
        [
            ["experiments/aggregate_results.py"],
        ],
    ),
}

EXPERIMENT_ORDER = list(EXPERIMENTS)


def parse_only(raw: str) -> list[str]:
    ids = [tok.strip() for tok in raw.split(",") if tok.strip()]
    unknown = [tok for tok in ids if tok not in EXPERIMENTS]
    if unknown:
        raise SystemExit(
            f"Unknown experiment id(s): {', '.join(unknown)}. "
            f"Valid ids: {', '.join(EXPERIMENT_ORDER)}"
        )
    return ids


def print_list() -> None:
    print(f"{'ID':<22} {'COST':<18} RUNS  DETAIL")
    for exp_id in EXPERIMENT_ORDER:
        cost, argvs = EXPERIMENTS[exp_id]
        detail = "; ".join(" ".join(a) for a in argvs)
        print(f"{exp_id:<22} {cost:<18} {len(argvs):<5} {detail}")
    total = sum(len(argvs) for _, argvs in EXPERIMENTS.values())
    print(f"\nTotal runs in full suite: {total} "
          f"(training-heavy groups are HOURS-each on CPU; full suite = multiple CPU-days)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true",
                    help="print experiment IDs + estimated cost, then exit")
    ap.add_argument("--only", default=None, metavar="id[,id...]",
                    help="comma-separated subset of experiment IDs to run (default: ALL)")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue with remaining experiments after a failure")
    args = ap.parse_args()

    if args.list:
        print_list()
        return 0

    selected = EXPERIMENT_ORDER if args.only is None else parse_only(args.only)

    failures: list[tuple[str, int]] = []
    for exp_id in selected:
        _, argvs = EXPERIMENTS[exp_id]
        for argv in argvs:
            cmd = [sys.executable, *argv]
            print(f"[START {exp_id}] {' '.join(argv)}", flush=True)
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            rc = proc.returncode
            print(f"[DONE {exp_id} rc={rc}] {' '.join(argv)}", flush=True)
            if rc != 0:
                failures.append((exp_id, rc))
                if not args.keep_going:
                    print(f"[ABORT] stopping on first failure "
                          f"(use --keep-going to continue)", flush=True)
                    return 1

    if failures:
        print(f"[SUMMARY] {len(failures)} failed run(s): "
              f"{', '.join(f'{i}(rc={r})' for i, r in failures)}", flush=True)
        return 1
    print(f"[SUMMARY] all requested experiments finished OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
