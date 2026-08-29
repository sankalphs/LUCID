"""Pre-flight checks for all queued experiment configurations.
Builds nothing heavy; verifies code paths only (plus small dataset probes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from framework import ARCHITECTURES, SPLIT_PRESETS, load_config   # noqa: E402
from train_experiment import build_datasets                       # noqa: E402
from src.models.losses import WeightedBCEDiceLoss                 # noqa: E402


def ns(**kw):
    base = dict(exp_id="preflight", seed=42, arch="unet",
                split_preset="legacy", classes="mixed", pos_weight=3.0,
                dice_weight=0.5, no_aug=False, no_clahe=False,
                no_morphology=False, fallback_threshold=None,
                max_epochs=None, patience=15, batch_size=None,
                es_val_frac=0.1, limit_train=None, out_root=None,
                eval_only=None, no_registry=True)
    base.update(kw)
    return argparse.Namespace(**base)


def main():
    cfg = load_config()

    # 1. architecture constructors
    for name, cls in ARCHITECTURES.items():
        m = cls(encoder_name="resnet18", encoder_weights=None,
                in_channels=1, classes=1, activation=None)
        n = sum(p.numel() for p in m.parameters())
        print(f"arch {name}: OK ({n:,} params)")
        del m

    # 2. loss variants used by ablations
    for pw, dw in ((3.0, 0.5), (1.0, 0.5), (3.0, 0.0), (3.0, 1.0)):
        WeightedBCEDiceLoss(pos_weight=pw, dice_weight=dw)
    print("loss variants: OK")

    # 3. transform variants
    from framework import build_transforms
    t1, _ = build_transforms(False, True)     # no-clahe pipeline
    names = [type(t).__name__ for t in t1.transforms]
    assert not any("Clahe" in n for n in names), "CLAHE leaked into no-clahe"
    print("no-clahe transforms:", names)
    t2, _ = build_transforms(True, False)
    assert t2 is None
    print("no-aug transforms: OK (None)")

    # 4. strip-split dataset construction (small probe: limit via classes trick
    #    is not available -> full build but single-threaded, acceptable)
    for preset in ("A_strip", "B", "C"):
        args = ns(split_preset=preset)
        tr, es, te, cte, sizes = build_datasets(args, cfg)
        print(f"split {preset}: {sizes} | es subset of train: "
              f"{isinstance(es, __import__('torch.utils.data', fromlist=['Subset']).Subset)}")
        del tr, es, te, cte

    print("ALL PRE-FLIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
