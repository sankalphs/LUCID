"""
Ablation runner for PSR Shadow Boundary Segmentation.

Runs each experiment from scratch under a controlled change, evaluating the
same five metrics reported by the full model. Supports:

  1. No augmentation
  2. No CLAHE
  3. Unweighted loss (pos_weight = 1.0)
  4. Fallback threshold sweep (0.03, 0.04, 0.0484, 0.06)

All runs share the same seed, optimizer, scheduler, and early-stopping
configuration. Each variant writes its raw 5-metric JSON to
``ablation_results/<variant>.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentations import (
    get_train_transforms_no_aug,
    get_train_transforms_no_clahe,
    get_train_transforms,
    get_val_transforms,
)
from src.data.dataset import CombinedPSRDataset, PSRDataset
from src.evaluate import Evaluator, compute_all_metrics
from src.models.losses import WeightedBCEDiceLoss
from src.train import (
    EarlyStopping,
    MetricTracker,
    build_model,
    set_seed,
    train_one_epoch,
    validate,
)


ABLATION_RESULTS_DIR = PROJECT_ROOT / "ablation_results"


def _subsample_train_indices(
    train_loader: torch.utils.data.DataLoader,
    fraction: float,
    seed: int,
) -> tuple[list[int], list[str]]:
    """Return a stratified subsample of train indices and the originating class label.

    Walks the underlying combined dataset and draws ``fraction`` of each
    class's indices using ``seed`` for reproducibility. The val set is
    intentionally NOT subsampled - all ablations evaluate on the same
    ``shackleton_02`` patches.
    """
    rng = np.random.default_rng(seed)
    combined = train_loader.dataset
    class_names = ["psr", "sunlit", "mixed"]
    selected: list[int] = []
    origins: list[str] = []

    running_offset = 0
    for cls_name, sub_ds in zip(class_names, combined.datasets):
        n = len(sub_ds)
        if n == 0:
            continue
        take = max(1, int(round(n * fraction)))
        local_idx = rng.choice(n, size=take, replace=False)
        global_idx = running_offset + local_idx
        selected.extend(int(i) for i in global_idx)
        origins.extend([cls_name] * take)
        running_offset += n

    rng.shuffle(selected)
    return selected, origins


def _flatten_aggregate(agg: dict) -> dict:
    """Flatten evaluator aggregate (mean/std/median/...) to a single mean per metric."""
    return {
        "iou": float(agg.get("iou", {}).get("mean", 0.0)),
        "dice": float(agg.get("dice", {}).get("mean", 0.0)),
        "accuracy": float(agg.get("pixel_accuracy", {}).get("mean", 0.0)),
        "hd95": float(agg.get("hd95", {}).get("mean", 0.0)),
        "bf1": float(agg.get("boundary_f1", {}).get("mean", 0.0)),
    }


def _evaluate(model: torch.nn.Module, val_loader, device: torch.device) -> dict:
    """Run model on val set and compute the 5-metric aggregate."""
    model.eval()
    evaluator = Evaluator()
    all_patches, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for patches, masks in val_loader:
            patches = patches.to(device)
            probs = torch.sigmoid(model(patches))
            evaluator.update(probs.cpu().numpy(), masks.numpy())
            all_patches.append(patches.cpu().numpy().squeeze(1))
            all_targets.append(masks.numpy().squeeze(1))
            all_probs.append(probs.cpu().numpy().squeeze(1))
    aggregate = evaluator.compute_aggregate()
    flat = _flatten_aggregate(aggregate)

    confusion = _compute_confusion(all_probs, all_targets)
    return flat, confusion, evaluator


def _compute_confusion(all_probs, all_targets):
    """Return a confusion summary used for failure analysis."""
    probs = np.concatenate(all_probs).reshape(-1)
    targets = np.concatenate(all_targets).reshape(-1)
    pred = (probs > 0.5).astype(np.uint8)
    target = (targets > 0.5).astype(np.uint8)
    tn = int(((pred == 0) & (target == 0)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())  # shadow→light: predicted light, was shadow
    fn = int(((pred == 0) & (target == 1)).sum())  # light→shadow: predicted shadow, was light
    tp = int(((pred == 1) & (target == 1)).sum())
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "shadow_to_light": fp,
        "light_to_shadow": fn,
    }


def _build_dataloaders_for_variant(
    config: dict,
    variant_name: str,
    fallback_threshold: float | None,
    train_subset_fraction: float | None = None,
    seed: int = 42,
):
    """Build train and val dataloaders for an ablation variant.

    If ``train_subset_fraction`` is provided (0, 1], the training set is
    reduced to that stratified fraction of each class so that wall-clock
    training time stays tractable on CPU. The val set is left intact.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    aug_cfg = config.get("augmentation", {})

    patches_dir = str(Path(data_cfg["base_dir"]) / "patches")
    train_classes = ["psr", "sunlit", "mixed"]

    if variant_name == "no_augmentation":
        train_transform = get_train_transforms_no_aug()
    elif variant_name == "no_clahe":
        train_transform = get_train_transforms_no_clahe(aug_cfg)
    else:
        train_transform = get_train_transforms() if aug_cfg else None

    val_transform = get_val_transforms()

    train_ds = CombinedPSRDataset(
        patches_dir, split="train",
        classes=train_classes, transform=train_transform,
        fallback_threshold=fallback_threshold,
    )
    val_ds = PSRDataset(
        patches_dir, class_label="mixed", split="val",
        transform=val_transform,
        fallback_threshold=fallback_threshold,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False,
    )

    if train_subset_fraction is not None and train_subset_fraction < 1.0:
        indices, origins = _subsample_train_indices(
            train_loader, train_subset_fraction, seed
        )
        subset_ds = torch.utils.data.Subset(train_ds, indices)
        train_loader = torch.utils.data.DataLoader(
            subset_ds, batch_size=train_cfg["batch_size"],
            shuffle=True, num_workers=0, pin_memory=False,
        )
        unique_classes, class_counts = np.unique(origins, return_counts=True)
        per_class = {str(c): int(n) for c, n in zip(unique_classes, class_counts)}
        print(f"  Stratified train subset: {len(indices)} patches "
              f"({per_class}) [val unchanged: {len(val_ds)}]")

    return train_loader, val_loader


def _save_variant_result(variant: str, metrics: dict, confusion: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": variant,
        "iou": metrics["iou"],
        "dice": metrics["dice"],
        "accuracy": metrics["accuracy"],
        "hd95": metrics["hd95"],
        "bf1": metrics["bf1"],
        "_confusion": confusion,
    }
    path = out_dir / f"{variant}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> wrote {path}")
    return path


def run_variant(
    variant_name: str,
    config: dict,
    device: torch.device,
    pos_weight_override: float | None = None,
    fallback_threshold: float | None = None,
    out_dir: Path = ABLATION_RESULTS_DIR,
    log_to_tensorboard: bool = True,
    train_subset_fraction: float | None = None,
) -> dict:
    """Train a single ablation variant from scratch and return the 5 metrics."""
    print("\n" + "=" * 70)
    print(f"ABLATION VARIANT: {variant_name}")
    if fallback_threshold is not None:
        print(f"  fallback_threshold={fallback_threshold}")
    if pos_weight_override is not None:
        print(f"  pos_weight={pos_weight_override}")
    print("=" * 70)

    cfg = copy.deepcopy(config)
    set_seed(cfg["project"]["seed"])

    if pos_weight_override is not None:
        cfg["loss"]["pos_weight"] = pos_weight_override

    train_loader, val_loader = _build_dataloaders_for_variant(
        cfg, variant_name, fallback_threshold,
        train_subset_fraction=train_subset_fraction,
        seed=cfg["project"]["seed"],
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(cfg, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = WeightedBCEDiceLoss(
        pos_weight=cfg["loss"]["pos_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=cfg["training"]["scheduler"]["factor"],
        patience=cfg["training"]["scheduler"]["patience"],
        min_lr=cfg["training"]["scheduler"]["min_lr"],
    )
    early_stopping = EarlyStopping(
        patience=cfg["training"]["early_stopping"]["patience"],
        metric=cfg["training"]["early_stopping"]["metric"],
    )

    n_epochs = cfg["training"]["epochs"]
    best_iou = 0.0
    best_state = None
    best_epoch = 0

    writer = None
    if log_to_tensorboard:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(PROJECT_ROOT) / cfg["outputs"]["log_dir"] / f"ablation_{variant_name}_{ts}"
        writer = SummaryWriter(log_dir)

    print(f"\nTraining for up to {n_epochs} epochs...")
    t_start = time.time()
    for epoch in range(n_epochs):
        epoch_t = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["iou"])

        if writer is not None:
            writer.add_scalars("loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
            writer.add_scalars("iou", {"train": train_metrics["iou"], "val": val_metrics["iou"]}, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train L: {train_metrics['loss']:.4f} IoU: {train_metrics['iou']:.4f} | "
                f"Val L: {val_metrics['loss']:.4f} IoU: {val_metrics['iou']:.4f} "
                f"Dice: {val_metrics['dice']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} "
                f"| {time.time()-epoch_t:.1f}s"
            )

        if early_stopping(val_metrics["iou"]):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t_start
    print(f"Training finished in {train_time/60:.1f} min, best IoU={best_iou:.4f} at epoch {best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, confusion, evaluator = _evaluate(model, val_loader, device)
    print(f"Final metrics: {metrics}")

    if writer is not None:
        writer.close()

    _save_variant_result(variant_name, metrics, confusion, out_dir)
    return {
        "variant": variant_name,
        "iou": metrics["iou"],
        "dice": metrics["dice"],
        "accuracy": metrics["accuracy"],
        "hd95": metrics["hd95"],
        "bf1": metrics["bf1"],
        "confusion": confusion,
        "best_epoch": best_epoch,
        "train_minutes": train_time / 60.0,
    }


THRESHOLD_VARIANTS = [
    ("fallback_threshold_0.0300", 0.03),
    ("fallback_threshold_0.0400", 0.04),
    ("fallback_threshold_0.0484", 0.0484),
    ("fallback_threshold_0.0600", 0.06),
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PSR ablation experiments.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only a single variant by name (no_augmentation|no_clahe|unweighted_loss|fallback_threshold_*)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging to speed up training.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override config training.epochs (useful for fast sanity runs).",
    )
    parser.add_argument(
        "--train-subset",
        type=float,
        default=None,
        help="Fraction of training set to use, stratified by class. 1.0 = full set.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Override early stopping patience (useful with --max-epochs).",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_epochs is not None:
        config["training"]["epochs"] = args.max_epochs
    if args.early_stopping_patience is not None:
        config["training"]["early_stopping"]["patience"] = args.early_stopping_patience

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, threads: {torch.get_num_threads()}")
    print(f"Seed: {config['project']['seed']}")
    print(f"Max epochs: {config['training']['epochs']}, "
          f"early stop patience: {config['training']['early_stopping']['patience']}")
    if args.train_subset is not None:
        print(f"Train subset fraction: {args.train_subset}")
    print(f"Output dir: {ABLATION_RESULTS_DIR}")

    ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    single_variants = [
        ("no_augmentation", {"pos_weight_override": None, "fallback_threshold": None}),
        ("no_clahe", {"pos_weight_override": None, "fallback_threshold": None}),
        ("unweighted_loss", {"pos_weight_override": 1.0, "fallback_threshold": None}),
    ]
    threshold_variants = [
        (name, {"pos_weight_override": None, "fallback_threshold": t})
        for name, t in THRESHOLD_VARIANTS
    ]

    all_runs: list[dict] = []
    planned = single_variants + threshold_variants
    if args.only:
        planned = [
            r for r in planned
            if r[0] == args.only or r[0].startswith(args.only)
        ]
        if not planned:
            raise SystemExit(f"No variant matched --only={args.only}")

    overall_start = time.time()
    for variant_name, kwargs in planned:
        try:
            res = run_variant(
                variant_name,
                config,
                device,
                pos_weight_override=kwargs["pos_weight_override"],
                fallback_threshold=kwargs["fallback_threshold"],
                log_to_tensorboard=not args.no_tensorboard,
                train_subset_fraction=args.train_subset,
            )
            all_runs.append(res)
        except Exception as exc:
            print(f"!! Variant {variant_name} failed: {exc}")
            import traceback
            traceback.print_exc()
            continue
    total_minutes = (time.time() - overall_start) / 60.0
    print(f"\nAll ablations finished in {total_minutes:.1f} min")

    summary_path = ABLATION_RESULTS_DIR / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()