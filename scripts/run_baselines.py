"""
Baseline comparison runner for PSR Shadow Boundary Segmentation.

Trains U-Net++ and DeepLabV3+ (both with ResNet18 encoder) under IDENTICAL
conditions to the existing U-Net/ResNet18 run reported in
``outputs/results.json``. No hyperparameter changes, no pretraining, same
seed (42), same Shackleton-01+Cabeus-01 -> Shackleton-02 split, same
Multi-Otsu pseudo-labels with fallback threshold 0.0484.

Architecture contract (per user spec):
  smp.UnetPlusPlus(   encoder_name="resnet18", encoder_weights=None,
                      in_channels=1, classes=1, activation=None )
  smp.DeepLabV3Plus(  encoder_name="resnet18", encoder_weights=None,
                      in_channels=1, classes=1, activation=None )

Outputs:
  baselines/unetplusplus_resnet18.json
  baselines/deeplabv3plus_resnet18.json
  baselines/comparison_table.json
  baselines/status_<arch>.json     (live progress; updated every batch)

Usage:
  python run_baselines.py --arch unetplusplus
  python run_baselines.py --arch deeplabv3plus
  python run_baselines.py --arch all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import segmentation_models_pytorch as smp
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.dataset import CombinedPSRDataset, PSRDataset
from src.evaluate import Evaluator
from src.models.losses import WeightedBCEDiceLoss
from src.train import EarlyStopping, MetricTracker, set_seed, validate

BASELINES_DIR = PROJECT_ROOT / "baselines"


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_architecture(arch: str, in_channels: int = 1, classes: int = 1) -> torch.nn.Module:
    """Instantiate the requested SMP architecture with the locked config."""
    common = dict(
        encoder_name="resnet18",
        encoder_weights=None,        # NO ImageNet pretraining
        in_channels=in_channels,     # grayscale: 1 channel
        classes=classes,
        activation=None,             # raw logits; loss applies sigmoid
    )
    if arch == "unetplusplus":
        return smp.UnetPlusPlus(**common)
    if arch == "deeplabv3plus":
        # ResNet18 + 64x64 input works on smp 0.5 without output_stride override
        # (verified manually before kicking off this script).
        return smp.DeepLabV3Plus(**common)
    raise ValueError(f"Unknown architecture: {arch}")


# ---------------------------------------------------------------------------
# Dataloaders (identical to run_ablations.py to match U-Net run conditions)
# ---------------------------------------------------------------------------

def build_dataloaders(config: dict):
    data_cfg = config["data"]
    train_cfg = config["training"]
    patches_dir = str(Path(data_cfg["base_dir"]) / "patches")

    train_classes = ["psr", "sunlit", "mixed"]
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    train_ds = CombinedPSRDataset(
        patches_dir, split="train",
        classes=train_classes, transform=train_transform,
    )
    val_ds = PSRDataset(
        patches_dir, class_label="mixed", split="val",
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _flatten_aggregate(agg: dict) -> dict:
    return {
        "iou": float(agg.get("iou", {}).get("mean", 0.0)),
        "dice": float(agg.get("dice", {}).get("mean", 0.0)),
        "accuracy": float(agg.get("pixel_accuracy", {}).get("mean", 0.0)),
        "hd95": float(agg.get("hd95", {}).get("mean", 0.0)),
        "bf1": float(agg.get("boundary_f1", {}).get("mean", 0.0)),
    }


def evaluate_model(model: torch.nn.Module, val_loader, device: torch.device,
                    desc: str = "eval") -> dict:
    model.eval()
    evaluator = Evaluator()
    with torch.no_grad():
        for patches, masks in tqdm(val_loader, desc=f"  {desc}", ncols=100, leave=False):
            patches = patches.to(device)
            probs = torch.sigmoid(model(patches))
            evaluator.update(probs.cpu().numpy(), masks.numpy())
    return _flatten_aggregate(evaluator.compute_aggregate())


def _write_status(status_path: Path, payload: dict) -> None:
    """Atomically write a JSON status file for external monitoring."""
    try:
        tmp = status_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        tmp.replace(status_path)
    except Exception:
        pass


def evaluate_from_checkpoint(arch: str, ckpt_path: Path, config: dict,
                             device: torch.device, out_dir: Path) -> dict:
    """Re-evaluate a saved best checkpoint and write the per-model JSON."""
    print(f"\nRe-evaluating {arch} from {ckpt_path} ...")
    _, val_loader = build_dataloaders(config)
    model = build_architecture(arch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    metrics = evaluate_model(model, val_loader, device)
    payload = {
        "model": arch,
        "encoder": "resnet18",
        "iou": metrics["iou"],
        "dice": metrics["dice"],
        "accuracy": metrics["accuracy"],
        "hd95": metrics["hd95"],
        "bf1": metrics["bf1"],
        "best_epoch": int(ckpt.get("epoch", 0)),
        "final_epoch": int(ckpt.get("epoch", 0)),
        "parameters": n_params,
        "best_val_iou_at_save": float(ckpt.get("val_iou", 0.0)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{arch}_resnet18.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> wrote {json_path}")
    print(f"  -> metrics: {metrics}")
    return payload


# ---------------------------------------------------------------------------
# Training pipeline for a single architecture
# ---------------------------------------------------------------------------

def run_arch(arch: str, config: dict, device: torch.device,
             out_dir: Path, log_to_tensorboard: bool = True) -> dict:
    print("\n" + "=" * 70)
    print(f"BASELINE: {arch}")
    print("=" * 70)

    set_seed(config["project"]["seed"])

    train_loader, val_loader = build_dataloaders(config)
    n_train_batches = len(train_loader)
    print(f"Train batches: {n_train_batches}, Val batches: {len(val_loader)}")

    model = build_architecture(arch)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: smp.{arch} / resnet18 | params={n_params:,}")

    criterion = WeightedBCEDiceLoss(
        pos_weight=config["loss"]["pos_weight"],
        dice_weight=config["loss"]["dice_weight"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=config["training"]["scheduler"]["factor"],
        patience=config["training"]["scheduler"]["patience"],
        min_lr=config["training"]["scheduler"]["min_lr"],
    )
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping"]["patience"],
        metric=config["training"]["early_stopping"]["metric"],
    )

    n_epochs = config["training"]["epochs"]
    best_iou = 0.0
    best_state = None
    best_epoch = 0
    final_epoch = 0

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"{arch}_resnet18_best.pth"
    status_path = out_dir / f"status_{arch}.json"

    writer = None
    if log_to_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = PROJECT_ROOT / config["outputs"]["log_dir"] / f"baseline_{arch}_{ts}"
        writer = SummaryWriter(log_dir)

    print(f"\nTraining for up to {n_epochs} epochs...")
    print(f"Best checkpoint path: {best_ckpt_path}")
    print(f"Status file: {status_path}")
    print(f"Total batches/epoch: {n_train_batches} train + {len(val_loader)} val")

    t_start = time.time()
    epoch_pbar = tqdm(range(1, n_epochs + 1), desc=f"[{arch}] epochs", ncols=120,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for epoch_idx in epoch_pbar:
        epoch = epoch_idx - 1
        epoch_t = time.time()
        model.train()
        tracker = MetricTracker()
        tracker.reset()

        train_pbar = tqdm(train_loader,
                          desc=f"  ep{epoch_idx:>3} train",
                          ncols=120, leave=False,
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        running_loss = 0.0
        running_n = 0
        for batch_i, (patches, masks) in enumerate(train_pbar):
            patches = patches.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(patches)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits.detach())
            tracker.update(probs, masks.detach(), loss.item())

            running_loss += loss.item() * patches.size(0)
            running_n += patches.size(0)
            train_pbar.set_postfix(loss=f"{running_loss/max(running_n,1):.4f}")

            # Update status file every 10 batches so external monitors can poll it
            if (batch_i + 1) % 10 == 0:
                elapsed_total = time.time() - t_start
                done_batches = epoch_idx * n_train_batches + (batch_i + 1)
                total_batches = n_epochs * n_train_batches
                eta_sec = (elapsed_total / max(done_batches, 1)) * (total_batches - done_batches)
                _write_status(status_path, {
                    "arch": arch,
                    "phase": "train",
                    "epoch": epoch_idx,
                    "batch": batch_i + 1,
                    "batches_per_epoch": n_train_batches,
                    "running_train_loss": running_loss / max(running_n, 1),
                    "best_val_iou": best_iou,
                    "best_epoch": best_epoch,
                    "elapsed_sec": int(elapsed_total),
                    "eta_sec": int(eta_sec),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })

        train_metrics = tracker.compute_metrics()
        train_pbar.close()

        # Validation
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
            torch.save({
                "arch": arch,
                "epoch": best_epoch,
                "val_iou": best_iou,
                "model_state_dict": best_state,
            }, best_ckpt_path)
            print(f"  [ep{epoch_idx}] NEW BEST val_iou={best_iou:.4f} (checkpoint saved)")

        epoch_sec = time.time() - epoch_t
        elapsed_total = time.time() - t_start
        eta_sec = (elapsed_total / max(epoch_idx, 1)) * (n_epochs - epoch_idx)

        epoch_pbar.set_postfix(
            trL=f"{train_metrics['loss']:.3f}",
            vlL=f"{val_metrics['loss']:.3f}",
            vlIoU=f"{val_metrics['iou']:.4f}",
            best=f"{best_iou:.4f}@ep{best_epoch}",
            ep_t=f"{epoch_sec:.0f}s",
        )

        # Update status file at end of each epoch
        _write_status(status_path, {
            "arch": arch,
            "phase": "val_done",
            "epoch": epoch_idx,
            "batches_per_epoch": n_train_batches,
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "best_val_iou": best_iou,
            "best_epoch": best_epoch,
            "epoch_sec": epoch_sec,
            "elapsed_sec": int(elapsed_total),
            "eta_sec": int(eta_sec),
            "early_stopping_counter": early_stopping.counter,
            "early_stopping_patience": early_stopping.patience,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })

        final_epoch = epoch + 1
        if early_stopping(val_metrics["iou"]):
            print(f"\n  [ep{epoch_idx}] Early stopping triggered (no improvement for {early_stopping.patience} epochs).")
            break

    epoch_pbar.close()
    train_time = time.time() - t_start
    print(f"\nTraining finished in {train_time/60:.1f} min ({train_time/3600:.2f} hours)")
    print(f"Best val IoU={best_iou:.4f} at epoch {best_epoch} (ran {final_epoch} epochs total)")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nEvaluating best checkpoint on {len(val_loader.dataset)} val patches...")
    metrics = evaluate_model(model, val_loader, device, desc=f"{arch} eval")
    print(f"Final metrics: {metrics}")

    if writer is not None:
        writer.close()

    _write_status(status_path, {
        "arch": arch,
        "phase": "done",
        "epoch": final_epoch,
        "best_epoch": best_epoch,
        "best_val_iou": best_iou,
        "metrics": metrics,
        "train_time_sec": int(train_time),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })

    payload = {
        "model": arch,
        "encoder": "resnet18",
        "iou": metrics["iou"],
        "dice": metrics["dice"],
        "accuracy": metrics["accuracy"],
        "hd95": metrics["hd95"],
        "bf1": metrics["bf1"],
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "parameters": n_params,
        "train_time_minutes": round(train_time / 60.0, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{arch}_resnet18.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> wrote {json_path}")

    return payload


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

UNET_REFERENCE_RESULTS = {
    "model": "U-Net/ResNet18 (LUCID)",
    "iou": 0.9203,
    "dice": 0.9576,
    "accuracy": 0.9640,
    "hd95": 0.3454,
    "bf1": 0.9373,
}


def build_comparison(results: dict, out_dir: Path) -> list[dict]:
    """Build the consolidated comparison table across all three models."""
    rows = [
        UNET_REFERENCE_RESULTS,
        {
            "model": "U-Net++/ResNet18",
            "iou": results["unetplusplus"]["iou"],
            "dice": results["unetplusplus"]["dice"],
            "accuracy": results["unetplusplus"]["accuracy"],
            "hd95": results["unetplusplus"]["hd95"],
            "bf1": results["unetplusplus"]["bf1"],
        },
        {
            "model": "DeepLabV3+/ResNet18",
            "iou": results["deeplabv3plus"]["iou"],
            "dice": results["deeplabv3plus"]["dice"],
            "accuracy": results["deeplabv3plus"]["accuracy"],
            "hd95": results["deeplabv3plus"]["hd95"],
            "bf1": results["deeplabv3plus"]["bf1"],
        },
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "comparison_table.json"
    with open(table_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nConsolidated comparison saved to: {table_path}")
    return rows


def print_ascii_table(rows: list[dict]) -> None:
    header = f"{'Model':<25} {'IoU':>8} {'Dice':>8} {'Acc':>8} {'HD95':>8} {'B-F1':>8}"
    sep = "-" * len(header)
    print()
    print(sep)
    print("MODEL COMPARISON (Shackleton-02 mixed-boundary validation)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['model']:<25} "
            f"{r['iou']:>8.4f} "
            f"{r['dice']:>8.4f} "
            f"{r['accuracy']:>8.4f} "
            f"{r['hd95']:>8.4f} "
            f"{r['bf1']:>8.4f}"
        )
    print(sep)
    print("Notes: HD95 lower is better; all other metrics higher is better.")
    print("       IoU=Dice*2/(1+Dice) monotonic relationship; small differences are meaningful.")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run U-Net++ and DeepLabV3+ baselines.")
    parser.add_argument("--arch", type=str, default="all",
                        choices=["unetplusplus", "deeplabv3plus", "all"],
                        help="Which architecture(s) to train.")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to YAML config.")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging.")
    parser.add_argument("--out-dir", type=str, default="baselines",
                        help="Output directory for per-model JSON files.")
    parser.add_argument("--eval-from-ckpt", action="store_true",
                        help="Skip training; evaluate the saved best checkpoint and exit.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {config['project']['seed']}")
    print(f"Batch size: {config['training']['batch_size']}, "
          f"LR: {config['training']['learning_rate']}, "
          f"WD: {config['training'].get('weight_decay', 1e-4)}")
    print(f"Max epochs: {config['training']['epochs']}, "
          f"Early-stopping patience: {config['training']['early_stopping']['patience']}")
    print(f"Loss: weighted BCE(pos_w={config['loss']['pos_weight']}) + "
          f"Dice(weight={config['loss']['dice_weight']})")
    print(f"Pseudo-labels: Multi-Otsu with fallback threshold "
          f"{config['mask']['fallback_threshold']}")

    out_dir = PROJECT_ROOT / args.out_dir
    log_tb = not args.no_tensorboard

    if args.eval_from_ckpt:
        results = {}
        archs = ["unetplusplus", "deeplabv3plus"]
        for arch in archs:
            ckpt_path = out_dir / "checkpoints" / f"{arch}_resnet18_best.pth"
            if not ckpt_path.exists():
                print(f"  No checkpoint for {arch} at {ckpt_path}; skipping.")
                continue
            results[arch] = evaluate_from_checkpoint(arch, ckpt_path, config, device, out_dir)
        if len(results) == 2:
            rows = build_comparison(results, out_dir)
            print_ascii_table(rows)
        return

    archs = ["unetplusplus", "deeplabv3plus"] if args.arch == "all" else [args.arch]
    results = {}
    for arch in archs:
        results[arch] = run_arch(arch, config, device, out_dir, log_to_tensorboard=log_tb)

    if len(archs) == 2:
        rows = build_comparison(results, out_dir)
        print_ascii_table(rows)


if __name__ == "__main__":
    main()