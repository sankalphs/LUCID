"""Unified LUCID experiment runner (Phase 2 framework entry point).

Reproduction example (Phase 1):
  python experiments/train_experiment.py --exp-id repro_seed42 --seed 42

New experiments select any combination of arch/split/loss/aug/seed options.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, ".")

import segmentation_models_pytorch as smp                                    # noqa: E402

from framework import (ARCHITECTURES, PROJECT_ROOT, RESULTS_DIR,             # noqa: E402
                       EarlyStopping, PatchSourceDataset, evaluate_on,
                       load_config, make_logger, plot_curves, register_result,
                       set_seed, stamp)
from src.models.losses import WeightedBCEDiceLoss                            # noqa: E402
from src.train import MetricTracker                                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arch", default="unet",
                   choices=["unet", "unetplusplus", "deeplabv3plus"])
    p.add_argument("--split-preset", default="legacy",
                   choices=["legacy", "A_strip", "B", "C"])
    p.add_argument("--classes", default="mixed",
                   help="comma list among psr,sunlit,mixed used for BOTH train/test sources")
    p.add_argument("--pos-weight", type=float, default=3.0)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--no-clahe", action="store_true")
    p.add_argument("--no-morphology", action="store_true",
                   help="disable closing(disk(1)) in pseudo-label generation")
    p.add_argument("--fallback-threshold", type=float, default=None)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="default from config.yaml (100)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--es-val-frac", type=float, default=0.1,
                   help="fraction of TRAIN held out for early stopping in strip presets")
    p.add_argument("--limit-train", type=int, default=None, help="debug only")
    p.add_argument("--out-root", default=None, help="override results/ root")
    p.add_argument("--eval-only", default=None,
                   help="path to existing state_dict checkpoint; skip training")
    p.add_argument("--no-registry", action="store_true", help="debug runs")
    return p.parse_args()


def build_datasets(args, cfg):
    from src.data.augmentations import get_val_transforms
    from framework import SPLIT_PRESETS, build_transforms

    patches_dir = Path(cfg["data"]["base_dir"]) / "patches"
    classes = [c.strip() for c in args.classes.split(",")]
    preset = SPLIT_PRESETS[args.split_preset]

    train_sources = [(c, name) for (c, name) in preset["train_sources"]
                     if c in classes]
    test_sources = [(c, name) for (c, name) in preset["final_test_sources"]
                    if c in classes]

    train_tf, val_tf = build_transforms(args.no_aug, args.no_clahe)
    clean = not args.no_morphology

    train_full_ds = PatchSourceDataset(
        patches_dir, train_sources, transform=train_tf,
        clean_masks=clean, fallback_threshold=args.fallback_threshold)

    if args.split_preset == "legacy":
        # Paper protocol: early stopping on the materialized val arrays
        # (shackleton_02 content - documented contamination, kept for fidelity).
        from torch.utils.data import ConcatDataset
        parts = []
        for c in classes:
            ds = PatchSourceDataset(patches_dir, [(c, "val")], transform=val_tf,
                                    clean_masks=clean,
                                    fallback_threshold=args.fallback_threshold)
            parts.append(ds)
        es_val_ds = ConcatDataset(parts)
        n_train_view = len(train_full_ds)
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(train_full_ds))
        n_es = max(1, int(round(len(idx) * args.es_val_frac)))
        es_idx, tr_idx = idx[:n_es], idx[n_es:]
        # NOTE: Subset shares the underlying dataset incl. its transform; the
        # es-val subset therefore sees train augmentations. To keep ES honest,
        # we instead build a separate no-geometry copy for the val fraction.
        es_val_full = PatchSourceDataset(
            patches_dir, train_sources, transform=val_tf, clean_masks=clean,
            fallback_threshold=args.fallback_threshold)
        es_val_ds = Subset(es_val_full, es_idx.tolist())
        train_full_ds = Subset(train_full_ds, tr_idx.tolist())
        n_train_view = len(tr_idx)

    test_ds = PatchSourceDataset(
        patches_dir, test_sources, transform=get_val_transforms(),
        clean_masks=clean, fallback_threshold=args.fallback_threshold)
    clean_test_ds = PatchSourceDataset(
        patches_dir, test_sources, transform=None, clean_masks=clean,
        fallback_threshold=args.fallback_threshold)

    info = {"n_train": n_train_view, "n_es_val": len(es_val_ds),
            "n_test": len(test_ds)}
    if args.limit_train:
        train_full_ds = Subset(train_full_ds,
                               list(range(min(args.limit_train, n_train_view))))
        info["n_train"] = len(train_full_ds)
    return train_full_ds, es_val_ds, test_ds, clean_test_ds, info


def main():
    args = parse_args()
    cfg = load_config()
    out_root = Path(args.out_root) if args.out_root else RESULTS_DIR / "experiments"
    exp_dir = out_root / f"{args.exp_id}_{stamp()}" if not args.no_registry else \
        RESULTS_DIR / "_debug" / f"{args.exp_id}_{stamp()}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = make_logger(exp_dir / "train.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = {
        "exp_id": args.exp_id, "timestamp": stamp(), "arch": args.arch,
        "encoder": cfg["model"]["encoder"], "split_preset": args.split_preset,
        "classes": args.classes, "seed": args.seed,
        "pseudo_label_method": "multiotsu3_lowest+closing_disk1"
        if not args.no_morphology else "multiotsu3_lowest_no_morphology",
        "fallback_threshold": args.fallback_threshold,
        "morphology": not args.no_morphology,
        "augmentation": "none" if args.no_aug else ("no_clahe" if args.no_clahe else "paper_default"),
        "loss": f"wbce(pos_weight={args.pos_weight})x{1 - args.dice_weight}+dice x{args.dice_weight}",
        "optimizer": "Adam", "lr": cfg["training"]["learning_rate"],
        "batch_size": args.batch_size or cfg["training"]["batch_size"],
        "weight_decay": cfg["training"]["weight_decay"],
        "scheduler": "ReduceLROnPlateau(max,0.5,7,min1e-5)",
        "max_epochs": args.max_epochs or cfg["training"]["epochs"],
        "early_stopping_patience": args.patience,
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
    }
    (exp_dir / "config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    logger.info("EXP %s | %s", args.exp_id, json.dumps(resolved))

    set_seed(args.seed)
    train_ds, es_val_ds, test_ds, clean_test_ds, sizes = build_datasets(args, cfg)
    logger.info("sizes: %s", sizes)

    model = ARCHITECTURES[args.arch](
        encoder_name=cfg["model"]["encoder"], encoder_weights=None,
        in_channels=cfg["model"]["in_channels"], classes=cfg["model"]["classes"],
        activation=None).to(device)
    ckpt_path = None

    if args.eval_only:
        ckpt_path = args.eval_only
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        criterion = WeightedBCEDiceLoss(pos_weight=args.pos_weight,
                                        dice_weight=args.dice_weight).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=resolved["lr"],
                                     weight_decay=resolved["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=7, min_lr=1e-5)
        stopper = EarlyStopping(patience=args.patience)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=resolved["batch_size"], shuffle=True,
            num_workers=0, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            es_val_ds, batch_size=resolved["batch_size"], shuffle=False,
            num_workers=0, pin_memory=False)

        best_iou, best_epoch, final_epoch = -1.0, 0, 0
        history = {"train_loss": [], "val_loss": [], "train_iou": [],
                   "val_iou": [], "val_dice": [], "val_accuracy": [],
                   "learning_rate": []}
        ckpt_path = str(exp_dir / "best.pth")
        t_start = time.time()

        for epoch in range(1, resolved["max_epochs"] + 1):
            t0 = time.time()
            model.train()
            tr = MetricTracker(); tr.reset()
            for patches, masks in train_loader:
                optimizer.zero_grad()
                logits = model(patches.to(device))
                loss = criterion(logits, masks.to(device))
                loss.backward()
                optimizer.step()
                pred = torch.sigmoid(logits.detach())
                tr.update(pred.cpu(), masks, loss.item())

            model.eval()
            vm = MetricTracker(); vm.reset()
            vloss_sum, nb = 0.0, 0
            with torch.inference_mode():
                for patches, masks in val_loader:
                    logits = model(patches.to(device))
                    loss = criterion(logits, masks.to(device)).item()
                    vloss_sum += loss; nb += 1
                    pred = torch.sigmoid(logits).cpu()
                    vm.update(pred, masks, loss)

            trm, vtm = tr.compute_metrics(), vm.compute_metrics()
            scheduler.step(vtm["iou"])
            lr_now = optimizer.param_groups[0]["lr"]

            for k, v in (("train_loss", trm["loss"]), ("val_loss", vloss_sum / max(nb, 1)),
                         ("train_iou", trm["iou"]), ("val_iou", vtm["iou"]),
                         ("val_dice", vtm["dice"]), ("val_accuracy", vtm.get("accuracy", 0)),
                         ("learning_rate", lr_now)):
                history[k].append(float(v))

            marker = ""
            if vtm["iou"] > best_iou:
                best_iou, best_epoch = float(vtm["iou"]), epoch
                torch.save(model.state_dict(), ckpt_path)
                marker = " *"

            logger.info(
                "epoch %03d | trL %.4f trI %.4f | vlL %.4f vlI %.4f vlD %.4f "
                "| lr %.6f | %.1fs%s", epoch, trm["loss"], trm["iou"],
                vloss_sum / max(nb, 1), vtm["iou"], vtm["dice"], lr_now,
                time.time() - t0, marker)

            final_epoch = epoch
            (exp_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8")
            plot_curves(history, exp_dir / "training_curves.png")

            if stopper(vtm["iou"]):
                logger.info("early stop at epoch %d", epoch)
                break

        train_minutes = (time.time() - t_start) / 60.0
        resolved.update({"best_epoch": best_epoch, "final_epoch": final_epoch,
                         "best_val_global_iou": best_iou,
                         "train_minutes": round(train_minutes, 1)})
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu",
                                         weights_only=True))

    eval_clahe = evaluate_on(model, test_ds, device, exp_dir, "test_valclahe",
                             logger, save_probs=True)
    eval_clean = evaluate_on(model, clean_test_ds, device, exp_dir,
                             "test_notransform", logger)

    row = {**resolved}
    row.update({
        "checkpoint": ckpt_path or "",
        "iou": round(eval_clahe["iou_mean"], 4), "dice": round(eval_clahe["dice_mean"], 4),
        "accuracy": round(eval_clahe["pixel_accuracy_mean"], 4),
        "hd95": round(eval_clahe["hd95_mean"], 4), "bf1": round(eval_clahe["boundary_f1_mean"], 4),
        "clean_iou": round(eval_clean["iou_mean"], 4),
        "clean_dice": round(eval_clean["dice_mean"], 4),
        "clean_accuracy": round(eval_clean["pixel_accuracy_mean"], 4),
        "clean_hd95": round(eval_clean["hd95_mean"], 4),
        "clean_bf1": round(eval_clean["boundary_f1_mean"], 4),
        **sizes,
    })
    (exp_dir / "result_row.json").write_text(json.dumps(row, indent=2),
                                             encoding="utf-8")
    if not args.no_registry:
        register_result(row)
    logger.info("DONE %s", json.dumps({k: row.get(k) for k in
                 ("exp_id", "iou", "dice", "accuracy", "hd95", "bf1")}))


if __name__ == "__main__":
    main()
