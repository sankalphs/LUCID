"""LUCID standardized experiment framework.

Every experiment launched through train_experiment.py records its full
configuration, logs, checkpoints, predictions and metrics under its own
results subdirectory and appends a row to the central registry
(results/all_results.csv, results/all_results.json).

Training-loop semantics faithfully mirror train_full.py (the script that
produced the published headline numbers): mixed-class patches, activation=None
model + manual sigmoid, WeightedBCEDiceLoss, Adam(lr=1e-3, wd=1e-4),
ReduceLROnPlateau(max, 0.5, 7, 1e-5) stepped on global-confusion val IoU,
EarlyStopping(patience, min_delta=1e-4), batch 32, shuffle=True, num_workers=0.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, ".")

from src.data.mask_generator import generate_mask, clean_mask          # noqa: E402
from src.models.losses import WeightedBCEDiceLoss                      # noqa: E402
from src.evaluate import Evaluator                                     # noqa: E402
from src.train import MetricTracker                                    # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
REGISTRY_CSV = RESULTS_DIR / "all_results.csv"
REGISTRY_JSON = RESULTS_DIR / "all_results.json"

SPLIT_PRESETS = {
    # legacy reproduces train_full.py exactly: materialized train.npy/val.npy
    "legacy": {
        "train_sources": [("mixed", "train")],
        "final_test_sources": [("mixed", "val")],
        "es_val_from": "legacy_val",
    },
    # region-disjoint rebuilds from per-strip arrays only
    "A_strip": {  # closest region-disjoint analogue of the paper split
        "train_sources": [("mixed", "shackleton_01"), ("mixed", "cabeus_01")],
        "final_test_sources": [("mixed", "shackleton_02")],
    },
    "B": {
        "train_sources": [("mixed", "shackleton_01"), ("mixed", "shackleton_02")],
        "final_test_sources": [("mixed", "cabeus_01")],
    },
    "C": {
        "train_sources": [("mixed", "cabeus_01")],
        "final_test_sources": [("mixed", "shackleton_01"), ("mixed", "shackleton_02")],
    },
}

ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def set_seed(seed: int) -> None:
    """Mirrors train_full.set_seed (+CUDA seeding, harmless on CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PatchSourceDataset(Dataset):
    """Loads arbitrary (class_label, array_name) .npy patch stacks and
    synthesizes LUCID pseudo-labels exactly like PSRDataset does."""

    def __init__(self, patches_dir: Path, sources, transform=None,
                 clean_masks: bool = True, fallback_threshold: float | None = None):
        self.transform = transform
        parts, labels = [], []
        for cls, name in sources:
            arr = np.load(Path(patches_dir) / cls / f"{name}.npy").astype(np.float32)
            assert arr.ndim == 3 and arr.shape[1:] == (64, 64)
            masks = np.zeros_like(arr, dtype=np.float32)
            for i in range(len(arr)):
                m, _ = generate_mask(arr[i], cls, fallback_threshold)
                if clean_masks:
                    m = clean_mask(m)
                masks[i] = m
            parts.append((arr, masks))

        self.patches = np.concatenate([p[0] for p in parts])[:, np.newaxis]
        self.masks = np.concatenate([p[1] for p in parts])[:, np.newaxis]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch, mask = self.patches[idx], self.masks[idx]
        if self.transform is not None:
            aug = self.transform(
                image=np.transpose(patch, (1, 2, 0)),
                mask=np.transpose(mask, (1, 2, 0)),
            )
            patch = np.transpose(aug["image"], (2, 0, 1))
            mask = np.transpose(aug["mask"], (2, 0, 1))
        return torch.from_numpy(patch), torch.from_numpy(mask)


class EarlyStopping:
    """Mirrors train_full.EarlyStopping."""

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_score, self.should_stop = 0, None, False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score + self.min_delta:
            self.best_score, self.counter = score, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def build_transforms(no_aug: bool, no_clahe: bool):
    from src.data.augmentations import get_train_transforms, get_val_transforms
    import albumentations as A

    train_tf = None
    if not no_aug:
        if no_clahe:
            # Correct implementation of "training pipeline without CLAHE"
            # (repo helper get_train_transforms_no_clahe never inserts CLAHE
            # and is dead code; documented in EXPERIMENT_AUDIT.md 4.8).
            train_tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.15),
                                           contrast_limit=(-0.3, 0.3), p=0.7),
                A.RandomGamma(gamma_limit=(70, 130), p=0.4),
                A.CoarseDropout(num_holes_range=(1, 4),
                                hole_height_range=(8, 16),
                                hole_width_range=(8, 16), p=0.3),
            ])
        else:
            train_tf = get_train_transforms()
    val_tf = get_val_transforms()  # paper-faithful stochastic CLAHE
    return train_tf, val_tf


def make_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"lucid.{log_path.parent.name}.{time.time_ns()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def evaluate_on(model, dataset, device, out_dir: Path, tag: str,
                logger=None, save_probs: bool = False) -> dict:
    """Evaluator-based evaluation (per-patch mean metrics, src/evaluate.py)."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False,
                        num_workers=0, pin_memory=False)
    ev = Evaluator()
    pooled = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    probs_all = []
    with torch.inference_mode():
        for patches, masks in loader:
            probs = torch.sigmoid(model(patches.to(device)))
            ev.update(probs.cpu(), masks)
            pred = (probs.cpu() > 0.5).numpy().astype(bool)
            tgt = masks.numpy().astype(bool)
            pooled["tp"] += int(np.logical_and(pred, tgt).sum())
            pooled["fp"] += int(np.logical_and(pred, ~tgt).sum())
            pooled["fn"] += int(np.logical_and(~pred, tgt).sum())
            pooled["tn"] += int(np.logical_and(~pred, ~tgt).sum())
            if save_probs:
                probs_all.append(probs.numpy().astype(np.float16))
    agg = {k: {m: float(v) for m, v in d.items()} for k, d in ev.compute_aggregate().items()}
    result = {"metrics_per_patch_aggregate": agg, "global_confusion_counts": pooled}
    flat = {f"{k}_mean": agg[k]["mean"] for k in ("iou", "dice", "pixel_accuracy", "hd95", "boundary_f1")}
    result.update(flat)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"eval_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    if save_probs:
        np.savez_compressed(out_dir / f"predictions_{tag}.npz",
                            probs=np.concatenate(probs_all, axis=0))
    if logger:
        logger.info("EVAL[%s] iou=%.4f dice=%.4f acc=%.4f hd95=%.4f bf1=%.4f" % (
            tag, flat["iou_mean"], flat["dice_mean"], flat["pixel_accuracy_mean"],
            flat["hd95_mean"], flat["boundary_f1_mean"]))
    return result


def plot_curves(history: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = np.arange(1, len(history["val_iou"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="train loss")
    axes[0].plot(ep, history["val_loss"], label="val loss")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(ep, history["train_iou"], label="train IoU")
    axes[1].plot(ep, history["val_iou"], label="val IoU")
    axes[1].set_title("IoU (global confusion)"); axes[1].legend()
    axes[2].plot(ep, history["learning_rate"])
    axes[2].set_yscale("log"); axes[2].set_title("Learning rate")
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


REGISTRY_FIELDS = [
    "exp_id", "timestamp", "arch", "encoder", "split_preset", "classes",
    "seed", "n_train", "n_es_val", "n_test", "pseudo_label_method",
    "fallback_threshold", "morphology", "augmentation", "loss", "optimizer",
    "lr", "batch_size", "weight_decay", "scheduler", "max_epochs",
    "early_stopping_patience", "best_epoch", "final_epoch",
    "checkpoint", "iou", "dice", "accuracy", "hd95", "bf1",
    "clean_iou", "clean_dice", "clean_accuracy", "clean_hd95", "clean_bf1",
    "train_minutes", "torch_threads", "notes",
]


def register_result(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exists = REGISTRY_CSV.exists()
    with open(REGISTRY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
    data = []
    if REGISTRY_JSON.exists():
        data = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    data.append(row)
    REGISTRY_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
