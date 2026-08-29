"""Re-evaluate the existing best_model.pth (produced by train_full.py) on the
mixed-class validation split, using the repo's own evaluation implementation,
to verify that outputs/results.json metrics are reproducible from the stored
checkpoint without retraining.

Saves: results/reproduction/checkpoint_reeval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, ".")

from src.data.dataset import PSRDataset          # noqa: E402
from src.data.augmentations import get_val_transforms  # noqa: E402
from src.evaluate import Evaluator               # noqa: E402
from src.train import set_seed                   # noqa: E402


def main() -> None:
    out_dir = PROJECT_ROOT / "results" / "reproduction"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth"
    patches_dir = PROJECT_ROOT / "dataset" / "kaggle_dataset" / "patches"
    device = torch.device("cpu")

    set_seed(0)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    results: dict = {"checkpoint": str(ckpt_path)}

    for label, tf in [("val_clahe_paper_faithful", get_val_transforms()),
                      ("val_no_transform", None)]:
        ds = PSRDataset(patches_dir, "mixed", "val", transform=tf)
        loader = DataLoader(ds, batch_size=16, shuffle=False,
                            num_workers=0, pin_memory=False)
        ev = Evaluator()
        probs_all = []
        with torch.inference_mode():
            for patches, masks in loader:
                probs = torch.sigmoid(model(patches))
                ev.update(probs, masks)
                probs_all.append(probs.numpy().astype(np.float16))
        agg = ev.compute_aggregate()
        results[label] = {
            k: {m: float(v) for m, v in d.items()} for k, d in agg.items()
        }
        if label == "val_clahe_paper_faithful":
            np.savez_compressed(out_dir / "checkpoint_reeval_probs.npz",
                                probs=np.concatenate(probs_all, axis=0))

    versions = {
        "torch": torch.__version__,
        "smp": smp.__version__,
        "numpy": np.__version__,
    }
    try:
        import albumentations
        versions["albumentations"] = albumentations.__version__
    except Exception:
        pass
    results["versions"] = versions

    with open(out_dir / "checkpoint_reeval.json", "w") as f:
        json.dump(results, f, indent=2)

    for label in ("val_clahe_paper_faithful", "val_no_transform"):
        r = results[label]
        print(label,
              "iou=%.4f dice=%.4f acc=%.4f hd95=%.4f bf1=%.4f" % (
                  r["iou"]["mean"], r["dice"]["mean"],
                  r["pixel_accuracy"]["mean"], r["hd95"]["mean"],
                  r["boundary_f1"]["mean"]))


if __name__ == "__main__":
    main()
