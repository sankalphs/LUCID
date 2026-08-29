"""
Full evaluation script with tqdm progress bars.
Evaluates saved model + baseline + TTA + post-processing.
Generates all paper figures.
"""
import sys
sys.path.insert(0, '.')

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import yaml
import json
import torch
import time
from pathlib import Path
from tqdm import tqdm

from src.data.dataset import PSRDataset
from src.data.augmentations import get_val_transforms
from src.data.mask_generator import generate_mask, clean_mask
from src.baselines.intensity_threshold import intensity_baseline
from src.evaluate import Evaluator, compute_all_metrics
from src.models.tta import TTAPredictor
from src.models.post_processing import morphological_postprocess, gaussian_smooth_prediction
from src.visualize import (
    plot_mask_validation, plot_confusion_matrix,
    plot_baseline_comparison, plot_qualitative_results,
    plot_cross_crater_comparison
)
import segmentation_models_pytorch as smp


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    patches_dir = str(Path(config["data"]["base_dir"]) / "patches")
    val_transform = get_val_transforms()
    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    fig_dir = Path(config["outputs"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print("\nLoading model...")
    model = smp.Unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=None,
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"],
        activation=None,
    ).to(device)

    ckpt_path = ckpt_dir / "best_model.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded best checkpoint from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint at {ckpt_path}")
        return

    model.eval()

    # ── Load validation data ──
    print("Loading validation data...")
    val_ds = PSRDataset(patches_dir, "mixed", "val", transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=0
    )
    print(f"Val: {len(val_ds)} patches")

    # ── Phase 1: Model prediction ──
    print("\n" + "=" * 60)
    print("PHASE 1: MODEL PREDICTION")
    print("=" * 60)

    evaluator = Evaluator()
    all_patches, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for patches, masks in tqdm(val_loader, desc="Predicting", ncols=80):
            patches = patches.to(device)
            logits = model(patches)
            probs = torch.sigmoid(logits)
            evaluator.update(probs.cpu().numpy(), masks.numpy())
            all_patches.append(patches.cpu().numpy().squeeze(1))
            all_targets.append(masks.numpy().squeeze(1))
            all_probs.append(probs.cpu().numpy().squeeze(1))

    all_patches = np.concatenate(all_patches)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    evaluator.print_summary("U-Net Model Results")
    model_results = evaluator.compute_aggregate()

    # ── Phase 2: Intensity Baseline ──
    print("\n" + "=" * 60)
    print("PHASE 2: INTENSITY BASELINE")
    print("=" * 60)

    thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    best_iou, best_thresh = 0, 0.1

    for t in thresholds:
        preds = np.array([intensity_baseline(p, threshold=t) for p in all_patches])
        ev = Evaluator()
        ev.update(preds[:, np.newaxis, :, :], all_targets[:, np.newaxis, :, :])
        m = ev.compute_aggregate()
        iou_val = m.get("iou", {}).get("mean", 0)
        if iou_val > best_iou:
            best_iou = iou_val
            best_thresh = t

    print(f"Best threshold: {best_thresh} (IoU={best_iou:.4f})")

    baseline_preds = np.array([intensity_baseline(p, threshold=best_thresh) for p in all_patches])
    baseline_eval = Evaluator()
    baseline_eval.update(baseline_preds[:, np.newaxis, :, :], all_targets[:, np.newaxis, :, :])
    baseline_eval.print_summary(f"Intensity Baseline (threshold={best_thresh})")
    baseline_results = baseline_eval.compute_aggregate()

    # ── Phase 3: Comparison ──
    print("\n" + "=" * 60)
    print("MODEL vs BASELINE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Model':>10} {'Baseline':>10} {'Delta':>10} {'Status':>8}")
    print("-" * 62)
    for key in ["iou", "dice", "pixel_accuracy", "hd95", "boundary_f1"]:
        m_val = model_results.get(key, {}).get("mean", 0)
        b_val = baseline_results.get(key, {}).get("mean", 0)
        delta = m_val - b_val
        sign = "+" if delta > 0 else ""
        if key == "hd95":
            status = "PASS" if m_val < b_val else "FAIL"
        else:
            status = "PASS" if m_val > b_val else "FAIL"
        print(f"{key:<20} {m_val:>10.4f} {b_val:>10.4f} {sign}{delta:>9.4f} {status:>8}")

    # ── Phase 4: TTA ──
    print("\n" + "=" * 60)
    print("PHASE 3: TEST-TIME AUGMENTATION")
    print("=" * 60)

    predictor = TTAPredictor(model, device, n_views=6)
    evaluator_tta = Evaluator()

    for i in tqdm(range(len(all_patches)), desc="TTA predict", ncols=80):
        pred_tta = predictor.predict(all_patches[i])
        evaluator_tta.update(
            pred_tta[np.newaxis, np.newaxis, :, :],
            all_targets[i:i+1, np.newaxis, :, :]
        )

    evaluator_tta.print_summary("With TTA (all val patches)")
    tta_results = evaluator_tta.compute_aggregate()

    print("\nTTA Improvement over base model:")
    for key in ["iou", "dice", "hd95", "boundary_f1"]:
        mv = model_results.get(key, {}).get("mean", 0)
        tv = tta_results.get(key, {}).get("mean", 0)
        d = tv - mv
        print(f"  {key:<20} Base: {mv:.4f}  TTA: {tv:.4f}  Delta: {'+' if d>0 else ''}{d:.4f}")

    # ── Phase 5: Post-Processing ──
    print("\n" + "=" * 60)
    print("PHASE 4: POST-PROCESSING")
    print("=" * 60)

    ev_raw = Evaluator()
    ev_morph = Evaluator()
    ev_gauss = Evaluator()

    for i in tqdm(range(len(all_probs)), desc="Post-processing", ncols=80):
        prob = all_probs[i]
        target = all_targets[i]

        ev_raw.update(prob[np.newaxis, np.newaxis, :, :], target[np.newaxis, np.newaxis, :, :])

        morph = morphological_postprocess(prob)
        ev_morph.update(morph[np.newaxis, np.newaxis, :, :], target[np.newaxis, np.newaxis, :, :])

        gauss = gaussian_smooth_prediction(prob, sigma=1.0)
        ev_gauss.update(gauss[np.newaxis, np.newaxis, :, :], target[np.newaxis, np.newaxis, :, :])

    ev_raw.print_summary("Raw Prediction")
    ev_morph.print_summary("Morphological Post-Processing")
    ev_gauss.print_summary("Gaussian Smoothed")

    raw_results = ev_raw.compute_aggregate()
    morph_results = ev_morph.compute_aggregate()
    gauss_results = ev_gauss.compute_aggregate()

    # ── Phase 6: Figures ──
    print("\n" + "=" * 60)
    print("PHASE 5: GENERATING PAPER FIGURES")
    print("=" * 60)

    # Confusion matrix
    pred_binary = (all_probs > 0.5).astype(np.uint8).flatten()
    target_binary = (all_targets > 0.5).astype(np.uint8).flatten()
    plot_confusion_matrix(pred_binary, target_binary, output_path=str(fig_dir / "confusion_matrix.png"))
    print("  [1/4] Confusion matrix saved")

    # Qualitative results
    plot_qualitative_results(
        all_patches[:12], all_targets[:12], all_probs[:12],
        n_samples=min(6, len(all_patches)),
        output_path=str(fig_dir / "qualitative_results.png")
    )
    print("  [2/4] Qualitative results saved")

    # Baseline comparison
    plot_baseline_comparison(
        model_results, baseline_results,
        output_path=str(fig_dir / "baseline_comparison.png")
    )
    print("  [3/4] Baseline comparison saved")

    # Mask validation
    mixed_data = np.load(Path(patches_dir) / "mixed" / "train.npy").astype(np.float32)
    vis_idx = np.random.choice(len(mixed_data), 10, replace=False)
    vis_patches = mixed_data[vis_idx]
    vis_masks, vis_methods = [], []
    for p in vis_patches:
        m, met = generate_mask(p, "mixed")
        vis_masks.append(clean_mask(m))
        vis_methods.append(met)
    plot_mask_validation(
        vis_patches, np.array(vis_masks), vis_methods, n_samples=10,
        output_path=str(fig_dir / "mask_validation.png")
    )
    print("  [4/4] Mask validation saved")

    # ── Save all results ──
    all_results = {
        "model_results": model_results,
        "baseline_results": baseline_results,
        "tta_results": tta_results,
        "raw_results": raw_results,
        "morph_results": morph_results,
        "gauss_results": gauss_results,
        "comparison": {
            key: {
                "model": model_results.get(key, {}).get("mean", 0),
                "baseline": baseline_results.get(key, {}).get("mean", 0),
                "tta": tta_results.get(key, {}).get("mean", 0),
                "morph": morph_results.get(key, {}).get("mean", 0),
                "gauss": gauss_results.get(key, {}).get("mean", 0),
            }
            for key in ["iou", "dice", "pixel_accuracy", "hd95", "boundary_f1"]
        }
    }
    with open("outputs/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nAll results saved to outputs/results.json")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<25} {'IoU':>8} {'Dice':>8} {'Acc':>8} {'HD95':>8} {'B-F1':>8}")
    print("-" * 70)
    methods = [
        ("Intensity Baseline", baseline_results),
        ("U-Net (Raw)", raw_results),
        ("U-Net + Morph", morph_results),
        ("U-Net + Gaussian", gauss_results),
        ("U-Net + TTA", tta_results),
    ]
    for name, res in methods:
        iou = res.get("iou", {}).get("mean", 0)
        dice = res.get("dice", {}).get("mean", 0)
        acc = res.get("pixel_accuracy", {}).get("mean", 0)
        hd95 = res.get("hd95", {}).get("mean", 0)
        bf1 = res.get("boundary_f1", {}).get("mean", 0)
        print(f"{name:<25} {iou:>8.4f} {dice:>8.4f} {acc:>8.4f} {hd95:>8.4f} {bf1:>8.4f}")

    print("\nTargets: IoU>0.70, Dice>0.80, Acc>0.90, HD95<5px, B-F1>0.50")
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
