"""Fast pipeline testing with reduced data/epochs for CPU."""
import sys
sys.path.insert(0, '.')

import numpy as np
import yaml
import json
import torch
import time
from pathlib import Path

from src.data.mask_generator import generate_mask, clean_mask, validate_mask_quality
from src.data.splits import print_split_summary
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.baselines.intensity_threshold import intensity_baseline
from src.evaluate import Evaluator, compute_all_metrics
from src.train import set_seed, build_model, train_one_epoch, validate, EarlyStopping
from src.models.losses import WeightedBCEDiceLoss
from src.visualize import (
    plot_mask_validation, plot_confusion_matrix,
    plot_baseline_comparison, plot_qualitative_results
)

from torch.utils.data import DataLoader, Subset


def test_mask_validation(config):
    print("\n" + "=" * 60)
    print("PHASE 1: MASK VALIDATION ON REAL DATA")
    print("=" * 60)

    patches_dir = str(Path(config["data"]["base_dir"]) / "patches")
    print_split_summary(patches_dir)

    all_stats = {}
    for cls in ["psr", "sunlit", "mixed"]:
        npy_path = Path(patches_dir) / cls / "train.npy"
        data = np.load(npy_path).astype(np.float32)
        print(f"\n{cls.upper()} class: {len(data)} patches")

        n_check = min(300, len(data))
        indices = np.random.choice(len(data), n_check, replace=False)
        patches = data[indices]

        masks, methods = [], []
        for p in patches:
            mask, method = generate_mask(p, cls)
            masks.append(clean_mask(mask))
            methods.append(method)

        masks = np.array(masks)
        stats = validate_mask_quality(patches, masks, methods)
        all_stats[cls] = stats

        mc = stats["method_counts"]
        fr = stats["fallback_rate"]
        mif = stats["mean_illuminated_fraction"]
        print(f"  Methods: {mc}")
        print(f"  Fallback rate: {fr:.2%}")
        print(f"  Mean illuminated fraction: {mif:.4f}")

    out_dir = Path(config["outputs"]["figure_dir"]) / "mask_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
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
        output_path=str(out_dir / "mask_validation_real.png")
    )
    print("\n[PASS] Mask validation complete.")
    return all_stats


def test_baseline(config):
    print("\n" + "=" * 60)
    print("PHASE 2: INTENSITY BASELINE EVALUATION")
    print("=" * 60)

    patches_dir = str(Path(config["data"]["base_dir"]) / "patches")
    baseline_results = {}

    for cls in ["psr", "sunlit", "mixed"]:
        npy_path = Path(patches_dir) / cls / "val.npy"
        if not npy_path.exists():
            npy_path = Path(patches_dir) / cls / "train.npy"
        data = np.load(npy_path).astype(np.float32)
        n_eval = min(200, len(data))
        patches = data[:n_eval]

        targets = []
        for p in patches:
            m, _ = generate_mask(p, cls)
            targets.append(m)
        targets = np.array(targets)

        thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
        best_iou, best_thresh = 0, 0
        for t in thresholds:
            preds = np.array([intensity_baseline(p, threshold=t) for p in patches])
            ev = Evaluator()
            ev.update(preds[:, np.newaxis, :, :], targets[:, np.newaxis, :, :])
            m = ev.compute_aggregate()
            iou_val = m.get("iou", {}).get("mean", 0)
            if iou_val > best_iou:
                best_iou = iou_val
                best_thresh = t

        print(f"\n  {cls.upper()}: Best threshold = {best_thresh}, IoU = {best_iou:.4f}")

        preds_best = np.array([intensity_baseline(p, threshold=best_thresh) for p in patches])
        ev = Evaluator()
        ev.update(preds_best[:, np.newaxis, :, :], targets[:, np.newaxis, :, :])
        ev.print_summary(f"Baseline ({cls})")
        baseline_results[cls] = ev.compute_aggregate()

    print("\n[PASS] Baseline evaluation complete.")
    return baseline_results


def test_training(config):
    print("\n" + "=" * 60)
    print("PHASE 3: TRAINING (15 epochs, small data)")
    print("=" * 60)

    set_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    patches_dir = str(Path(config["data"]["base_dir"]) / "patches")
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    from src.data.dataset import PSRDataset
    full_train = PSRDataset(patches_dir, "mixed", "train", transform=train_transform)
    full_val = PSRDataset(patches_dir, "mixed", "val", transform=val_transform)

    train_subset = Subset(full_train, range(min(1000, len(full_train))))
    val_subset = Subset(full_val, range(min(200, len(full_val))))

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Train: {len(train_subset)} patches, Val: {len(val_subset)} patches")

    criterion = WeightedBCEDiceLoss(
        pos_weight=config["loss"]["pos_weight"],
        dice_weight=config["loss"]["dice_weight"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=10, metric="val_iou")

    best_iou = 0.0
    n_epochs = 15
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [], "val_dice": []}

    ckpt_dir = Path(config["outputs"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for epoch in range(n_epochs):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step(val_m["iou"])
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_iou"].append(train_m["iou"])
        history["val_iou"].append(val_m["iou"])
        history["val_dice"].append(val_m["dice"])

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:2d}/{n_epochs} | "
              f"Train Loss: {train_m['loss']:.4f} IoU: {train_m['iou']:.4f} | "
              f"Val Loss: {val_m['loss']:.4f} IoU: {val_m['iou']:.4f} "
              f"Dice: {val_m['dice']:.4f} | LR: {lr:.6f} | {elapsed:.1f}s")

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

        if early_stopping(val_m["iou"]):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best validation IoU: {best_iou:.4f}")

    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n[PASS] Training complete.")
    return model, history


def test_evaluation(model, config):
    print("\n" + "=" * 60)
    print("PHASE 4: MODEL EVALUATION + BASELINE COMPARISON")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(config["outputs"]["checkpoint_dir"]) / "best_model.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded best checkpoint")

    from src.data.dataset import PSRDataset
    from torch.utils.data import DataLoader, Subset

    patches_dir = str(Path(config["data"]["base_dir"]) / "patches")
    val_ds = PSRDataset(patches_dir, "mixed", "val", transform=get_val_transforms())
    val_subset = Subset(val_ds, range(min(200, len(val_ds))))
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=0)

    evaluator = Evaluator()
    model.eval()
    all_patches, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for patches, masks in val_loader:
            patches = patches.to(device)
            probs = torch.sigmoid(model(patches))
            evaluator.update(probs.cpu().numpy(), masks.numpy())
            all_patches.append(patches.cpu().numpy().squeeze(1))
            all_targets.append(masks.numpy().squeeze(1))
            all_probs.append(probs.cpu().numpy().squeeze(1))

    all_patches = np.concatenate(all_patches)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    evaluator.print_summary("U-Net Model Results")
    model_results = evaluator.compute_aggregate()

    baseline_eval = Evaluator()
    for i in range(len(all_patches)):
        pred = intensity_baseline(all_patches[i], threshold=0.1)
        baseline_eval.update(
            pred[np.newaxis, np.newaxis, :, :],
            all_targets[i:i+1, np.newaxis, :, :]
        )
    baseline_eval.print_summary("Intensity Baseline Results")
    baseline_results = baseline_eval.compute_aggregate()

    print("\n" + "-" * 55)
    print(f"{'Metric':<20} {'Model':>10} {'Baseline':>10} {'Delta':>10}")
    print("-" * 55)
    for key in ["iou", "dice", "pixel_accuracy", "hd95", "boundary_f1"]:
        m_val = model_results.get(key, {}).get("mean", 0)
        b_val = baseline_results.get(key, {}).get("mean", 0)
        delta = m_val - b_val
        sign = "+" if delta > 0 else ""
        print(f"{key:<20} {m_val:>10.4f} {b_val:>10.4f} {sign}{delta:>9.4f}")

    results = {"model_results": model_results, "baseline_results": baseline_results}
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)

    fig_dir = Path(config["outputs"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    pred_binary = (all_probs > 0.5).astype(np.uint8).flatten()
    target_binary = (all_targets > 0.5).astype(np.uint8).flatten()
    plot_confusion_matrix(pred_binary, target_binary, output_path=str(fig_dir / "confusion_matrix.png"))
    plot_qualitative_results(
        all_patches[:10], all_targets[:10], all_probs[:10],
        n_samples=min(6, len(all_patches)),
        output_path=str(fig_dir / "qualitative_results.png")
    )
    plot_baseline_comparison(
        model_results, baseline_results,
        output_path=str(fig_dir / "baseline_comparison.png")
    )

    print("\n[PASS] Evaluation complete.")
    return model_results, baseline_results


def test_tta(model, config):
    print("\n" + "=" * 60)
    print("PHASE 5: TEST-TIME AUGMENTATION (20 patches)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.models.tta import TTAPredictor

    from src.data.dataset import PSRDataset
    from torch.utils.data import DataLoader, Subset

    patches_dir = str(Path(config["data"]["base_dir"]) / "patches")
    val_ds = PSRDataset(patches_dir, "mixed", "val", transform=get_val_transforms())
    val_subset = Subset(val_ds, range(min(20, len(val_ds))))
    val_loader = DataLoader(val_subset, batch_size=5, shuffle=False, num_workers=0)

    predictor = TTAPredictor(model, device, n_views=6)

    evaluator_no_tta = Evaluator()
    evaluator_tta = Evaluator()

    with torch.no_grad():
        for patches, masks in val_loader:
            patches_np = patches.numpy().squeeze(1)
            masks_np = masks.numpy()

            for i in range(len(patches_np)):
                pred_no_tta = torch.sigmoid(model(patches[i:i+1].to(device))).cpu().numpy().squeeze()
                pred_tta = predictor.predict(patches_np[i])

                evaluator_no_tta.update(pred_no_tta[np.newaxis, np.newaxis, :, :], masks_np[i:i+1])
                evaluator_tta.update(pred_tta[np.newaxis, np.newaxis, :, :], masks_np[i:i+1])

    evaluator_no_tta.print_summary("Without TTA (20 patches)")
    evaluator_tta.print_summary("With TTA (20 patches)")

    no_tta = evaluator_no_tta.compute_aggregate()
    tta = evaluator_tta.compute_aggregate()

    print("\nTTA Improvement:")
    for key in ["iou", "dice", "hd95", "boundary_f1"]:
        nv = no_tta.get(key, {}).get("mean", 0)
        tv = tta.get(key, {}).get("mean", 0)
        d = tv - nv
        sign = "+" if d > 0 else ""
        print(f"  {key:<20} No TTA: {nv:.4f}  TTA: {tv:.4f}  Delta: {sign}{d:.4f}")

    print("\n[PASS] TTA test complete.")
    return no_tta, tta


def test_postprocessing(model, config):
    print("\n" + "=" * 60)
    print("PHASE 6: POST-PROCESSING (20 patches)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.models.post_processing import morphological_postprocess, gaussian_smooth_prediction

    from src.data.dataset import PSRDataset
    from torch.utils.data import DataLoader, Subset

    patches_dir = str(Path(config["data"]["base_dir"]) / "patches")
    val_ds = PSRDataset(patches_dir, "mixed", "val", transform=get_val_transforms())
    val_subset = Subset(val_ds, range(min(20, len(val_ds))))
    val_loader = DataLoader(val_subset, batch_size=5, shuffle=False, num_workers=0)

    ev_raw = Evaluator()
    ev_morph = Evaluator()
    ev_gauss = Evaluator()

    with torch.no_grad():
        for patches, masks in val_loader:
            probs = torch.sigmoid(model(patches.to(device))).cpu().numpy()
            for i in range(len(probs)):
                prob = probs[i, 0]
                target = masks.numpy()[i, 0]

                ev_raw.update(prob[np.newaxis, np.newaxis, :, :], target[np.newaxis, np.newaxis, :, :])

                morph = morphological_postprocess(prob)
                ev_morph.update(morph[np.newaxis, np.newaxis, :, :], target[np.newaxis, np.newaxis, :, :])

                gauss = gaussian_smooth_prediction(prob, sigma=1.0)
                ev_gauss.update(gauss[np.newaxis, np.newaxis, :, :], target[np.newaxis, np.newaxis, :, :])

    ev_raw.print_summary("Raw Prediction")
    ev_morph.print_summary("Morphological Post-Processing")
    ev_gauss.print_summary("Gaussian Smoothed")

    raw = ev_raw.compute_aggregate()
    morph = ev_morph.compute_aggregate()
    gauss = ev_gauss.compute_aggregate()

    print("\n[PASS] Post-processing test complete.")
    return raw, morph, gauss


def main():
    print("=" * 60)
    print("PSR SHADOW BOUNDARY SEGMENTATION - FULL TESTING")
    print("=" * 60)

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])

    mask_stats = test_mask_validation(config)
    baseline_results = test_baseline(config)
    model, history = test_training(config)
    model_results, baseline_results_val = test_evaluation(model, config)
    no_tta, tta = test_tta(model, config)
    raw, morph, gauss = test_postprocessing(model, config)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY - ALL RESULTS")
    print("=" * 60)

    print("\n[1] MASK VALIDATION: PASSED")
    for cls, stats in mask_stats.items():
        fr = stats["fallback_rate"]
        mif = stats["mean_illuminated_fraction"]
        print(f"    {cls.upper()}: fallback={fr:.2%}, illuminated={mif:.4f}")

    print("\n[2] INTENSITY BASELINE: PASSED")
    for cls in ["psr", "sunlit", "mixed"]:
        if cls in baseline_results:
            iou = baseline_results[cls].get("iou", {}).get("mean", 0)
            print(f"    {cls.upper()}: IoU={iou:.4f}")

    print("\n[3] TRAINING:")
    print(f"    Epochs trained: {len(history['val_iou'])}")
    print(f"    Best Val IoU: {max(history['val_iou']):.4f}")
    print(f"    Best Val Dice: {max(history['val_dice']):.4f}")

    print("\n[4] MODEL vs BASELINE:")
    beat_count = 0
    for key in ["iou", "dice", "pixel_accuracy"]:
        m = model_results.get(key, {}).get("mean", 0)
        b = baseline_results_val.get(key, {}).get("mean", 0)
        status = "PASS" if m > b else "FAIL"
        if m > b:
            beat_count += 1
        print(f"    {key:<20} Model: {m:.4f}  Baseline: {b:.4f}  [{status}]")
    print(f"    U-Net beats baseline on {beat_count}/3 metrics")

    print("\n[5] TTA:")
    for key in ["iou", "dice"]:
        nv = no_tta.get(key, {}).get("mean", 0)
        tv = tta.get(key, {}).get("mean", 0)
        d = tv - nv
        print(f"    {key:<20} No TTA: {nv:.4f}  TTA: {tv:.4f}  Delta: {'+' if d>0 else ''}{d:.4f}")

    print("\n[6] POST-PROCESSING:")
    for key in ["iou", "dice", "boundary_f1"]:
        r = raw.get(key, {}).get("mean", 0)
        mo = morph.get(key, {}).get("mean", 0)
        g = gauss.get(key, {}).get("mean", 0)
        print(f"    {key:<20} Raw: {r:.4f}  Morph: {mo:.4f}  Gauss: {g:.4f}")

    print("\n" + "=" * 60)
    print("ALL TESTING PHASES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
