"""
Auto-generate pixel-level binary masks from patch-level labels.

Uses Multi-Otsu thresholding for mixed patches, with fixed thresholds
for pure PSR/sunlit patches. Includes morphological cleanup.

Reference: Otsu, N. (1979). "A Threshold Selection Method from
Gray-Level Histograms." IEEE Transactions on Systems, Man, and
Cybernetics, 9(1), 62–66.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.morphology import closing, disk

logger = logging.getLogger(__name__)

# Data-derived constants from pixel intensity statistics.
# PSR training patches (N=8663): mean=0.0284, std=0.0215, median=0.024
PSR_MEAN: float = 0.0284
FALLBACK_THRESHOLD: float = PSR_MEAN + 0.02  # = 0.0484

VALID_LABELS: frozenset[str] = frozenset({"psr", "sunlit", "mixed"})


def generate_mask(
    patch: np.ndarray, class_label: str, fallback_threshold: float | None = None
) -> tuple[np.ndarray, str]:
    """
    Generate a binary mask from a patch based on its class label.

    For pure 'psr' or 'sunlit' patches, returns uniform masks using fixed
    thresholds derived from training set statistics. For 'mixed' patches,
    applies Multi-Otsu thresholding to separate shadow from illuminated
    pixels within the patch.

    Args:
        patch: 2D float32 array (H, W) with pixel intensities in [0, 1].
        class_label: One of 'psr', 'sunlit', or 'mixed'.
        fallback_threshold: Optional override for the fallback threshold
            used when Multi-Otsu fails on near-uniform mixed patches.
            If None, uses the module-level FALLBACK_THRESHOLD
            (= PSR_MEAN + 0.02 = 0.0484).

    Returns:
        A tuple of:
            mask: 2D float32 array (H, W) with values in {0.0, 1.0}.
                  0.0 = shadow, 1.0 = illuminated.
            method: String identifying the thresholding method used.
                One of 'psr_fixed', 'sunlit_fixed', 'multi_otsu', or
                'fallback'.

    Raises:
        ValueError: If class_label is not in VALID_LABELS.
        TypeError: If patch is not a numpy array.
        ValueError: If patch is not 2D.
    """
    if class_label not in VALID_LABELS:
        raise ValueError(
            f"Invalid class_label '{class_label}'. "
            f"Must be one of {sorted(VALID_LABELS)}."
        )
    if not isinstance(patch, np.ndarray):
        raise TypeError(f"patch must be a numpy ndarray, got {type(patch).__name__}.")
    if patch.ndim != 2:
        raise ValueError(f"patch must be 2D (H, W), got {patch.ndim}D.")

    threshold = FALLBACK_THRESHOLD if fallback_threshold is None else float(fallback_threshold)

    if class_label == "psr":
        return np.zeros_like(patch, dtype=np.float32), "psr_fixed"
    elif class_label == "sunlit":
        return np.ones_like(patch, dtype=np.float32), "sunlit_fixed"
    else:
        # Mixed patch: apply Multi-Otsu to separate shadow/illuminated.
        try:
            thresholds = threshold_multiotsu(patch, classes=3)
            # Use the lowest threshold to separate the darkest class
            # (shadow) from the rest (illuminated transition + sunlit).
            mask = (patch > thresholds[0]).astype(np.float32)
            return mask, "multi_otsu"
        except ValueError:
            # Multi-Otsu fails when the patch lacks sufficient contrast
            # (e.g., near-uniform intensity). Fall back to a fixed
            # threshold derived from PSR training set statistics.
            logger.debug(
                "Multi-Otsu failed on patch with std=%.4f; using fallback threshold=%.4f",
                patch.std(),
                threshold,
            )
            return (patch > threshold).astype(np.float32), "fallback"


def clean_mask(mask: np.ndarray, disk_radius: int = 1) -> np.ndarray:
    """
    Apply morphological closing to remove small holes and salt noise.

    Closing (dilation followed by erosion) fills small gaps in the
    illuminated regions and smooths jagged mask edges without altering
    the overall boundary topology.

    Args:
        mask: Binary mask (H, W) with values in {0.0, 1.0}.
        disk_radius: Radius of the disk structuring element for the
            morphological operation. Larger values produce smoother
            boundaries but may over-smooth fine detail. Default: 1.

    Returns:
        Cleaned binary mask (H, W) as float32 with values in {0.0, 1.0}.
    """
    return closing(mask, disk(disk_radius)).astype(np.float32)


def validate_mask_quality(
    patches: np.ndarray,
    masks: np.ndarray,
    methods: list[str],
) -> dict[str, Any]:
    """
    Validate mask quality across a batch of patches.

    Computes fallback frequency, per-method counts, and mask statistics.
    Issues a warning if fallback rate exceeds 20%, indicating that many
    mixed patches lack sufficient contrast for Multi-Otsu.

    Args:
        patches: (N, H, W) float32 array of patches.
        masks: (N, H, W) float32 array of generated masks.
        methods: List of N method strings (one per patch).

    Returns:
        Dictionary with keys:
            total_patches: int
            method_counts: dict mapping method name to count
            fallback_rate: float in [0, 1]
            mean_illuminated_fraction: float in [0, 1]
            std_illuminated_fraction: float
            mean_patch_std: float
            warning_low_std: bool — True if fallback_rate > 0.2
    """
    total = len(methods)

    method_counts: dict[str, int] = {}
    for m in methods:
        method_counts[m] = method_counts.get(m, 0) + 1

    fallback_rate = method_counts.get("fallback", 0) / total if total > 0 else 0.0

    # Per-patch statistics
    illuminated_fractions = np.array([m.mean() for m in masks])
    patch_stds = np.array([p.std() for p in patches])

    stats: dict[str, Any] = {
        "total_patches": total,
        "method_counts": method_counts,
        "fallback_rate": float(fallback_rate),
        "mean_illuminated_fraction": float(illuminated_fractions.mean()),
        "std_illuminated_fraction": float(illuminated_fractions.std()),
        "mean_patch_std": float(patch_stds.mean()),
        "warning_low_std": bool(fallback_rate > 0.2),
    }

    if stats["warning_low_std"]:
        logger.warning(
            "High fallback rate (%.1f%%). Consider filtering mixed patches "
            "with std < 0.03 before mask generation.",
            fallback_rate * 100,
        )

    return stats


def generate_masks_batch(
    patches: np.ndarray,
    class_labels: list[str],
    clean: bool = True,
    fallback_threshold: float | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """
    Generate masks for a batch of patches with validation.

    Iterates over each patch, generates a binary mask using the
    appropriate method for its class label, optionally applies
    morphological cleaning, and validates overall mask quality.

    Args:
        patches: (N, H, W) float32 array of patches.
        class_labels: List of N class labels ('psr', 'sunlit', 'mixed').
        clean: Whether to apply morphological closing to each mask.
            Default: True.
        fallback_threshold: Optional override for the fallback threshold
            passed to each ``generate_mask`` call. See ``generate_mask``.

    Returns:
        A tuple of:
            masks: (N, H, W) float32 array of binary masks.
            methods: List of N method strings.
            stats: Validation statistics dictionary (see validate_mask_quality).
    """
    N = patches.shape[0]
    if len(class_labels) != N:
        raise ValueError(
            f"Length mismatch: {N} patches but {len(class_labels)} labels."
        )

    masks = np.zeros_like(patches, dtype=np.float32)
    methods: list[str] = []

    for i in range(N):
        mask, method = generate_mask(patches[i], class_labels[i], fallback_threshold)
        if clean:
            mask = clean_mask(mask)
        masks[i] = mask
        methods.append(method)

    stats = validate_mask_quality(patches, masks, methods)

    return masks, methods, stats


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Run a self-contained demonstration of the mask generator."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rng = np.random.default_rng(seed=42)
    H, W = 64, 64

    # Synthesise test patches mimicking real intensity distributions.
    # PSR: very dark, low variance.
    psr_patch = rng.normal(loc=0.028, scale=0.02, size=(H, W)).clip(0, 1).astype(np.float32)
    # Sunlit: bright, moderate variance.
    sunlit_patch = rng.normal(loc=0.44, scale=0.16, size=(H, W)).clip(0, 1).astype(np.float32)
    # Mixed: mostly dark with a bright region (simulated boundary).
    mixed_patch = rng.normal(loc=0.05, scale=0.04, size=(H, W)).clip(0, 1).astype(np.float32)
    mixed_patch[20:44, 32:64] = rng.normal(loc=0.30, scale=0.10, size=(24, 32)).clip(0, 1)

    patches = np.stack([psr_patch, sunlit_patch, mixed_patch], axis=0)
    labels = ["psr", "sunlit", "mixed"]

    # Generate masks
    masks, methods, stats = generate_masks_batch(patches, labels, clean=True)

    # Report results
    print("=" * 60)
    print("Mask Generator — Demonstration")
    print("=" * 60)

    for i, (label, method) in enumerate(zip(labels, methods)):
        frac = masks[i].mean()
        print(f"  Patch {i} ({label:>6s}): method={method:<12s}  "
              f"illuminated_fraction={frac:.3f}")

    print()
    print("Validation statistics:")
    for k, v in stats.items():
        if k == "method_counts":
            print(f"  {k}:")
            for mk, mv in v.items():
                print(f"    {mk}: {mv}")
        else:
            print(f"  {k}: {v}")

    print("=" * 60)


if __name__ == "__main__":
    _demo()
