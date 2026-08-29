"""
DINOv2-based mask generation for PSR shadow segmentation.

Alternative to Multi-Otsu thresholding using self-supervised visual features
from DINOv2-ViT-B/14 for creating training masks from unlabeled patches.

Reference: Oquab et al. (2024). "DINOv2: Learning Robust Visual Features
           without Supervision."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DINOViT_SIZE: int = 224
FEATURE_DIM: int = 768


class DINOv2FeatureExtractor:
    """
    Feature extractor using DINOv2-ViT-B/14 via torchvision.

    Extracts patch-level features from grayscale lunar imagery by
    replicating to 3-channel and resizing to DINOv2 input resolution.

    Args:
        device: Torch device for inference. Defaults to CPU.
        batch_size: Number of patches to process simultaneously.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.model: Optional[torch.nn.Module] = None

    def _load_model(self) -> None:
        """Lazy-load DINOv2 model from torchvision."""
        if self.model is not None:
            return

        try:
            from torchvision.models import ViT_B_14_Weights, vit_b_14

            weights = ViT_B_14_Weights.DEFAULT
            self.model = vit_b_14(weights=weights)
            self.model.eval()
            self.model.to(self.device)

            logger.info("Loaded DINOv2-ViT-B/14 on %s", self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load DINOv2 model: {exc}. "
                "Install torchvision with ViT support."
            ) from exc

    def _preprocess_batch(self, patches: np.ndarray) -> torch.Tensor:
        """
        Preprocess grayscale patches for DINOv2.

        - Replicate grayscale to 3 channels: (N, H, W) -> (N, 3, H, W)
        - Resize to 224x224
        - Apply ImageNet normalization

        Args:
            patches: (N, H, W) float32 array in [0, 1].

        Returns:
            Tensor of shape (N, 3, 224, 224), normalized.
        """
        batch_3ch = np.stack([patches] * 3, axis=1)
        tensor = torch.from_numpy(batch_3ch).float()

        h, w = patches.shape[1], patches.shape[2]
        if h != DINOViT_SIZE or w != DINOViT_SIZE:
            tensor = F.interpolate(
                tensor,
                size=(DINOViT_SIZE, DINOViT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean.to(tensor.device)) / std.to(tensor.device)

        return tensor

    def extract_features(self, patches: np.ndarray) -> np.ndarray:
        """
        Extract DINOv2 features from a batch of patches.

        Args:
            patches: (N, H, W) float32 array, grayscale patches in [0, 1].

        Returns:
            features: (N, 768) float32 array of feature vectors.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        self._load_model()
        assert self.model is not None

        n_patches = patches.shape[0]
        all_features: list[np.ndarray] = []

        with torch.no_grad():
            for start_idx in range(0, n_patches, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_patches)
                batch = patches[start_idx:end_idx]

                batch_tensor = self._preprocess_batch(batch).to(self.device)
                features = self.model(batch_tensor)

                # ViT returns (N, num_classes); for embeddings we may get
                # (N, seq_len, dim). Collapse to (N, dim).
                if features.dim() > 2:
                    features = features.mean(dim=1)

                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def dinov2_cluster_mask(
    features: np.ndarray,
    patch_intensities: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate binary masks using K-means clustering on DINOv2 features.

    Runs 2-means clustering on the feature vectors and assigns the cluster
    with the higher mean patch intensity as the "illuminated" class.

    Args:
        features: (N, D) feature vectors from DINOv2.
        patch_intensities: (N,) mean intensities per patch, used to
            determine which cluster corresponds to illuminated regions.

    Returns:
        (N,) float32 array with values in {0.0, 1.0}.  1.0 = illuminated,
        0.0 = shadow.
    """
    n_clusters = min(2, len(features))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)

    if patch_intensities is not None:
        cluster_means: list[float] = []
        for c in range(n_clusters):
            mask = cluster_labels == c
            if mask.any():
                cluster_means.append(float(patch_intensities[mask].mean()))
            else:
                cluster_means.append(0.0)
        illuminated_cluster = int(np.argmax(cluster_means))
    else:
        illuminated_cluster = 1

    binary_labels = (cluster_labels == illuminated_cluster).astype(np.float32)
    return binary_labels


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def generate_dinov2_masks(
    patches_dir: str | Path,
    output_dir: str | Path,
    class_label: str = "mixed",
    split: str = "train",
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Generate DINOv2 masks for all patches of a given class/split.

    Loads patches from ``patches_dir / class_label / split.npy``, extracts
    DINOv2 features, clusters them, and saves the resulting binary labels
    alongside summary statistics.

    Args:
        patches_dir: Path to the ``patches/`` directory.
        output_dir: Directory to save generated masks.
        class_label: Class to generate masks for (``'mixed'``, ``'psr'``,
            ``'sunlit'``).
        split: ``'train'`` or ``'val'``.
        device: Torch device for inference.
        batch_size: Batch size for feature extraction.

    Returns:
        Dictionary with keys ``total_patches``, ``illuminated_fraction``,
        ``shadow_fraction``, ``feature_dim``.

    Raises:
        FileNotFoundError: If the patches file does not exist.
    """
    patches_path = Path(patches_dir) / class_label / f"{split}.npy"

    if not patches_path.exists():
        raise FileNotFoundError(f"Patches not found: {patches_path}")

    patches = np.load(patches_path).astype(np.float32)
    logger.info("Loaded %d patches from %s", len(patches), patches_path)

    extractor = DINOv2FeatureExtractor(device=device, batch_size=batch_size)

    logger.info("Extracting DINOv2 features...")
    features = extractor.extract_features(patches)
    logger.info("Feature shape: %s", features.shape)

    mean_intensities = patches.mean(axis=(1, 2))
    dinov2_labels = dinov2_cluster_mask(features, mean_intensities)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(
        output_path / f"dinov2_labels_{class_label}_{split}.npy",
        dinov2_labels,
    )

    stats: dict[str, Any] = {
        "total_patches": len(patches),
        "illuminated_fraction": float(dinov2_labels.mean()),
        "shadow_fraction": float(1.0 - dinov2_labels.mean()),
        "feature_dim": int(features.shape[1]) if features.ndim > 1 else 0,
    }

    with open(output_path / f"dinov2_stats_{class_label}_{split}.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    logger.info("DINOv2 mask statistics:")
    logger.info("  Total patches: %d", stats["total_patches"])
    logger.info("  Illuminated: %.2f%%", stats["illuminated_fraction"] * 100)
    logger.info("  Shadow: %.2f%%", stats["shadow_fraction"] * 100)
    logger.info("  Saved to: %s", output_path)

    return stats


# ---------------------------------------------------------------------------
# Ablation: compare DINOv2 masks against Multi-Otsu
# ---------------------------------------------------------------------------


def compare_mask_methods(
    otsu_masks: np.ndarray,
    dinov2_masks: np.ndarray,
) -> dict[str, float]:
    """
    Compare Multi-Otsu and DINOv2 mask quality.

    Computes pixel-level IoU and agreement rate between two sets of
    binary masks (flattened), plus the fraction of illuminated pixels
    reported by each method.

    Args:
        otsu_masks: (N, H, W) or (N,) Multi-Otsu generated masks.
        dinov2_masks: (N, H, W) or (N,) DINOv2 generated masks.

    Returns:
        Dictionary with keys:
            iou_between_methods: Intersection-over-Union.
            agreement_rate: Fraction of pixels where both methods agree.
            otsu_illuminated_fraction: Mean illuminated fraction for Otsu.
            dinov2_illuminated_fraction: Mean illuminated fraction for DINOv2.
            fractional_difference: Absolute difference in illuminated
                fractions.
    """
    otsu_flat = otsu_masks.flatten().astype(bool)
    dinov2_flat = dinov2_masks.flatten().astype(bool)

    intersection = np.logical_and(otsu_flat, dinov2_flat).sum()
    union = np.logical_or(otsu_flat, dinov2_flat).sum()
    iou = float(intersection / max(union, 1))

    agreement = float((otsu_flat == dinov2_flat).mean())

    otsu_illuminated = float(otsu_flat.mean())
    dinov2_illuminated = float(dinov2_flat.mean())

    return {
        "iou_between_methods": iou,
        "agreement_rate": agreement,
        "otsu_illuminated_fraction": otsu_illuminated,
        "dinov2_illuminated_fraction": dinov2_illuminated,
        "fractional_difference": abs(otsu_illuminated - dinov2_illuminated),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="DINOv2 mask generation for lunar PSR patches."
    )
    parser.add_argument(
        "--patches_dir",
        type=str,
        default="dataset/kaggle_dataset/patches",
        help="Path to patches/ directory (relative to repo root).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dinov2_masks",
        help="Directory to save generated masks (relative to repo root).",
    )
    parser.add_argument(
        "--class_label",
        type=str,
        default="mixed",
        choices=("psr", "sunlit", "mixed"),
        help="Class to generate masks for.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val"),
        help="Dataset split.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    stats = generate_dinov2_masks(
        args.patches_dir,
        args.output_dir,
        args.class_label,
        args.split,
        batch_size=args.batch_size,
    )
    logger.info("Done: %s", stats)


if __name__ == "__main__":
    _main()
