"""
PyTorch Dataset for PSR Shadow Boundary Segmentation.

Loads .npy patches from the Kaggle dataset and generates binary masks
using the Multi-Otsu mask generation pipeline.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

from .mask_generator import generate_mask, clean_mask


class PSRDataset(Dataset):
    """
    Dataset for PSR shadow/illuminated segmentation.

    Loads patches from .npy files and generates binary masks on-the-fly.
    Each patch is a 64x64 float32 grayscale image.

    Args:
        patches_dir: Path to patches/ directory
        class_label: 'psr', 'sunlit', or 'mixed'
        split: 'train', 'val', or 'test'
        transform: Optional albumentations transform
        clean_masks: Whether to apply morphological closing to masks
    """

    def __init__(self, patches_dir: str, class_label: str, split: str,
                 transform=None, clean_masks: bool = True):
        self.patches_dir = Path(patches_dir)
        self.class_label = class_label
        self.split = split
        self.transform = transform
        self.clean_masks = clean_masks

        # Load patches from .npy file
        npy_path = self.patches_dir / class_label / f"{split}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Patches not found: {npy_path}")

        self.patches = np.load(npy_path).astype(np.float32)
        assert self.patches.ndim == 3, f"Expected 3D array, got {self.patches.ndim}D"
        assert self.patches.shape[1:] == (64, 64), f"Expected 64x64 patches, got {self.patches.shape[1:]}"

        # Generate masks
        self.masks = np.zeros_like(self.patches, dtype=np.float32)
        for i in range(len(self.patches)):
            mask, _ = generate_mask(self.patches[i], class_label)
            if clean_masks:
                mask = clean_mask(mask)
            self.masks[i] = mask

        # Add channel dimension: (N, H, W) -> (N, 1, H, W)
        self.patches = self.patches[:, np.newaxis, :, :]
        self.masks = self.masks[:, np.newaxis, :, :]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]  # (1, 64, 64)
        mask = self.masks[idx]     # (1, 64, 64)

        # Apply augmentation if specified
        if self.transform:
            # Albumentations expects HWC format
            patch_hwc = np.transpose(patch, (1, 2, 0))  # (64, 64, 1)
            mask_hwc = np.transpose(mask, (1, 2, 0))    # (64, 64, 1)
            augmented = self.transform(image=patch_hwc, mask=mask_hwc)
            patch_hwc = augmented['image']
            mask_hwc = augmented['mask']
            # Convert back to CHW
            patch = np.transpose(patch_hwc, (2, 0, 1))
            mask = np.transpose(mask_hwc, (2, 0, 1))

        return torch.from_numpy(patch), torch.from_numpy(mask)


class CombinedPSRDataset(Dataset):
    """
    Combined dataset from multiple class sources (for mixed training).

    Concatenates patches from psr, sunlit, and mixed classes with their
    respective masks.

    Args:
        patches_dir: Path to patches/ directory
        split: 'train' or 'val'
        classes: List of class labels to include
        transform: Optional albumentations transform
        clean_masks: Whether to apply morphological closing
    """

    def __init__(self, patches_dir: str, split: str,
                 classes: list[str] = None,
                 transform=None, clean_masks: bool = True):
        if classes is None:
            classes = ['psr', 'sunlit', 'mixed']

        self.datasets = []
        for cls in classes:
            try:
                ds = PSRDataset(patches_dir, cls, split, transform, clean_masks)
                self.datasets.append(ds)
            except FileNotFoundError:
                continue

        if not self.datasets:
            raise FileNotFoundError(f"No datasets found in {patches_dir} for split={split}")

        # Compute offsets for indexing
        self.lengths = [len(ds) for ds in self.datasets]
        self.offsets = np.cumsum([0] + self.lengths[:-1])
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, offset in enumerate(self.offsets):
            if idx < offset + self.lengths[i]:
                local_idx = idx - offset
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range for combined dataset")
