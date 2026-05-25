"""
Crater-based data splits to prevent spatial leakage.

Train: shackleton_01 + cabeus_01 (all 3 classes)
Val: shackleton_02 (held-out crater)
Test: Per-crater metrics reported separately
"""

import numpy as np
from pathlib import Path
from typing import Optional


def load_split(patches_dir: str, split: str,
               classes: list[str] = None,
               strips: Optional[list[str]] = None) -> dict:
    """
    Load patches and metadata for a given split.

    Args:
        patches_dir: Path to patches/ directory
        split: 'train', 'val', or 'test'
        classes: List of class labels. Default: ['psr', 'sunlit', 'mixed']
        strips: Optional list of strip names to load. If None, loads all.

    Returns:
        Dictionary with:
            'patches': (N, H, W) float32 array
            'masks': (N, H, W) float32 array (auto-generated)
            'class_labels': list of class labels per patch
            'strip_labels': list of strip labels per patch
    """
    if classes is None:
        classes = ['psr', 'sunlit', 'mixed']

    all_patches = []
    all_class_labels = []
    all_strip_labels = []

    base = Path(patches_dir)

    for cls in classes:
        cls_dir = base / cls

        if strips is not None:
            # Load specific strips
            for strip in strips:
                npy_path = cls_dir / f"{strip}.npy"
                if npy_path.exists():
                    patches = np.load(npy_path).astype(np.float32)
                    all_patches.append(patches)
                    all_class_labels.extend([cls] * len(patches))
                    all_strip_labels.extend([strip] * len(patches))
        else:
            # Load the standard split file (train.npy or val.npy)
            npy_path = cls_dir / f"{split}.npy"
            if npy_path.exists():
                patches = np.load(npy_path).astype(np.float32)
                all_patches.append(patches)
                all_class_labels.extend([cls] * len(patches))
                all_strip_labels.extend([cls] * len(patches))

    if not all_patches:
        raise FileNotFoundError(
            f"No data found for split={split} in {patches_dir}"
        )

    patches = np.concatenate(all_patches, axis=0)

    return {
        'patches': patches,
        'class_labels': all_class_labels,
        'strip_labels': all_strip_labels,
    }


def get_split_info(patches_dir: str) -> dict:
    """
    Get summary information about available data splits.

    Args:
        patches_dir: Path to patches/ directory

    Returns:
        Dictionary with per-class per-split counts
    """
    base = Path(patches_dir)
    info = {}

    for cls in ['psr', 'sunlit', 'mixed']:
        cls_dir = base / cls
        if cls_dir.exists():
            info[cls] = {}
            for npy_file in cls_dir.glob("*.npy"):
                data = np.load(npy_file)
                info[cls][npy_file.stem] = {
                    'shape': data.shape,
                    'count': data.shape[0],
                    'dtype': str(data.dtype),
                }

    return info


def print_split_summary(patches_dir: str):
    """Print a formatted summary of available data splits."""
    info = get_split_info(patches_dir)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    for cls, splits in info.items():
        print(f"\nClass: {cls}")
        print("-" * 40)
        total = 0
        for split_name, details in splits.items():
            print(f"  {split_name}: {details['count']} patches, shape={details['shape']}")
            total += details['count']
        print(f"  Total: {total} patches")

    print("\n" + "=" * 60)
    print("SPLIT STRATEGY:")
    print("  Train: shackleton_01 + cabeus_01")
    print("  Val:   shackleton_02 (held-out crater)")
    print("  Test:  Per-crater metrics reported separately")
    print("=" * 60 + "\n")
