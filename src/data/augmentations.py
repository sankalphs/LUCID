"""
Data augmentations for PSR shadow boundary segmentation.

Uses Albumentations library for efficient GPU-accelerated transforms.
Includes CLAHE for low-contrast PSR patches.
"""

import albumentations as A


def get_train_transforms() -> A.Compose:
    """
    Get training augmentations pipeline.
    
    Includes:
    - Geometric: flips and 90-degree rotations (preserve boundary orientation)
    - Radiometric: CLAHE, brightness/contrast, gamma (handle extreme darkness)
    - Regularization: CoarseDropout for spatial robustness
    
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.CLAHE(
            clip_limit=(1.0, 4.0),
            tile_grid_size=(8, 8),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.15, 0.15),
            contrast_limit=(-0.3, 0.3),
            p=0.7
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            fill_value=0,
            p=0.3
        ),
    ])


def get_val_transforms() -> A.Compose:
    """
    Get validation augmentations (minimal - only CLAHE for consistency).
    
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.CLAHE(
            clip_limit=(1.0, 4.0),
            tile_grid_size=(8, 8),
            p=1.0
        ),
    ])


def get_test_transforms() -> A.Compose:
    """
    Get test-time augmentations (TTA) for inference.
    
    Returns:
        Albumentations Compose pipeline with TTA
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
