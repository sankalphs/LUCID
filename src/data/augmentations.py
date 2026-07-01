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


def get_train_transforms_no_aug() -> A.Compose | None:
    """Return None to disable training augmentations entirely.

    Returning ``None`` is the contract used by ``PSRDataset.__getitem__`` to
    skip albumentations entirely (no normalization-only pipeline needed for
    this task, since pixel values are already in ``[0, 1]`` float32).
    """
    return None


def get_train_transforms_no_clahe(aug_cfg: dict) -> A.Compose:
    """Build the training pipeline with CLAHE removed.

    Args:
        aug_cfg: The ``augmentation`` section of the project config. The
            CLAHE probability is forced to 0.0 before the pipeline is built.
    """
    import copy
    cfg = copy.deepcopy(aug_cfg) if aug_cfg else {}
    if "clahe" in cfg:
        cfg["clahe"]["p"] = 0.0

    transforms = []
    if cfg.get("horizontal_flip", 0) > 0:
        transforms.append(A.HorizontalFlip(p=cfg["horizontal_flip"]))
    if cfg.get("vertical_flip", 0) > 0:
        transforms.append(A.VerticalFlip(p=cfg["vertical_flip"]))
    if cfg.get("random_rotate90", 0) > 0:
        transforms.append(A.RandomRotate90(p=cfg["random_rotate90"]))
    if "random_brightness_contrast" in cfg:
        bc = cfg["random_brightness_contrast"]
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=tuple(bc["brightness_limit"]),
            contrast_limit=tuple(bc["contrast_limit"]),
            p=bc["p"],
        ))
    if "random_gamma" in cfg:
        transforms.append(A.RandomGamma(
            gamma_limit=tuple(cfg["random_gamma"]["gamma_limit"]),
            p=cfg["random_gamma"]["p"],
        ))
    if "coarse_dropout" in cfg:
        cd = cfg["coarse_dropout"]
        transforms.append(A.CoarseDropout(
            num_holes_range=tuple(cd["num_holes"]),
            hole_height_range=tuple(cd["hole_height"]),
            hole_width_range=tuple(cd["hole_width"]),
            fill_value=cd["fill_value"],
            p=cd["p"],
        ))

    return A.Compose(transforms) if transforms else None
