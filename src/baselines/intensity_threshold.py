"""
Intensity threshold baseline for shadow segmentation.

Simple thresholding that classifies pixels as illuminated if their
intensity exceeds a fixed threshold. Used as a baseline to demonstrate
that the U-Net provides meaningful improvement.
"""

import numpy as np
from typing import Optional


def intensity_baseline(patch: np.ndarray, 
                       threshold: float = 0.1) -> np.ndarray:
    """
    Simple intensity threshold baseline.
    
    Args:
        patch: (H, W) float32 array, pixel intensities in [0, 1]
        threshold: Intensity threshold. Pixels above are classified as illuminated.
    
    Returns:
        Binary mask (H, W) float32, 0=shadow, 1=illuminated
    """
    return (patch > threshold).astype(np.float32)


def adaptive_threshold_baseline(patch: np.ndarray) -> np.ndarray:
    """
    Adaptive threshold using Otsu's method.
    
    Better than fixed threshold when intensity distribution varies.
    
    Args:
        patch: (H, W) float32 array
    
    Returns:
        Binary mask (H, W) float32
    """
    from skimage.filters import threshold_otsu
    try:
        threshold = threshold_otsu(patch)
        return (patch > threshold).astype(np.float32)
    except ValueError:
        # Fallback for uniform patches
        return (patch > 0.0484).astype(np.float32)


def multi_threshold_baseline(patch: np.ndarray,
                             thresholds: Optional[list[float]] = None) -> dict:
    """
    Evaluate multiple threshold values for baseline comparison.
    
    Args:
        patch: (H, W) float32 array
        thresholds: List of threshold values to test
    
    Returns:
        Dictionary mapping threshold to binary mask
    """
    if thresholds is None:
        thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    
    results = {}
    for t in thresholds:
        results[f'threshold_{t}'] = (patch > t).astype(np.float32)
    
    return results
