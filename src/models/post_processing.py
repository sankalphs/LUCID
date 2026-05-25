"""
Post-processing for shadow boundary segmentation.

Implements:
- Dense CRF (if pydensecrf available)
- Morphological post-processing
- Gaussian smoothing
- Active contour refinement
"""

import numpy as np
from typing import Optional
from pathlib import Path


def dense_crf_postprocess(image: np.ndarray, prediction: np.ndarray,
                          iterations: int = 5,
                          sxy_gaussian: float = 3.0,
                          sxy_bilateral: float = 50.0,
                          srgb_bilateral: float = 13.0,
                          compat_gaussian: float = 3.0,
                          compat_bilateral: float = 10.0) -> np.ndarray:
    """
    Apply Dense CRF post-processing to refine predictions.

    Uses intensity edges in the original image to snap
    predicted boundaries to actual edges.

    Args:
        image: (H, W) float32 image in [0, 1]
        prediction: (H, W) float32 probability map in [0, 1]
        iterations: Number of CRF iterations
        sxy_gaussian: Spatial bandwidth for Gaussian prior
        sxy_bilateral: Spatial bandwidth for bilateral filter
        srgb_bilateral: Color/intensity bandwidth for bilateral filter
        compat_gaussian: Gaussian compatibility weight
        compat_bilateral: Bilateral compatibility weight

    Returns:
        Refined prediction (H, W) float32
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels

        H, W = prediction.shape

        label_map = (prediction > 0.5).astype(np.uint8)

        n_labels = 2
        d = dcrf.DenseCRF2D(W, H, n_labels)

        unary = unary_from_labels(label_map, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(unary)

        d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        image_uint8 = (image * 255).astype(np.uint8)
        image_3ch = np.stack([image_uint8] * 3, axis=-1)

        d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=image_3ch,
                               compat=compat_bilateral, kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(iterations)
        result = np.argmax(Q, axis=0).reshape(H, W).astype(np.float32)

        return result

    except ImportError:
        print("pydensecrf not available. Using morphological fallback.")
        return morphological_postprocess(prediction)


def morphological_postprocess(prediction: np.ndarray,
                               threshold: float = 0.5,
                               disk_size: int = 2,
                               fill_holes: bool = True) -> np.ndarray:
    """
    Morphological post-processing to clean up predictions.

    Applies opening, closing, and hole filling to produce
    cleaner binary masks.

    Args:
        prediction: (H, W) float32 probability map
        threshold: Binary threshold
        disk_size: Size of morphological structuring element
        fill_holes: Whether to fill holes in regions

    Returns:
        Cleaned binary mask (H, W) float32
    """
    from skimage.morphology import disk, opening, closing
    from scipy.ndimage import binary_fill_holes
    
    binary = (prediction > threshold).astype(np.uint8)
    
    selem = disk(disk_size)
    cleaned = opening(binary, selem)
    cleaned = closing(cleaned, selem)
    
    if fill_holes:
        cleaned = binary_fill_holes(cleaned).astype(np.uint8)

    return cleaned.astype(np.float32)


def gaussian_smooth_prediction(prediction: np.ndarray,
                                sigma: float = 1.0,
                                threshold: float = 0.5) -> np.ndarray:
    """
    Apply Gaussian smoothing to prediction before thresholding.

    Args:
        prediction: (H, W) float32 probability map
        sigma: Gaussian sigma
        threshold: Binary threshold after smoothing

    Returns:
        Binary mask (H, W) float32
    """
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(prediction, sigma=sigma)
    return (smoothed > threshold).astype(np.float32)


def boundary_snapping(prediction: np.ndarray, image: np.ndarray,
                      boundary_band: int = 3) -> np.ndarray:
    """
    Snap predicted boundaries to nearest intensity edges.

    For each boundary pixel, search within a band for the
    strongest intensity gradient and snap to it.

    Args:
        prediction: (H, W) float32 probability map
        image: (H, W) float32 image
        boundary_band: Search band width in pixels

    Returns:
        Snapped binary mask (H, W) float32
    """
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import distance_transform_edt

    binary = (prediction > 0.5).astype(np.uint8)
    boundary = find_boundaries(binary, mode='thick')

    if not boundary.any():
        return binary.astype(np.float32)

    grad_y, grad_x = np.gradient(image)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    dt = distance_transform_edt(~boundary)

    snapped = binary.copy()

    boundary_pixels = np.argwhere(boundary)
    for r, c in boundary_pixels:
        r_min = max(0, r - boundary_band)
        r_max = min(image.shape[0], r + boundary_band + 1)
        c_min = max(0, c - boundary_band)
        c_max = min(image.shape[1], c + boundary_band + 1)

        band = grad_mag[r_min:r_max, c_min:c_max]
        if band.size > 0:
            max_idx = np.unravel_index(band.argmax(), band.shape)
            snap_r = r_min + max_idx[0]
            snap_c = c_min + max_idx[1]

            if prediction[snap_r, snap_c] > 0.3:
                snapped[snap_r, snap_c] = 1

    return snapped.astype(np.float32)
