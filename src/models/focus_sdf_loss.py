"""
FocusSDF Loss for boundary-aware segmentation.

Implements adaptive weighting based on signed distance to boundary
to focus training on boundary pixels where precision matters most.

Reference: FocusSDF (arXiv:2511.11864) - adaptive weighting by
           signed distance field for improved boundary delineation.

Usage:
    L_total = 0.5 * Dice + 0.5 * FocusSDF
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries


class FocusSDFLoss(nn.Module):
    """
    FocusSDF Loss: adaptive boundary-weighted Dice loss.

    Computes signed distance transform from ground truth boundaries
    and uses it to upweight boundary pixels during training.

    Args:
        sigma: Controls the spatial extent of boundary focus (in pixels)
        boundary_weight: Maximum weight assigned to boundary pixels
        dice_weight: Weight for the FocusSDF component
        smooth: Smoothing factor to avoid division by zero
    """

    def __init__(self, sigma: float = 5.0, boundary_weight: float = 5.0,
                 dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.sigma = sigma
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute FocusSDF loss.

        Args:
            pred: (B, 1, H, W) predicted probabilities in [0, 1]
            target: (B, 1, H, W) ground truth binary mask

        Returns:
            Scalar loss value
        """
        B, C, H, W = pred.shape

        weights = self._compute_boundary_weights(target)

        intersection = (pred * target * weights).sum()
        pred_weighted = pred * weights

        dice = 1 - (2 * intersection + self.smooth) / (
            pred_weighted.sum() + target.sum() + self.smooth
        )

        bce = F.binary_cross_entropy(pred, target, reduction='none')
        weighted_bce = (bce * weights).mean()

        return self.dice_weight * dice + (1 - self.dice_weight) * weighted_bce

    def _compute_boundary_weights(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive weights based on distance to boundary.

        Pixels near boundaries get higher weights.
        Weight decays exponentially with distance from boundary.
        """
        B, C, H, W = target.shape
        weights = torch.ones_like(target)

        for b in range(B):
            mask_np = target[b, 0].cpu().numpy().astype(np.uint8)

            if mask_np.sum() == 0 or mask_np.sum() == mask_np.size:
                continue

            boundary = find_boundaries(mask_np, mode='thick')

            dist_to_boundary = distance_transform_edt(~boundary)

            w = 1.0 + (self.boundary_weight - 1.0) * torch.exp(
                -torch.from_numpy(dist_to_boundary).float() / self.sigma
            )

            weights[b, 0] = w.to(target.device)

        return weights


class CombinedFocusSDFLoss(nn.Module):
    """
    Combined loss: Weighted BCE + Dice + FocusSDF.

    L_total = alpha * WeightedBCE + beta * Dice + gamma * FocusSDF

    Args:
        pos_weight: Weight for illuminated class in BCE
        focus_sigma: Sigma for FocusSDF boundary focus
        focus_boundary_weight: Max weight for boundary pixels
        alpha: Weight for BCE component
        beta: Weight for Dice component
        gamma: Weight for FocusSDF component
    """

    def __init__(self, pos_weight: float = 3.0,
                 focus_sigma: float = 5.0,
                 focus_boundary_weight: float = 5.0,
                 alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4):
        super().__init__()
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focus_sdf = FocusSDFLoss(
            sigma=focus_sigma,
            boundary_weight=focus_boundary_weight,
            dice_weight=0.5
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = torch.where(target == 1, self.pos_weight, 1.0)
        bce = F.binary_cross_entropy(pred, target, weight=weight)

        smooth = 1.0
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + smooth) / (
            pred.sum() + target.sum() + smooth
        )

        focus_sdf = self.focus_sdf(pred, target)

        return self.alpha * bce + self.beta * dice + self.gamma * focus_sdf
