"""
Physics-Informed Loss for shadow boundary segmentation.

Regularizes segmentation by incorporating physical constraints:
- Illuminance gradient consistency (boundaries align with intensity gradients)
- Shadow region smoothness (shadow regions should be spatially coherent)
- Boundary sharpness (transition should be abrupt, not gradual)

Reference: Physics-Informed Loss (arXiv:2511.20501)
           GitHub: irfantahir301/Physicsis_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss combining segmentation with physical constraints.

    Loss = alpha * BCE + beta * Dice + gamma * PIL

    where PIL penalizes:
    - Gradient inconsistency: boundaries should align with intensity gradients
    - Over-smoothing: shadow regions should be spatially coherent
    - Under-smoothing: illuminated regions should be spatially coherent

    Args:
        gradient_weight: Weight for gradient consistency term
        smoothness_weight: Weight for region smoothness term
        alpha: Weight for BCE component
        beta: Weight for Dice component
        gamma: Weight for Physics-Informed component
        pos_weight: Weight for illuminated class in BCE
    """

    def __init__(self, gradient_weight: float = 1.0,
                 smoothness_weight: float = 0.5,
                 alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2,
                 pos_weight: float = 3.0):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.smoothness_weight = smoothness_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Physics-Informed Loss.

        Args:
            pred: (B, 1, H, W) predicted probabilities
            target: (B, 1, H, W) ground truth binary mask
            image: (B, 1, H, W) original image for gradient consistency

        Returns:
            Scalar loss value
        """
        weight = torch.where(target == 1, self.pos_weight, 1.0)
        bce = F.binary_cross_entropy(pred, target, weight=weight)

        smooth = 1.0
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + smooth) / (
            pred.sum() + target.sum() + smooth
        )

        pil = self._compute_physics_loss(pred, target, image)

        return self.alpha * bce + self.beta * dice + self.gamma * pil

    def _compute_physics_loss(self, pred: torch.Tensor, target: torch.Tensor,
                               image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the physics-informed regularization term.

        Combines:
        1. Gradient consistency: prediction gradients should align with image gradients
        2. Region smoothness: interior regions should be spatially coherent
        """
        gradient_loss = self._gradient_consistency_loss(pred, image)
        smoothness_loss = self._region_smoothness_loss(pred, target)

        return self.gradient_weight * gradient_loss + self.smoothness_weight * smoothness_loss

    def _gradient_consistency_loss(self, pred: torch.Tensor,
                                    image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Penalize misalignment between prediction boundaries and image gradients.

        High image gradient should correspond to high prediction gradient (boundary).
        """
        if image is None:
            return self._boundary_sharpness_loss(pred)

        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        image_grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        image_grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]

        pred_grad_mag = torch.sqrt(pred_grad_x ** 2 + 1e-8)
        image_grad_mag = torch.sqrt(image_grad_x ** 2 + 1e-8)

        pred_grad_norm = pred_grad_mag / (pred_grad_mag.max() + 1e-8)
        image_grad_norm = image_grad_mag / (image_grad_mag.max() + 1e-8)

        loss = F.mse_loss(pred_grad_norm, image_grad_norm)

        return loss

    def _boundary_sharpness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Encourage sharp boundaries (predictions close to 0 or 1).

        Penalizes ambiguous predictions in transition regions.
        """
        pred_flat = pred.view(pred.size(0), -1)

        entropy = -pred_flat * torch.log(pred_flat + 1e-8) - \
                  (1 - pred_flat) * torch.log(1 - pred_flat + 1e-8)

        return entropy.mean()

    def _region_smoothness_loss(self, pred: torch.Tensor,
                                 target: torch.Tensor) -> torch.Tensor:
        """
        Encourage spatial smoothness within shadow and illuminated regions.

        Interior pixels should have similar predictions to their neighbors.
        """
        pred_smooth_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_smooth_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        interior_mask_x = target[:, :, :, 1:] * target[:, :, :, :-1]
        exterior_mask_x = (1 - target[:, :, :, 1:]) * (1 - target[:, :, :, :-1])
        region_mask_x = interior_mask_x + exterior_mask_x

        interior_mask_y = target[:, :, 1:, :] * target[:, :, :-1, :]
        exterior_mask_y = (1 - target[:, :, 1:, :]) * (1 - target[:, :, :-1, :])
        region_mask_y = interior_mask_y + exterior_mask_y

        loss_x = (pred_smooth_x ** 2 * region_mask_x).sum() / (region_mask_x.sum() + 1e-8)
        loss_y = (pred_smooth_y ** 2 * region_mask_y).sum() / (region_mask_y.sum() + 1e-8)

        return (loss_x + loss_y) / 2


class GradientWeightedLoss(nn.Module):
    """
    Simple gradient-weighted loss for boundary emphasis.

    Uses image gradients to automatically weight boundary regions.

    Args:
        pos_weight: Weight for illuminated class
        gradient_sigma: Sigma for Gaussian weighting of gradients
    """

    def __init__(self, pos_weight: float = 3.0, gradient_sigma: float = 3.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gradient_sigma = gradient_sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                image: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient-weighted BCE loss.

        Args:
            pred: (B, 1, H, W) predicted probabilities
            target: (B, 1, H, W) ground truth
            image: (B, 1, H, W) original image

        Returns:
            Scalar loss
        """
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]

        grad_mag = torch.sqrt(grad_x[:, :, :, :-1] ** 2 + grad_y[:, :, :-1, :] ** 2 + 1e-8)
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)

        weights = 1.0 + grad_mag * self.gradient_sigma
        weights = F.interpolate(weights, size=pred.shape[2:], mode='bilinear', align_corners=False)
        weights = weights.expand_as(pred)

        bce = F.binary_cross_entropy(pred, target, reduction='none')

        return (bce * weights).mean()
