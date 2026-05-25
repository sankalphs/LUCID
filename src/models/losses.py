"""
Loss functions for shadow boundary segmentation.

Implements Weighted BCE + Dice loss combination that handles class imbalance
in lunar PSR imagery where shadow pixels dominate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEDiceLoss(nn.Module):
    """
    Combined weighted BCE and Dice loss for binary segmentation.
    
    Weighted BCE addresses class imbalance by assigning higher weight
    to the illuminated (minority) class. Dice loss encourages boundary
    sharpness and handles spatial imbalance.
    
    Args:
        pos_weight: Weight for illuminated pixels in BCE. Computed from
                    dataset statistics (e.g., 3.0 if 75% shadow).
        dice_weight: Weight for dice loss component (0-1).
        smooth: Smoothing factor to avoid division by zero.
    """
    
    def __init__(self, pos_weight: float = 3.0, dice_weight: float = 0.5,
                 smooth: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE + Dice loss.
        
        Args:
            pred: (B, 1, H, W) predicted probabilities in [0, 1]
            target: (B, 1, H, W) ground truth binary mask in {0, 1}
        
        Returns:
            Scalar loss value
        """
        # Weighted BCE
        weight = torch.where(target == 1, self.pos_weight, 1.0)
        bce = F.binary_cross_entropy(pred, target, weight=weight)
        
        # Dice loss
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return (1 - self.dice_weight) * bce + self.dice_weight * dice


class DiceLoss(nn.Module):
    """Pure Dice loss for binary segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    """
    Focal loss for handling hard examples in segmentation.
    
    Reference: Lin et al. (2017). "Focal Loss for Dense Object Detection."
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def compute_pos_weight(masks: torch.Tensor) -> float:
    """
    Compute positive weight from mask statistics.
    
    Args:
        masks: (N, 1, H, W) binary masks
    
    Returns:
        Weight for illuminated class
    """
    total = masks.numel()
    pos = masks.sum().item()
    neg = total - pos
    if pos == 0:
        return 1.0
    return neg / pos
