"""Tests for loss functions."""
import torch
import numpy as np
import pytest
from src.models.losses import WeightedBCEDiceLoss, DiceLoss, FocalLoss, compute_pos_weight
from src.models.focus_sdf_loss import FocusSDFLoss, CombinedFocusSDFLoss
from src.models.physics_informed_loss import PhysicsInformedLoss, GradientWeightedLoss


class TestWeightedBCEDiceLoss:
    def test_output_is_scalar(self):
        criterion = WeightedBCEDiceLoss(pos_weight=3.0, dice_weight=0.5)
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = criterion(pred, target)
        assert loss.dim() == 0

    def test_perfect_prediction_low_loss(self):
        criterion = WeightedBCEDiceLoss(pos_weight=3.0, dice_weight=0.5)
        target = torch.ones(2, 1, 64, 64)
        pred = torch.ones(2, 1, 64, 64) * 0.99
        loss = criterion(pred, target)
        assert loss.item() < 0.1


class TestDiceLoss:
    def test_perfect_overlap_zero_loss(self):
        criterion = DiceLoss()
        pred = torch.ones(2, 1, 64, 64)
        target = torch.ones(2, 1, 64, 64)
        loss = criterion(pred, target)
        assert loss.item() < 1e-6


class TestFocalLoss:
    def test_output_is_scalar(self):
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = criterion(pred, target)
        assert loss.dim() == 0


class TestFocusSDFLoss:
    def test_output_is_scalar(self):
        criterion = FocusSDFLoss(sigma=5.0, boundary_weight=5.0)
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = torch.zeros(2, 1, 64, 64)
        target[:, :, :32, :] = 1.0
        loss = criterion(pred, target)
        assert loss.dim() == 0

    def test_combined_loss(self):
        criterion = CombinedFocusSDFLoss(pos_weight=3.0)
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = criterion(pred, target)
        assert loss.dim() == 0


class TestPhysicsInformedLoss:
    def test_output_is_scalar(self):
        criterion = PhysicsInformedLoss()
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        image = torch.rand(2, 1, 64, 64)
        loss = criterion(pred, target, image)
        assert loss.dim() == 0

    def test_without_image(self):
        criterion = PhysicsInformedLoss()
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = criterion(pred, target)
        assert loss.dim() == 0


class TestComputePosWeight:
    def test_balanced_masks(self):
        masks = torch.ones(10, 1, 64, 64) * 0.5
        masks[:, :, :32, :] = 0
        masks[:, :, 32:, :] = 1
        weight = compute_pos_weight(masks)
        assert abs(weight - 1.0) < 0.01

    def test_imbalanced_masks(self):
        masks = torch.zeros(10, 1, 64, 64)
        masks[:, :, :16, :] = 1
        weight = compute_pos_weight(masks)
        assert weight > 1.0
