"""Tests for evaluation metrics."""
import numpy as np
import pytest
from src.evaluate import (
    compute_iou, compute_dice, compute_pixel_accuracy,
    compute_hd95, compute_boundary_f1, compute_all_metrics, Evaluator
)


class TestIoU:
    def test_perfect_match(self):
        pred = np.ones((64, 64), dtype=np.uint8)
        target = np.ones((64, 64), dtype=np.uint8)
        assert compute_iou(pred, target) == 1.0

    def test_no_overlap(self):
        pred = np.zeros((64, 64), dtype=np.uint8)
        target = np.ones((64, 64), dtype=np.uint8)
        assert compute_iou(pred, target) == 0.0

    def test_partial_overlap(self):
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[:32, :] = 1
        target = np.zeros((64, 64), dtype=np.uint8)
        target[:48, :] = 1
        iou = compute_iou(pred, target)
        assert 0 < iou < 1


class TestDice:
    def test_perfect_match(self):
        pred = np.ones((64, 64), dtype=np.uint8)
        target = np.ones((64, 64), dtype=np.uint8)
        assert abs(compute_dice(pred, target) - 1.0) < 1e-6

    def test_no_overlap(self):
        pred = np.zeros((64, 64), dtype=np.uint8)
        target = np.ones((64, 64), dtype=np.uint8)
        assert compute_dice(pred, target) < 0.01


class TestPixelAccuracy:
    def test_perfect(self):
        pred = np.ones((64, 64), dtype=np.uint8)
        target = np.ones((64, 64), dtype=np.uint8)
        assert compute_pixel_accuracy(pred, target) == 1.0

    def test_half_correct(self):
        pred = np.zeros((64, 64), dtype=np.uint8)
        target = np.zeros((64, 64), dtype=np.uint8)
        pred[:, :32] = 1
        target[:32, :] = 1
        assert abs(compute_pixel_accuracy(pred, target) - 0.5) < 0.01


class TestHD95:
    def test_identical_masks(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        assert compute_hd95(mask, mask) == 0.0

    def test_nonzero_for_different(self):
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[10:20, 10:20] = 1
        target = np.zeros((64, 64), dtype=np.uint8)
        target[40:50, 40:50] = 1
        hd95 = compute_hd95(pred, target)
        assert hd95 > 0


class TestBoundaryF1:
    def test_perfect_boundary(self):
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[16:48, 16:48] = 1
        target = pred.copy()
        bf1 = compute_boundary_f1(pred, target)
        assert bf1 > 0.9

    def test_offset_boundary(self):
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[16:48, 16:48] = 1
        target = np.zeros((64, 64), dtype=np.uint8)
        target[18:50, 18:50] = 1
        bf1 = compute_boundary_f1(pred, target, tolerance=3)
        assert bf1 > 0.5


class TestEvaluator:
    def test_update_and_compute(self):
        evaluator = Evaluator()
        probs = np.random.rand(4, 1, 64, 64).astype(np.float32)
        targets = np.random.randint(0, 2, (4, 1, 64, 64)).astype(np.float32)
        evaluator.update(probs, targets)
        metrics = evaluator.compute_aggregate()
        assert 'iou' in metrics
        assert 'dice' in metrics
        assert 'hd95' in metrics
        assert 'boundary_f1' in metrics

    def test_reset(self):
        evaluator = Evaluator()
        probs = np.random.rand(2, 1, 64, 64).astype(np.float32)
        targets = np.random.randint(0, 2, (2, 1, 64, 64)).astype(np.float32)
        evaluator.update(probs, targets)
        evaluator.reset()
        assert len(evaluator.per_patch_metrics) == 0
