"""Tests for post-processing methods."""
import numpy as np
import pytest
from src.models.post_processing import (
    morphological_postprocess, gaussian_smooth_prediction, boundary_snapping
)


def _has_binary_fill_holes():
    try:
        from skimage.morphology import binary_fill_holes
        return True
    except ImportError:
        return False


class TestMorphologicalPostprocess:
    @pytest.mark.skipif(
        not _has_binary_fill_holes(),
        reason="binary_fill_holes not available in this skimage version"
    )
    def test_output_shape(self):
        pred = np.random.rand(64, 64).astype(np.float32)
        result = morphological_postprocess(pred)
        assert result.shape == pred.shape

    @pytest.mark.skipif(
        not _has_binary_fill_holes(),
        reason="binary_fill_holes not available in this skimage version"
    )
    def test_output_binary(self):
        pred = np.random.rand(64, 64).astype(np.float32)
        result = morphological_postprocess(pred)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    @pytest.mark.skipif(
        not _has_binary_fill_holes(),
        reason="binary_fill_holes not available in this skimage version"
    )
    def test_removes_noise(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        pred[32, 32] = 1.0
        result = morphological_postprocess(pred, disk_size=3)
        assert result.sum() == 0


class TestGaussianSmoothPrediction:
    def test_output_shape(self):
        pred = np.random.rand(64, 64).astype(np.float32)
        result = gaussian_smooth_prediction(pred, sigma=1.0)
        assert result.shape == pred.shape

    def test_smoothed_is_binary(self):
        pred = np.random.rand(64, 64).astype(np.float32)
        result = gaussian_smooth_prediction(pred, sigma=2.0, threshold=0.5)
        assert set(np.unique(result)).issubset({0.0, 1.0})


class TestBoundarySnapping:
    def test_output_shape(self):
        pred = np.random.rand(64, 64).astype(np.float32)
        image = np.random.rand(64, 64).astype(np.float32)
        result = boundary_snapping(pred, image)
        assert result.shape == pred.shape

    def test_output_binary(self):
        pred = np.random.rand(64, 64).astype(np.float32)
        image = np.random.rand(64, 64).astype(np.float32)
        result = boundary_snapping(pred, image)
        assert set(np.unique(result)).issubset({0.0, 1.0})
