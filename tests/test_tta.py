"""Tests for Test-Time Augmentation."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from src.models.tta import TTAPredictor, tta_predict


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class TestTTAPredictor:
    def test_output_shape(self):
        model = SimpleModel()
        predictor = TTAPredictor(model, torch.device('cpu'), n_views=6)
        patch = np.random.rand(64, 64).astype(np.float32)
        result = predictor.predict(patch)
        assert result.shape == (64, 64)

    def test_n_views(self):
        model = SimpleModel()
        predictor = TTAPredictor(model, torch.device('cpu'), n_views=3)
        patch = np.random.rand(64, 64).astype(np.float32)
        result = predictor.predict(patch, return_all=True)
        assert result.shape[0] == 3

    def test_batch_predict(self):
        model = SimpleModel()
        predictor = TTAPredictor(model, torch.device('cpu'), n_views=6)
        patches = np.random.rand(4, 64, 64).astype(np.float32)
        results = predictor.predict_batch(patches)
        assert results.shape == (4, 64, 64)

    def test_invariance_to_flips(self):
        model = SimpleModel()
        predictor = TTAPredictor(model, torch.device('cpu'), n_views=6)
        patch = np.random.rand(64, 64).astype(np.float32)
        pred1 = predictor.predict(patch)
        pred2 = predictor.predict(np.fliplr(patch).copy())
        diff = np.abs(pred1 - np.fliplr(pred2)).mean()
        assert diff < 0.1
