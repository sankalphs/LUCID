"""Tests for ensemble methods."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import tempfile
from src.models.ensemble import ModelEnsemble, SWAModel


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class TestModelEnsemble:
    def test_add_model(self):
        ensemble = ModelEnsemble(torch.device('cpu'))
        model1 = SimpleModel()
        model2 = SimpleModel()
        ensemble.add_model(model1)
        ensemble.add_model(model2)
        assert ensemble.n_models == 2

    def test_predict_shape(self):
        ensemble = ModelEnsemble(torch.device('cpu'))
        ensemble.add_model(SimpleModel())
        patch = np.random.rand(64, 64).astype(np.float32)
        result = ensemble.predict(patch)
        assert result.shape == (64, 64)

    def test_predict_average(self):
        ensemble = ModelEnsemble(torch.device('cpu'))
        model1 = SimpleModel()
        model2 = SimpleModel()
        ensemble.add_model(model1)
        ensemble.add_model(model2)
        patch = np.random.rand(64, 64).astype(np.float32)
        result = ensemble.predict(patch)
        assert result.shape == (64, 64)
        assert 0 <= result.min() <= result.max() <= 1


class TestSWAModel:
    def test_update_and_average(self):
        swa = SWAModel(torch.device('cpu'))
        model1 = SimpleModel()
        model2 = SimpleModel()
        swa.update(model1)
        swa.update(model2)
        swa.average()
        assert swa.n_models == 2

    def test_apply_to_model(self):
        swa = SWAModel(torch.device('cpu'))
        model = SimpleModel()
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        swa.update(model)
        swa.average()
        swa.apply_to(model)
