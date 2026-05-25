"""Tests for CPU optimization utilities."""
import pytest
import torch
import torch.nn as nn
from src.models.cpu_optimization import (
    optimize_model_for_cpu, GradientAccumulator,
    get_cpu_optimization_config, apply_memory_efficient_training
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestOptimizeModelForCPU:
    def test_returns_model(self):
        model = SimpleModel()
        optimized = optimize_model_for_cpu(model, use_compile=False, use_channels_last=False)
        assert isinstance(optimized, nn.Module)

    def test_channels_last(self):
        model = SimpleModel()
        optimized = optimize_model_for_cpu(model, use_compile=False, use_channels_last=True)
        x = torch.randn(1, 1, 64, 64)
        out = optimized(x)
        assert out.shape == x.shape


class TestGradientAccumulator:
    def test_should_step(self):
        acc = GradientAccumulator(accumulation_steps=4)
        assert not acc.should_step()
        assert not acc.should_step()
        assert not acc.should_step()
        assert acc.should_step()

    def test_reset(self):
        acc = GradientAccumulator(accumulation_steps=4)
        for _ in range(4):
            acc.should_step()
        acc.reset()
        assert acc.current_step == 0


class TestGetCPUConfig:
    def test_returns_dict(self):
        config = get_cpu_optimization_config()
        assert isinstance(config, dict)
        assert 'num_threads' in config
        assert 'use_compile' in config


class TestMemoryEfficient:
    def test_wrapper_works(self):
        model = SimpleModel()
        wrapped = apply_memory_efficient_training(model)
        x = torch.randn(1, 1, 64, 64)
        out = wrapped(x)
        assert out.shape[2:] == x.shape[2:]
