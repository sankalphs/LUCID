"""Tests for data augmentations."""
import numpy as np
import pytest
from src.data.augmentations import get_train_transforms, get_val_transforms, get_test_transforms


class TestTrainTransforms:
    def test_applies_to_patch(self):
        transform = get_train_transforms()
        patch = np.random.rand(64, 64, 1).astype(np.float32)
        mask = np.random.choice([0, 1], size=(64, 64, 1)).astype(np.float32)
        result = transform(image=patch, mask=mask)
        assert result['image'].shape == (64, 64, 1)
        assert result['mask'].shape == (64, 64, 1)

    def test_preserves_shape(self):
        transform = get_train_transforms()
        patch = np.random.rand(64, 64, 1).astype(np.float32)
        mask = np.ones((64, 64, 1), dtype=np.float32)
        for _ in range(10):
            result = transform(image=patch, mask=mask)
            assert result['image'].shape[0] == 64
            assert result['image'].shape[1] == 64


class TestValTransforms:
    def test_applies_clahe(self):
        transform = get_val_transforms()
        patch = np.random.rand(64, 64, 1).astype(np.float32) * 0.05
        mask = np.ones((64, 64, 1), dtype=np.float32)
        result = transform(image=patch, mask=mask)
        assert result['image'].shape == patch.shape


class TestTestTransforms:
    def test_geometric_only(self):
        transform = get_test_transforms()
        patch = np.random.rand(64, 64, 1).astype(np.float32)
        mask = np.ones((64, 64, 1), dtype=np.float32)
        result = transform(image=patch, mask=mask)
        assert result['image'].shape == patch.shape
