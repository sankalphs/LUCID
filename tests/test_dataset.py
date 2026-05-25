"""Tests for dataset and data loading."""
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from src.data.dataset import PSRDataset, CombinedPSRDataset


@pytest.fixture
def temp_patches_dir(tmp_path):
    """Create temporary patches directory with test data."""
    for cls in ['psr', 'sunlit', 'mixed']:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for split in ['train', 'val']:
            patches = np.random.rand(10, 64, 64).astype(np.float32)
            np.save(cls_dir / f'{split}.npy', patches)
    return str(tmp_path)


class TestPSRDataset:
    def test_loads_correctly(self, temp_patches_dir):
        ds = PSRDataset(temp_patches_dir, 'psr', 'train')
        assert len(ds) == 10

    def test_returns_tensors(self, temp_patches_dir):
        ds = PSRDataset(temp_patches_dir, 'psr', 'train')
        patch, mask = ds[0]
        assert isinstance(patch, np.ndarray) or hasattr(patch, 'numpy')
        assert patch.shape == (1, 64, 64)
        assert mask.shape == (1, 64, 64)

    def test_mask_binary(self, temp_patches_dir):
        ds = PSRDataset(temp_patches_dir, 'psr', 'train', clean_masks=False)
        _, mask = ds[0]
        unique = np.unique(mask.numpy())
        assert set(unique).issubset({0.0, 1.0})


class TestCombinedPSRDataset:
    def test_loads_all_classes(self, temp_patches_dir):
        ds = CombinedPSRDataset(temp_patches_dir, 'train')
        assert len(ds) == 30  # 10 per class

    def test_specific_classes(self, temp_patches_dir):
        ds = CombinedPSRDataset(temp_patches_dir, 'train', classes=['psr', 'sunlit'])
        assert len(ds) == 20

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            CombinedPSRDataset('/nonexistent/path', 'train')
