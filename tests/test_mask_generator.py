"""Tests for mask generation pipeline."""
import numpy as np
import pytest
from src.data.mask_generator import generate_mask, clean_mask, validate_mask_quality


class TestGenerateMask:
    def test_psr_mask_all_zeros(self):
        patch = np.random.rand(64, 64).astype(np.float32) * 0.1
        mask, method = generate_mask(patch, 'psr')
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32
        assert np.all(mask == 0.0)
        assert method == 'psr_fixed'

    def test_sunlit_mask_all_ones(self):
        patch = np.random.rand(64, 64).astype(np.float32) * 0.5 + 0.5
        mask, method = generate_mask(patch, 'sunlit')
        assert np.all(mask == 1.0)
        assert method == 'sunlit_fixed'

    def test_mixed_mask_binary(self):
        patch = np.random.rand(64, 64).astype(np.float32)
        patch[:32, :] = 0.01
        patch[32:, :] = 0.5
        mask, method = generate_mask(patch, 'mixed')
        assert mask.shape == (64, 64)
        assert set(np.unique(mask)).issubset({0.0, 1.0})
        assert method in ('multi_otsu', 'fallback')

    def test_uniform_patch_uses_fallback(self):
        patch = np.full((64, 64), 0.03, dtype=np.float32)
        mask, method = generate_mask(patch, 'mixed')
        assert method == 'fallback'

    def test_invalid_label_raises(self):
        patch = np.random.rand(64, 64).astype(np.float32)
        with pytest.raises(ValueError):
            generate_mask(patch, 'invalid_label')


class TestCleanMask:
    def test_output_shape(self):
        mask = np.random.choice([0, 1], size=(64, 64)).astype(np.float32)
        cleaned = clean_mask(mask)
        assert cleaned.shape == mask.shape

    def test_output_dtype(self):
        mask = np.ones((64, 64), dtype=np.float32)
        cleaned = clean_mask(mask)
        assert cleaned.dtype == np.float32


class TestValidateMaskQuality:
    def test_returns_stats(self):
        patches = np.random.rand(10, 64, 64).astype(np.float32)
        masks = np.random.choice([0, 1], size=(10, 64, 64)).astype(np.float32)
        methods = ['multi_otsu'] * 8 + ['fallback'] * 2
        stats = validate_mask_quality(patches, masks, methods)
        assert stats['total_patches'] == 10
        assert stats['fallback_rate'] == 0.2
        assert 'method_counts' in stats
