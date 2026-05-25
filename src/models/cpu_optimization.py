"""
CPU Optimization utilities for shadow boundary segmentation.

Implements techniques for faster training and inference on CPU-only systems:
- torch.compile() for graph optimization
- BFloat16 autocast for reduced precision
- Channels Last memory format
- Thread tuning
- Gradient accumulation
"""

import torch
import torch.nn as nn
from typing import Optional
import contextlib


def optimize_model_for_cpu(model: nn.Module,
                           use_compile: bool = True,
                           use_channels_last: bool = True) -> nn.Module:
    """
    Apply CPU optimizations to a model.

    Args:
        model: PyTorch model
        use_compile: Use torch.compile() for graph optimization
        use_channels_last: Convert to Channels Last memory format

    Returns:
        Optimized model
    """
    torch.set_flush_denormal(True)
    torch.set_num_threads(12)

    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if use_compile:
        try:
            model = torch.compile(model)
            print("Applied torch.compile() optimization")
        except Exception as e:
            print(f"torch.compile() failed: {e}. Using eager mode.")

    return model


@contextlib.contextmanager
def bfloat16_autocast(enabled: bool = True):
    """
    Context manager for BFloat16 autocast on CPU.

    Reduces memory usage and can speed up inference.
    """
    if enabled and torch.cpu.is_available():
        try:
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                yield
        except Exception:
            yield
    else:
        yield


class GradientAccumulator:
    """
    Gradient accumulation for effective batch scaling on CPU.

    Accumulates gradients over multiple micro-batches before
    performing an optimizer step.

    Args:
        accumulation_steps: Number of micro-batches to accumulate
        scaler: Optional gradient scaler for mixed precision
    """

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_step(self) -> bool:
        """Check if optimizer should step."""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0

    def reset(self):
        """Reset step counter."""
        self.current_step = 0


def get_cpu_optimization_config() -> dict:
    """
    Get recommended CPU optimization settings.

    Returns:
        Dictionary of optimization parameters
    """
    n_cores = torch.get_num_threads()

    return {
        'num_threads': min(12, n_cores),
        'use_compile': True,
        'use_channels_last': True,
        'use_bfloat16': True,
        'flush_denormals': True,
        'gradient_accumulation_steps': 4,
        'recommended_batch_size': 16,
    }


def apply_memory_efficient_training(model: nn.Module,
                                     use_amp: bool = True) -> nn.Module:
    """
    Wrap model for memory-efficient training on CPU.

    Args:
        model: PyTorch model
        use_amp: Use automatic mixed precision

    Returns:
        Memory-efficient model wrapper
    """
    class MemoryEfficientWrapper(nn.Module):
        def __init__(self, base_model, use_amp=True):
            super().__init__()
            self.base_model = base_model
            self.use_amp = use_amp

        def forward(self, x):
            if self.use_amp and self.training:
                with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                    return self.base_model(x)
            return self.base_model(x)

    return MemoryEfficientWrapper(model, use_amp)
