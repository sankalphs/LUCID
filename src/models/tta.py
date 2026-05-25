"""
Test-Time Augmentation (TTA) for shadow boundary segmentation.

Applies geometric transforms during inference and averages predictions
after inverse transforms. Zero additional training cost.

6 views: original + H-flip + V-flip + rot90 + rot180 + rot270
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F


class TTAPredictor:
    """
    Test-Time Augmentation predictor.

    Applies multiple geometric transformations to input patches,
    runs model on each augmented view, and averages predictions
    after applying inverse transforms.

    Args:
        model: Trained segmentation model
        device: Torch device
        n_views: Number of augmented views (6 = 3 geometric + 3 flips)
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 n_views: int = 6):
        self.model = model
        self.device = device
        self.n_views = n_views
        self.model.eval()

    def predict(self, patch: np.ndarray,
                return_all: bool = False) -> np.ndarray:
        """
        Predict with TTA on a single patch.

        Args:
            patch: (H, W) float32 array
            return_all: If True, return all view predictions

        Returns:
            Averaged prediction (H, W) float32
        """
        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)

        augmented_views = self._get_augmented_views(tensor)

        predictions = []
        for view, inv_func in augmented_views:
            with torch.no_grad():
                pred = torch.sigmoid(self.model(view))
            pred_inv = inv_func(pred)
            predictions.append(pred_inv.cpu().numpy().squeeze())

        if return_all:
            return np.array(predictions)

        return np.mean(predictions, axis=0)

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        """
        Predict with TTA on a batch of patches.

        Args:
            patches: (N, H, W) float32 array

        Returns:
            Predictions (N, H, W) float32
        """
        N = patches.shape[0]
        results = np.zeros_like(patches, dtype=np.float32)

        for i in range(N):
            results[i] = self.predict(patches[i])

        return results

    def _get_augmented_views(self, tensor: torch.Tensor) -> list:
        """
        Generate augmented views and their inverse functions.

        Returns list of (augmented_tensor, inverse_function) pairs.
        """
        views = []

        views.append((tensor, lambda x: x))

        views.append((
            torch.flip(tensor, dims=[3]),
            lambda x: torch.flip(x, dims=[3])
        ))

        views.append((
            torch.flip(tensor, dims=[2]),
            lambda x: torch.flip(x, dims=[2])
        ))

        views.append((
            torch.rot90(tensor, k=1, dims=[2, 3]),
            lambda x: torch.rot90(x, k=3, dims=[2, 3])
        ))

        views.append((
            torch.rot90(tensor, k=2, dims=[2, 3]),
            lambda x: torch.rot90(x, k=2, dims=[2, 3])
        ))

        views.append((
            torch.rot90(tensor, k=3, dims=[2, 3]),
            lambda x: torch.rot90(x, k=1, dims=[2, 3])
        ))

        return views[:self.n_views]


def tta_predict(model: nn.Module, patches: np.ndarray,
                device: torch.device, n_views: int = 6) -> np.ndarray:
    """
    Convenience function for TTA prediction.

    Args:
        model: Trained model
        patches: (N, H, W) float32 array
        device: Torch device
        n_views: Number of TTA views

    Returns:
        (N, H, W) float32 predictions
    """
    predictor = TTAPredictor(model, device, n_views)
    return predictor.predict_batch(patches)
