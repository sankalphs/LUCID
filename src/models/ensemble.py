"""
Ensemble methods for shadow boundary segmentation.

Implements:
- K-Fold cross-validation with snapshot collection
- Stochastic Weight Averaging (SWA)
- Prediction ensembling by averaging
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import copy


class ModelEnsemble:
    """
    Ensemble of multiple trained models.

    Loads multiple checkpoints and averages their predictions
    for more robust and calibrated outputs.

    Args:
        device: Torch device
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.models = []

    def add_model(self, model: nn.Module):
        """Add a trained model to the ensemble."""
        model.eval()
        model.to(self.device)
        self.models.append(model)

    def load_checkpoint(self, model_class, checkpoint_path: str):
        """
        Load a model from checkpoint and add to ensemble.

        Args:
            model_class: Class/function to create model architecture
            checkpoint_path: Path to .pth checkpoint
        """
        model = model_class()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        self.add_model(model)

    def predict(self, patch: np.ndarray) -> np.ndarray:
        """
        Predict with ensemble averaging.

        Args:
            patch: (H, W) float32 array

        Returns:
            Averaged prediction (H, W) float32
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)

        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = torch.sigmoid(model(tensor))
                predictions.append(pred.cpu().numpy().squeeze())

        return np.mean(predictions, axis=0)

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        """
        Predict with ensemble on batch.

        Args:
            patches: (N, H, W) float32 array

        Returns:
            (N, H, W) float32 predictions
        """
        N = patches.shape[0]
        results = np.zeros_like(patches, dtype=np.float32)

        for i in range(N):
            results[i] = self.predict(patches[i])

        return results

    @property
    def n_models(self) -> int:
        return len(self.models)


class SnapshotEnsemble:
    """
    Collect model snapshots during training for ensemble.

    Saves model checkpoints at regular intervals (snapshots)
    and can load them for ensemble prediction.

    Args:
        save_dir: Directory to save snapshots
        n_snapshots: Number of snapshots to collect
        total_epochs: Total training epochs
    """

    def __init__(self, save_dir: str, n_snapshots: int = 4,
                 total_epochs: int = 100):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_snapshots = n_snapshots
        self.total_epochs = total_epochs
        self.snapshot_interval = max(1, total_epochs // n_snapshots)
        self.snapshot_paths = []

    def maybe_save_snapshot(self, model: nn.Module, epoch: int):
        """
        Save model snapshot at regular intervals.

        Args:
            model: Current model state
            epoch: Current epoch number
        """
        if (epoch + 1) % self.snapshot_interval == 0:
            path = self.save_dir / f'snapshot_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), path)
            self.snapshot_paths.append(path)

    def load_all_snapshots(self, model_class) -> ModelEnsemble:
        """
        Load all saved snapshots into an ensemble.

        Args:
            model_class: Class/function to create model architecture

        Returns:
            ModelEnsemble with all snapshots loaded
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ensemble = ModelEnsemble(device)

        for path in self.snapshot_paths:
            ensemble.load_checkpoint(model_class, str(path))

        return ensemble


class SWAModel:
    """
    Stochastic Weight Averaging (SWA) for model improvement.

    Averages model weights from multiple training checkpoints
    for better generalization.

    Args:
        device: Torch device
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.swa_state_dict = None
        self.n_models = 0

    def update(self, model: nn.Module):
        """
        Update SWA with current model weights.

        Args:
            model: Model with current weights
        """
        state_dict = model.state_dict()

        if self.swa_state_dict is None:
            self.swa_state_dict = {k: v.clone().float()
                                   for k, v in state_dict.items()}
        else:
            for k in self.swa_state_dict:
                self.swa_state_dict[k] += state_dict[k].float()

        self.n_models += 1

    def average(self):
        """Compute average weights."""
        if self.n_models == 0:
            return

        for k in self.swa_state_dict:
            self.swa_state_dict[k] /= self.n_models

    def apply_to(self, model: nn.Module):
        """
        Apply SWA weights to a model.

        Args:
            model: Model to update with SWA weights
        """
        if self.swa_state_dict is None:
            return

        self.average()
        model.load_state_dict(self.swa_state_dict)

    def save(self, path: str):
        """Save SWA weights."""
        if self.swa_state_dict is not None:
            self.average()
            torch.save(self.swa_state_dict, path)

    def load(self, path: str, model: nn.Module):
        """Load SWA weights and apply to model."""
        self.swa_state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(self.swa_state_dict)
