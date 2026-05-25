"""
Full-strip inference for PSR Shadow Boundary Segmentation.

Implements sliding window inference on raw OHRC image strips with
Gaussian-weighted overlap blending for seamless predictions.
Processes images in bands to manage memory on CPU-only systems.
"""

import numpy as np
import torch
from typing import Optional
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries
import yaml
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def gaussian_weight_map(window_size: int, sigma: float = None) -> np.ndarray:
    """
    Generate 2D Gaussian weight map for overlap blending.
    
    Args:
        window_size: Size of the window (e.g., 64)
        sigma: Standard deviation. If None, uses window_size / 4
    
    Returns:
        2D weight map with maximum at center
    """
    if sigma is None:
        sigma = window_size / 4.0
    
    ax = np.arange(window_size) - window_size / 2.0 + 0.5
    xx, yy = np.meshgrid(ax, ax)
    weights = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return weights.astype(np.float32)


def sliding_window_predict(model, image: np.ndarray,
                           window_size: int = 64, stride: int = 32,
                           batch_size: int = 64,
                           device: torch.device = None) -> np.ndarray:
    """
    Sliding window prediction with Gaussian-weighted overlap blending.
    
    Args:
        model: Trained segmentation model
        image: 2D float32 array (H, W), raw image
        window_size: Patch size for inference
        stride: Step size between patches (50% overlap recommended)
        batch_size: Number of patches to process simultaneously
        device: Torch device for inference
    
    Returns:
        2D float32 array (H, W) of probability maps in [0, 1]
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    H, W = image.shape
    prob_map = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)
    
    weights = gaussian_weight_map(window_size)
    
    rows = range(0, H - window_size + 1, stride)
    cols = range(0, W - window_size + 1, stride)
    
    patches_batch = []
    positions_batch = []
    
    for r in rows:
        for c in cols:
            patch = image[r:r + window_size, c:c + window_size]
            patches_batch.append(patch)
            positions_batch.append((r, c))
            
            if len(patches_batch) >= batch_size:
                _process_batch(model, patches_batch, positions_batch,
                             prob_map, weight_sum, weights, window_size, device)
                patches_batch = []
                positions_batch = []
    
    if patches_batch:
        _process_batch(model, patches_batch, positions_batch,
                     prob_map, weight_sum, weights, window_size, device)
    
    weight_sum = np.maximum(weight_sum, 1e-8)
    prob_map = (prob_map / weight_sum).astype(np.float32)
    
    return prob_map


def _process_batch(model, patches_batch, positions_batch,
                   prob_map, weight_sum, weights, window_size, device):
    """Process a batch of patches through the model."""
    batch = np.array(patches_batch)
    batch_tensor = torch.from_numpy(batch[:, np.newaxis, :, :]).to(device)
    
    with torch.no_grad():
        preds = torch.sigmoid(model(batch_tensor)).cpu().numpy().squeeze(1)
    
    for i, (r, c) in enumerate(positions_batch):
        prob_map[r:r + window_size, c:c + window_size] += preds[i] * weights
        weight_sum[r:r + window_size, c:c + window_size] += weights


def extract_boundaries(prob_map: np.ndarray, 
                       threshold: float = 0.5,
                       mode: str = 'outer') -> np.ndarray:
    """
    Extract shadow-illumination boundaries from probability map.
    
    Args:
        prob_map: 2D float32 probability map
        threshold: Binary threshold for segmentation
        mode: Boundary mode ('outer', 'inner', 'thick', 'subpixel')
    
    Returns:
        Binary boundary map (H, W) uint8
    """
    binary_mask = (prob_map > threshold).astype(np.uint8)
    return find_boundaries(binary_mask, mode=mode).astype(np.uint8)


def infer_strip(model, strip_path: str, config: dict,
                output_dir: Optional[str] = None,
                band_rows: int = 128,
                device: Optional[torch.device] = None) -> dict:
    """
    Run full inference on an OHRC image strip.
    
    Args:
        model: Trained model
        strip_path: Path to directory containing image.img
        config: Configuration dictionary
        output_dir: Directory to save outputs (optional)
        band_rows: Number of rows to process at once (memory management)
        device: Torch device
    
    Returns:
        Dictionary with probability map and boundary map paths
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    strip_dir = Path(strip_path)
    
    img_files = list(strip_dir.glob("*.img"))
    if not img_files:
        raise FileNotFoundError(f"No .img files found in {strip_dir}")
    
    img_file = img_files[0]
    metadata_path = strip_dir / "metadata.npy"
    if metadata_path.exists():
        metadata = np.load(metadata_path, allow_pickle=True).item()
        H = metadata['height']
        W = metadata['width']
    else:
        H, W = 100000, 12000
    
    window_size = config.get('inference', {}).get('window_size', 64)
    stride = config.get('inference', {}).get('stride', 32)
    
    prob_map = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)
    weights = gaussian_weight_map(window_size)
    
    with open(img_file, 'rb') as f:
        image_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(H, W)
    
    image_data = image_data.astype(np.float32) / 255.0
    
    print(f"Processing strip: {strip_dir.name} ({H}x{W})")
    
    n_bands = (H + band_rows - 1) // band_rows
    
    for band_idx in range(n_bands):
        start_row = band_idx * band_rows
        end_row = min(start_row + window_size, H)
        
        if start_row >= H:
            break
        
        print(f"  Band {band_idx + 1}/{n_bands} (rows {start_row}-{end_row})")
        
        band = image_data[start_row:end_row, :]
        
        band_prob = sliding_window_predict(
            model, band, window_size, stride, batch_size=64, device=device
        )
        
        actual_rows = band_prob.shape[0]
        prob_map[start_row:start_row + actual_rows, :] += band_prob
        weight_sum[start_row:start_row + actual_rows, :] += 1.0
    
    weight_sum = np.maximum(weight_sum, 1e-8)
    prob_map = (prob_map / weight_sum).astype(np.float32)
    
    boundary = extract_boundaries(prob_map, threshold=0.5)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'probability_map.npy', prob_map)
        np.save(output_dir / 'boundary_map.npy', boundary)
        
        print(f"  Saved probability map: {output_dir / 'probability_map.npy'}")
        print(f"  Saved boundary map: {output_dir / 'boundary_map.npy'}")
    
    return {
        'probability_map': prob_map,
        'boundary_map': boundary,
    }


if __name__ == '__main__':
    print("Inference module loaded. Use infer_strip() for full-strip processing.")
