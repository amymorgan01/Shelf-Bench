"""
Code to calculate the MDE of ground truth labels with predicted fronts

We use our segmentation masks to derive the front line.

Code inspired by Gourmelen et al. (2022)
"""
import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from scipy.spatial.distance import directed_hausdorff, cdist
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
from omegaconf import DictConfig, OmegaConf
from load_functions import load_model

@dataclass
class ModelSpec:
    arch: str
    name: str
    ckpt_path: str


def normalize_mask_to_2d(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    """
    Robustly convert any mask shape to 2D (H, W).
    
    Args:
        mask: Input mask of any shape
        name: Name for debugging
    
    Returns:
        2D numpy array (H, W)
    """
    original_shape = mask.shape
    original_dtype = mask.dtype
    
    # Step 1: Remove all dimensions of size 1
    mask = mask.squeeze()
    
    # Step 2: If still more than 2D, we need to decide which dimension to keep
    if mask.ndim > 2:
        # Common cases:
        # (batch=1, H, W) -> take [0]
        # (H, W, channels=1) -> take [..., 0]
        # (batch, channels, H, W) -> take [0, 0]
        
        if mask.ndim == 3:
            # 3D case - could be (B, H, W) or (H, W, C)
            if mask.shape[0] <= 4:  # Likely batch/channel dimension
                mask = mask[0]
            elif mask.shape[2] <= 4:  # Likely channel dimension at end
                mask = mask[..., 0]
            else:
                # Ambiguous - try first dimension
                mask = mask[0]
        elif mask.ndim == 4:
            # 4D case - likely (B, C, H, W)
            mask = mask[0, 0]
        else:
            # Unknown - just take first slice repeatedly
            while mask.ndim > 2:
                mask = mask[0]
    
    if mask.ndim != 2:
        raise ValueError(
            f"{name}: Could not reduce to 2D. "
            f"Original shape: {original_shape}, "
            f"Final shape: {mask.shape}"
        )
    
    return mask.astype(np.uint8)


def apply_morphological_filter(mask: np.ndarray, 
                               operation: str = 'opening',
                               iterations: int = 2,
                               structure: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply morphological filtering to clean up mask predictions.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        operation: 'erosion', 'dilation', 'opening', 'closing'
        iterations: Number of iterations to apply
        structure: Structuring element (default: 3x3 cross)
    
    Returns:
        Filtered binary mask
    """
    if structure is None:
        # Default: 3x3 cross-shaped structuring element
        structure = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.uint8)
    
    mask_binary = (mask > 0).astype(np.uint8)
    
    if operation == 'erosion':
        filtered = binary_erosion(mask_binary, structure=structure, iterations=iterations)
    elif operation == 'dilation':
        filtered = binary_dilation(mask_binary, structure=structure, iterations=iterations)
    elif operation == 'opening':
        # Opening = erosion followed by dilation (removes small bright spots)
        filtered = binary_opening(mask_binary, structure=structure, iterations=iterations)
    elif operation == 'closing':
        # Closing = dilation followed by erosion (fills small dark holes)
        filtered = binary_closing(mask_binary, structure=structure, iterations=iterations)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return filtered.astype(np.uint8)


def extract_boundary_contour_v2(mask: np.ndarray,
                                morphological_filter: bool = True,
                                filter_operation: str = 'opening',
                                filter_iterations: int = 2,
                                min_contour_length: int = 50) -> Optional[np.ndarray]:
    """
    Enhanced boundary extraction with optional morphological filtering.
    
    Args:
        mask: Binary mask (any shape/dtype)
        morphological_filter: Whether to apply morphological filtering
        filter_operation: Type of morphological operation
        filter_iterations: Number of filtering iterations
        min_contour_length: Minimum contour length to keep
    
    Returns:
        Nx2 array of boundary coordinates, or None if no boundary exists
    """
    # Normalize to 2D uint8
    try:
        binary_mask = normalize_mask_to_2d(mask, "extract_boundary_contour_v2")
    except ValueError as e:
        print(f"Warning: {e}")
        return None
    
    # Check if boundary exists
    unique_vals = np.unique(binary_mask)
    if len(unique_vals) < 2:
        return None
    
    # Apply morphological filtering if requested
    if morphological_filter:
        binary_mask = apply_morphological_filter(
            binary_mask, 
            operation=filter_operation,
            iterations=filter_iterations
        )
        
        # Check again after filtering
        unique_vals = np.unique(binary_mask)
        if len(unique_vals) < 2:
            return None
    
    # Scale to 0-255 if needed
    if binary_mask.max() == 1:
        binary_mask = binary_mask * 255
    
    # Ensure contiguous memory layout
    binary_mask = np.ascontiguousarray(binary_mask, dtype=np.uint8)
    
    try:
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return None
        
        # Filter by minimum length
        valid_contours = [c for c in contours if len(c) >= min_contour_length]
        
        if len(valid_contours) == 0:
            return None
        
        # Use the longest contour (main ice front)
        longest_contour = max(valid_contours, key=len)
        
        # Reshape from (N, 1, 2) to (N, 2)
        boundary_coords = longest_contour.squeeze()
        
        # Handle single point edge case
        if boundary_coords.ndim == 1:
            boundary_coords = boundary_coords.reshape(1, -1)
        
        # Swap from (x, y) to (row, col) format
        boundary_coords = boundary_coords[:, [1, 0]].astype(np.float32)
        
        return boundary_coords
    
    except cv2.error as e:
        print(f"OpenCV error in extract_boundary_contour_v2: {e}")
        return None


def extract_boundary_from_mask_v2(mask: np.ndarray, 
                                  boundary_width: int = 1,
                                  morphological_filter: bool = True,
                                  filter_operation: str = 'opening',
                                  filter_iterations: int = 2) -> Optional[np.ndarray]:
    """
    Enhanced erosion-based boundary extraction with morphological filtering.
    
    Args:
        mask: Binary mask where 1=ice, 0=ocean
        boundary_width: Width of the boundary in pixels
        morphological_filter: Whether to apply morphological filtering
        filter_operation: Type of morphological operation
        filter_iterations: Number of filtering iterations
    
    Returns:
        Nx2 array of boundary coordinates, or None if no boundary exists
    """
    try:
        binary_mask = normalize_mask_to_2d(mask, "extract_boundary_from_mask_v2")
    except ValueError as e:
        print(f"Warning: {e}")
        return None
    
    # Check if boundary exists
    unique_vals = np.unique(binary_mask)
    if len(unique_vals) < 2:
        return None
    
    # Apply morphological filtering if requested
    if morphological_filter:
        binary_mask = apply_morphological_filter(
            binary_mask,
            operation=filter_operation,
            iterations=filter_iterations
        )
        
        # Check again after filtering
        unique_vals = np.unique(binary_mask)
        if len(unique_vals) < 2:
            return None
    
    # Erosion-based boundary detection
    eroded = binary_erosion(binary_mask, iterations=boundary_width)
    boundary = binary_mask - eroded
    
    # Get coordinates
    boundary_coords = np.argwhere(boundary > 0)
    
    if len(boundary_coords) == 0:
        return None
    
    return boundary_coords.astype(np.float32)


def calculate_boundary_distance(pred_boundary: np.ndarray, 
                                gt_boundary: np.ndarray,
                                pixel_resolution_m: float,
                                metric: str = 'mean') -> float:
    """
    Calculate distance between predicted and ground truth boundaries.
    
    Args:
        pred_boundary: Nx2 array of predicted boundary coordinates
        gt_boundary: Mx2 array of ground truth boundary coordinates
        pixel_resolution_m: Resolution in meters per pixel
        metric: 'mean', 'median', or 'hausdorff'
    
    Returns:
        Distance in meters
    """
    if pred_boundary is None or gt_boundary is None:
        return np.nan
    
    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return np.nan
    
    if metric == 'hausdorff':
        # Symmetric Hausdorff distance
        d1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
        d2 = directed_hausdorff(gt_boundary, pred_boundary)[0]
        distance_pixels = max(d1, d2)
    else:
        # Mean or median distance from each predicted point to nearest GT point
        distances = cdist(pred_boundary, gt_boundary, metric='euclidean')
        min_distances = distances.min(axis=1)
        
        if metric == 'mean':
            distance_pixels = np.mean(min_distances)
        elif metric == 'median':
            distance_pixels = np.median(min_distances)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Convert to meters
    distance_meters = distance_pixels * pixel_resolution_m
    return distance_meters


def get_satellite_resolution(filename: str) -> float:
    """
    Extract pixel resolution from filename based on satellite type.
    
    Args:
        filename: Image filename containing satellite identifier
    
    Returns:
        Pixel resolution in meters
    """
    filename_upper = filename.upper()
    
    if 'S1' in filename_upper:
        return 40.0
    elif 'ERS' in filename_upper or 'ENV' in filename_upper:
        return 30.0
    else:
        # Default or raise warning
        print(f"Warning: Unknown satellite type in {filename}, assuming 30m")
        return 30.0


def evaluate_mde_with_filtering(pred_masks: np.ndarray,
                                gt_masks: np.ndarray,
                                filenames: List[str],
                                metric: str = 'mean',
                                boundary_method: str = 'contour',
                                apply_morphological: bool = True,
                                filter_operation: str = 'opening',
                                filter_iterations: int = 2,
                                verbose: bool = True) -> Tuple[List[float], List[str]]:
    """
    Evaluate MDE with optional morphological filtering.
    
    Args:
        pred_masks: (N, H, W) array of predicted masks
        gt_masks: (N, H, W) array of ground truth masks
        filenames: List of filenames
        metric: Distance metric ('mean', 'median', 'hausdorff')
        boundary_method: 'contour' or 'erosion'
        apply_morphological: Whether to apply morphological filtering
        filter_operation: 'opening', 'closing', 'erosion', 'dilation'
        filter_iterations: Number of filter iterations
        verbose: Print progress
    
    Returns:
        Tuple of (distances in meters, valid filenames)
    """
    distances = []
    valid_filenames = []
    skipped_count = 0
    
    # Select boundary extraction function with filtering
    if boundary_method == 'contour':
        def boundary_fn(mask):
            return extract_boundary_contour_v2(
                mask,
                morphological_filter=apply_morphological,
                filter_operation=filter_operation,
                filter_iterations=filter_iterations
            )
    else:
        def boundary_fn(mask):
            return extract_boundary_from_mask_v2(
                mask,
                morphological_filter=apply_morphological,
                filter_operation=filter_operation,
                filter_iterations=filter_iterations
            )
    
    if verbose:
        filter_status = "WITH" if apply_morphological else "WITHOUT"
        print(f"Processing {len(pred_masks)} patches {filter_status} morphological filtering...")
        if apply_morphological:
            print(f"  Filter: {filter_operation}, iterations: {filter_iterations}")
    
    for i in range(len(pred_masks)):
        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(pred_masks)} patches...")
        
        # Extract boundaries
        pred_boundary = boundary_fn(pred_masks[i])
        gt_boundary = boundary_fn(gt_masks[i])
        
        # Skip if either boundary doesn't exist
        if pred_boundary is None or gt_boundary is None:
            skipped_count += 1
            continue
        
        # Get resolution and calculate distance
        pixel_res = get_satellite_resolution(filenames[i])
        distance = calculate_boundary_distance(
            pred_boundary, 
            gt_boundary, 
            pixel_res, 
            metric
        )
        
        if not np.isnan(distance):
            distances.append(distance)
            valid_filenames.append(filenames[i])
    
    if verbose:
        print(f"\nCompleted: {len(distances)} valid patches, "
              f"{skipped_count} skipped (no boundary)")
    
    return distances, valid_filenames


# Keep all your existing functions (extract_metadata_from_filename, calculate_mde_with_subsets, 
# visualize_boundaries, build_model_specs, prepare_device, load_models, process_mask) unchanged...

# [Rest of your existing functions remain the same - I'm not repeating them to save space]

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extract metadata from filename."""
    basename = os.path.splitext(filename)[0]
    
    metadata = {
        'filename': filename,
        'satellite': 'unknown',
        'resolution': get_satellite_resolution(filename)
    }
    
    # Detect satellite
    filename_upper = filename.upper()
    if 'S1' in filename_upper:
        metadata['satellite'] = 'S1'
    elif 'ERS' in filename_upper:
        metadata['satellite'] = 'ERS'
    elif 'ENV' in filename_upper:
        metadata['satellite'] = 'ENV'
    
    return metadata


def calculate_mde_with_subsets(pred_masks: np.ndarray,
                               gt_masks: np.ndarray,
                               filenames: List[str],
                               metric: str = 'mean') -> Dict[str, Dict[str, float]]:
    """Calculate MDE overall and for various subsets (satellite)."""
    # Use the new filtering-enabled function
    distances, valid_filenames = evaluate_mde_with_filtering(
        pred_masks, gt_masks, filenames, metric
    )
    
    if len(distances) == 0:
        print("Warning: No valid boundaries found")
        return {}
    
    # Extract metadata for each valid sample
    metadata_list = [extract_metadata_from_filename(fn) for fn in valid_filenames]
    
    # Store results
    results = {
        'overall': {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'median': np.median(distances),
            'n_samples': len(distances)
        }
    }
    
    # Group by satellite
    satellite_groups = defaultdict(list)
    for dist, meta in zip(distances, metadata_list):
        satellite_groups[meta['satellite']].append(dist)
    
    for satellite, dists in satellite_groups.items():
        results[f'satellite_{satellite}'] = {
            'mean': np.mean(dists),
            'std': np.std(dists),
            'median': np.median(dists),
            'n_samples': len(dists)
        }
    
    return results


def visualize_boundaries(image: np.ndarray,
                        gt_mask: np.ndarray,
                        pred_masks: Dict[str, np.ndarray],
                        filename: str,
                        save_dir: str,
                        boundary_method: str = 'contour'):
    """Visualize ground truth and predicted boundaries for multiple models."""
    # Use the enhanced boundary extraction with filtering
    if boundary_method == 'contour':
        boundary_fn = lambda mask: extract_boundary_contour_v2(mask, morphological_filter=True)
    else:
        boundary_fn = lambda mask: extract_boundary_from_mask_v2(mask, morphological_filter=True)
    
    # Extract GT boundary
    gt_boundary = boundary_fn(gt_mask)
    
    if gt_boundary is None:
        return  # Skip if no GT boundary
    
    # Setup figure
    n_models = len(pred_masks)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))
    if n_models == 0:
        axes = [axes]
    
    # Define colors
    colors = {
        'ViT_best_iou': '#FF6B6B',      # Red
        'Unet_best_iou': '#4ECDC4',     # Teal
        'DeepLabV3_best_iou': '#45B7D1', # Blue
        'FPN_best_iou': '#FFA07A',      # Orange
        'DinoV3_best_iou': '#98D8C8'    # Mint
    }
    
    # Show original image with GT boundary
    if image.ndim == 2:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].plot(gt_boundary[:, 1], gt_boundary[:, 0], 'lime', linewidth=2, label='GT')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    axes[0].legend()
    
    # Show each model's prediction
    for idx, (model_name, pred_mask) in enumerate(pred_masks.items(), 1):
        pred_boundary = boundary_fn(pred_mask)
        
        if image.ndim == 2:
            axes[idx].imshow(image, cmap='gray')
        else:
            axes[idx].imshow(image)
        
        # Plot GT and prediction
        axes[idx].plot(gt_boundary[:, 1], gt_boundary[:, 0], 'lime', linewidth=2, label='GT', alpha=0.7)
        
        if pred_boundary is not None:
            color = colors.get(model_name, 'red')
            axes[idx].plot(pred_boundary[:, 1], pred_boundary[:, 0], color, linewidth=2, 
                          label=f'{model_name.split("_")[0]}', alpha=0.9)
            
            # Calculate distance for this sample
            pixel_res = get_satellite_resolution(filename)
            distance = calculate_boundary_distance(pred_boundary, gt_boundary, pixel_res, 'mean')
            axes[idx].set_title(f'{model_name.split("_")[0]}\nMDE: {distance:.1f}m')
        else:
            axes[idx].set_title(f'{model_name.split("_")[0]}\nNo boundary detected')
        
        axes[idx].axis('off')
        axes[idx].legend()
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{os.path.splitext(filename)[0]}_boundaries.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def build_model_specs(base_path: str, ckpt_names: Dict[str, str]) -> List[ModelSpec]:
    """Build model specifications from checkpoint names."""
    specs = []
    for model_key, ckpt in ckpt_names.items():
        # Extract architecture name (before the first underscore)
        arch = model_key.split('_')[0]
        ckpt_path = os.path.join(base_path, arch, ckpt)
        specs.append(ModelSpec(arch=arch, name=model_key, ckpt_path=ckpt_path))
    return specs


def prepare_device() -> torch.device:
    """Enhanced device preparation with memory management"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        gc.collect()
    return device


def load_models(model_specs: List[ModelSpec], cfg: DictConfig, device: torch.device) -> Dict[str, torch.nn.Module]:
    """Load models with better memory management"""
    models = {}
    
    for spec in model_specs:
        try:
            print(f"Loading model: {spec.name}")
            
            # Create a copy of config and update model name
            cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            cfg_copy.model.name = spec.arch
            
            # Load model to CPU first
            model = load_model(cfg_copy, torch.device('cpu'))
            
            # Load checkpoint
            ckpt = torch.load(spec.ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            
            # Keep model on CPU for now
            models[spec.name] = model
            
            # Clear checkpoint from memory
            del ckpt
            gc.collect()
            
            print(f"✓ {spec.name} loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading model {spec.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    return models


def process_mask(masks: torch.Tensor) -> torch.Tensor:
    """
    Matches your evaluation preprocessing:
    - Normalize [0,255] -> [0,1] if needed
    - Squeeze 1-channel mask to HxW
    - Invert (1 - mask)
    - Cast to long
    """
    if masks.max() > 1:
        masks = masks / 255.0
    if masks.dim() == 4 and masks.size(1) == 1:
        masks = masks.squeeze(1)
    masks = 1 - masks
    return masks.long()


def run_multi_model_mde_evaluation_streaming(models: Dict[str, torch.nn.Module],
                                            test_loader: DataLoader,
                                            device: torch.device,
                                            output_dir: str = './results',
                                            visualize_samples: int = 20,
                                            save_per_patch: bool = True,
                                            apply_morphological_filter: bool = True,
                                            filter_operation: str = 'opening',
                                            filter_iterations: int = 2) -> Dict[str, Dict]:
    """
    Single-pass streaming evaluation with morphological filtering options.
    """
    all_results = {}
    
    # Pre-extract all filenames from dataset to avoid DataLoader issues
    print("Extracting filenames from dataset...")
    all_filenames = []
    if hasattr(test_loader.dataset, 'image_files'):
        all_filenames = test_loader.dataset.image_files.copy()
    elif hasattr(test_loader.dataset, 'image_paths'):
        all_filenames = [os.path.basename(path) for path in test_loader.dataset.image_paths]
    else:
        all_filenames = [f"sample_{i}" for i in range(len(test_loader.dataset))]
    
    print(f"Found {len(all_filenames)} files in dataset")
    print(f"Morphological filtering: {'ENABLED' if apply_morphological_filter else 'DISABLED'}")
    if apply_morphological_filter:
        print(f"  Operation: {filter_operation}, iterations: {filter_iterations}")
    
    # Process each model sequentially
    for model_idx, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n{'='*70}")
        print(f"Processing {model_idx}/{len(models)}: {model_name}")
        print('='*70)
        
        model = model.to(device)
        model.eval()
        
        all_distances = []
        all_valid_filenames = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(test_loader)}")
                
                # Handle batch unpacking
                if len(batch) == 3:
                    images, masks, batch_filenames = batch
                else:
                    images, masks = batch
                    # Get filenames from our pre-extracted list
                    batch_start_idx = batch_idx * test_loader.batch_size
                    batch_filenames = []
                    for i in range(len(images)):
                        idx = batch_start_idx + i
                        if idx < len(all_filenames):
                            batch_filenames.append(all_filenames[idx])
                        else:
                            batch_filenames.append(f"sample_{idx}")
                
                # Move to device and run inference
                images = images.to(device)
                outputs = model(images)
                
                # Convert predictions and GT to numpy
                pred_masks_np = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                masks_np = process_mask(masks).cpu().numpy()
                
                # Handle model output shape
                if pred_masks_np.shape[1] == 2:
                    pred_masks_np = pred_masks_np[:, 1, :, :]
                elif pred_masks_np.shape[1] == 1:
                    pred_masks_np = pred_masks_np.squeeze(1)
                
                # Process each sample in batch
                actual_batch_size = len(batch_filenames)
                
                for i in range(actual_batch_size):
                    # Extract single sample masks
                    if pred_masks_np.ndim == 3:
                        pred_mask = pred_masks_np[i]
                    else:
                        pred_mask = pred_masks_np
                    
                    if masks_np.ndim == 3:
                        gt_mask = masks_np[i]
                    else:
                        gt_mask = masks_np
                    
                    filename = batch_filenames[i]
                    
                    # Extract boundaries with morphological filtering
                    pred_boundary = extract_boundary_contour_v2(
                        pred_mask,
                        morphological_filter=apply_morphological_filter,
                        filter_operation=filter_operation,
                        filter_iterations=filter_iterations
                    )
                    gt_boundary = extract_boundary_contour_v2(
                        gt_mask,
                        morphological_filter=apply_morphological_filter,
                        filter_operation=filter_operation,
                        filter_iterations=filter_iterations
                    )
                    
                    if pred_boundary is None or gt_boundary is None:
                        continue
                    
                    pixel_res = get_satellite_resolution(filename)
                    distance = calculate_boundary_distance(
                        pred_boundary, gt_boundary, pixel_res, 'mean'
                    )
                    
                    if not np.isnan(distance):
                        all_distances.append(distance)
                        all_valid_filenames.append(filename)
                
                # Clear GPU memory
                del outputs, pred_masks_np, masks_np, images
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Move model back to CPU
        model = model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
        
        # Calculate results from collected distances
        if len(all_distances) > 0:
            results = {
                'overall': {
                    'mean': float(np.mean(all_distances)),
                    'std': float(np.std(all_distances)),
                    'median': float(np.median(all_distances)),
                    'n_samples': len(all_distances)
                }
            }
            
            all_results[model_name] = results
            
            # Print results
            print(f"\nResults for {model_name}:")
            print(f"  Mean Distance: {results['overall']['mean']:.2f} m")
            print(f"  Std Dev: {results['overall']['std']:.2f} m")
            print(f"  Median: {results['overall']['median']:.2f} m")
            print(f"  N Patches: {results['overall']['n_samples']}")
            
            # Save results
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            np.savetxt(
                os.path.join(model_dir, 'distance_errors.txt'),
                all_distances,
                header=f'MDE distances in meters (n={len(all_distances)} patches)'
            )
            
            # Save detailed results with filenames
            with open(os.path.join(model_dir, 'detailed_results.csv'), 'w') as f:
                f.write("filename,distance_m,satellite_type\n")
                for fname, dist in zip(all_valid_filenames, all_distances):
                    sat_type = 'ENV' if 'ENV' in fname.upper() else 'ERS' if 'ERS' in fname.upper() else 'S1' if 'S1' in fname.upper() else 'unknown'
                    f.write(f"{fname},{dist:.3f},{sat_type}\n")
            
            with open(os.path.join(model_dir, 'mde_summary.txt'), 'w') as f:
                f.write(f"MDE Evaluation Summary for {model_name}\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Morphological filtering: {'ENABLED' if apply_morphological_filter else 'DISABLED'}\n")
                if apply_morphological_filter:
                    f.write(f"Filter operation: {filter_operation}\n")
                    f.write(f"Filter iterations: {filter_iterations}\n")
                f.write(f"\nResults:\n")
                f.write(f"overall:\n")
                for metric_name, value in results['overall'].items():
                    if metric_name == 'n_samples':
                        f.write(f"  {metric_name}: {value}\n")
                    else:
                        f.write(f"  {metric_name}: {value:.3f}\n")
        
        print(f"✓ {model_name} processing completed")
    
    # Save comparison summary
    print(f"\n{'='*70}")
    print("SAVING COMPARISON SUMMARY")
    print('='*70)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'model_comparison.txt'), 'w') as f:
        f.write("Model Comparison Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Morphological filtering: {'ENABLED' if apply_morphological_filter else 'DISABLED'}\n")
        if apply_morphological_filter:
            f.write(f"Filter operation: {filter_operation}\n")
            f.write(f"Filter iterations: {filter_iterations}\n")
        f.write("\n")
        
        f.write(f"{'Model':<20} {'Mean MDE (m)':<15} {'Std Dev (m)':<15} {'Median (m)':<15} {'N Patches':<10}\n")
        f.write("-" * 75 + "\n")
        
        for model_name, results in all_results.items():
            if 'overall' in results:
                f.write(f"{model_name:<20} {results['overall']['mean']:<15.2f} "
                       f"{results['overall']['std']:<15.2f} {results['overall']['median']:<15.2f} "
                       f"{results['overall']['n_samples']:<10}\n")
    
    gc.collect()
    print(f"\n✓ All results saved to {output_dir}")
    return all_results


# Main section - same as before but with morphological filtering options
if __name__ == "__main__":
    from data_processing.ice_data import IceDataset
    
    # Configuration
    parent_dir = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/ICE-BENCH"
    checkpoint_base = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/model_outputs"
    batch_size = 8
    device = prepare_device()
    
    cfg = OmegaConf.create({
        'model': {
            'name': 'Unet',  # This will be overridden for each model
            'encoder_name': 'resnet50',
            'encoder_weights': 'imagenet',
            'in_channels': 1,
            'classes': 2,
            'img_size': 256,
            'pretrained_path': '/home/users/amorgan/benchmark_CB_AM/models/ViT-L_16.npz',
            'satellite_weights_path': '/home/users/amorgan/benchmark_CB_AM/models/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
            'segmentation_head': 'unet',
            'freeze_backbone': True
        }
    })
    
    # Define model checkpoints
    iou_best_models = {
        "ViT_best_iou": "ViT_bs32_correct_labels_best_iou.pth",
        "Unet_best_iou": "Unet_bs32_correct_labels_latest_epoch.pth",
        "DeepLabV3_best_iou": "DeepLabV3_bs32_correct_labels_latest_epoch.pth",
        "FPN_best_iou": "FPN_bs32_correct_labels_best_iou.pth",
        "DinoV3_best_iou": "DinoV3_bs32_correct_labels_best_iou.pth",
    }
    
    # Build model specs
    model_specs = build_model_specs(checkpoint_base, iou_best_models)
    
    # Load models
    models = load_models(model_specs, cfg, device)
    
    # Create test dataloader
    test_datasets = IceDataset.create_test_datasets(parent_dir)
    test_dataset = list(test_datasets.values())[0]
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    # Use memory-efficient sequential evaluation with morphological filtering
    results = run_multi_model_mde_evaluation_streaming(
        models, 
        test_loader, 
        device,
        output_dir='./mde_results',
        visualize_samples=20,
        apply_morphological_filter=True,    # Enable morphological filtering
        filter_operation='opening',         # Use opening (removes noise)
        filter_iterations=2                 # Number of iterations
    )
    
    print("\n" + "="*70)
    print("MDE EVALUATION COMPLETED")
    print("="*70)
    print("Results saved to ./mde_results/")
    
    # Print quick summary
    for model_name, model_results in results.items():
        if 'overall' in model_results:
            print(f"{model_name}: {model_results['overall']['mean']:.2f}m ± {model_results['overall']['std']:.2f}m")