"""
Code to calculate the MDE of ground truth labels with predicted fronts

We use our segmentation masks to derive the front line.

Code inspired by Gourmelen et al. (2022)
    
"""

#TODO: CHECK THROUGH ALL OF THIS CODE AND UNDERSTAND

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict, Optional
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial.distance import directed_hausdorff
import cv2
from collections import defaultdict


def extract_boundary_from_mask(mask: np.ndarray, boundary_width: int = 1) -> Optional[np.ndarray]:
    """
    Extract the ice-ocean boundary from a binary segmentation mask.
    
    Args:
        mask: Binary mask where 1=ice, 0=ocean
        boundary_width: Width of the boundary in pixels
    
    Returns:
        Nx2 array of boundary pixel coordinates, or None if no boundary exists
    """
    # Ensure binary mask
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Check if there's both ice and ocean (boundary exists)
    if binary_mask.sum() == 0 or binary_mask.sum() == binary_mask.size:
        return None  # No boundary - all ice or all ocean
    
    # Method 1: Erosion-based boundary detection - inner boundary
    eroded = binary_erosion(binary_mask, iterations=boundary_width)
    boundary = eroded - binary_mask
    
    # # Method 2: Dilation-based boundary detection - outer boundary
    # dilated = binary_dilation(binary_mask, iterations=boundary_width)
    # boundary = dilated - binary_mask
    
    # Method 3: symmetric boundary (both sides)
    # eroded = binary_erosion(binary_mask, iterations=boundary_width)
    # dilated = binary_dilation(binary_mask, iterations=boundary_width)
    # boundary = dilated - eroded

    # Get coordinates of boundary pixels
    boundary_coords = np.argwhere(boundary > 0)
    
    if len(boundary_coords) == 0:
        return None
    
    return boundary_coords


def extract_boundary_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract boundary using contour detection (alternative method).
    Often gives cleaner single-pixel boundaries.
    
    Returns:
        Nx2 array of boundary coordinates
    """
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Check if boundary exists
    if binary_mask.sum() == 0 or binary_mask.sum() == binary_mask.size:
        return None
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return None
    
    # Use the longest contour (main ice front)
    longest_contour = max(contours, key=len)
    
    # Reshape from (N, 1, 2) to (N, 2) and swap x,y to row,col
    boundary_coords = longest_contour.squeeze()
    if boundary_coords.ndim == 1:  # Handle single point case
        boundary_coords = boundary_coords.reshape(1, -1)
    
    # Swap columns: cv2 returns (x,y) but we want (row,col)
    boundary_coords = boundary_coords[:, [1, 0]]
    
    return boundary_coords


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
    elif 'ERS' in filename_upper or 'ENV' in filename_upper or 'ENV' in filename_upper:
        return 30.0
    else:
        # Default or raise warning
        print(f"Warning: Unknown satellite type in {filename}, assuming 30m")
        return 30.0


def evaluate_mde_from_masks(pred_masks: np.ndarray,
                            gt_masks: np.ndarray,
                            filenames: List[str],
                            metric: str = 'mean',
                            boundary_method: str = 'contour') -> Tuple[List[float], List[str]]:
    """
    Evaluate MDE for a batch of predictions.
    
    Args:
        pred_masks: (N, H, W) array of predicted masks
        gt_masks: (N, H, W) array of ground truth masks
        filenames: List of filenames for each sample
        metric: Distance metric to use
        boundary_method: 'contour' or 'erosion'
    
    Returns:
        Tuple of (distances in meters, valid filenames)
    """
    distances = []
    valid_filenames = []
    
    boundary_fn = extract_boundary_contour if boundary_method == 'contour' else extract_boundary_from_mask
    
    for i in range(len(pred_masks)):
        # Extract boundaries
        pred_boundary = boundary_fn(pred_masks[i])
        gt_boundary = boundary_fn(gt_masks[i])
        
        # Skip if either boundary doesn't exist
        if pred_boundary is None or gt_boundary is None:
            continue
        
        # Get resolution from filename
        pixel_res = get_satellite_resolution(filenames[i])
        
        # Calculate distance
        distance = calculate_boundary_distance(pred_boundary, gt_boundary, pixel_res, metric)
        
        if not np.isnan(distance):
            distances.append(distance)
            valid_filenames.append(filenames[i])
    
    return distances, valid_filenames


def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract metadata from filename.

    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    
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
    elif 'ENV' in filename_upper or 'ENV' in filename_upper:
        metadata['satellite'] = 'ENV'
    
    return metadata


def calculate_mde_with_subsets(pred_masks: np.ndarray,
                               gt_masks: np.ndarray,
                               filenames: List[str],
                               metric: str = 'mean') -> Dict[str, Dict[str, float]]:
    """
    Calculate MDE overall and for various subsets (satellite, glacier, etc.).
    
    Returns:
        Dictionary with results for each subset
    """
    # Calculate distances for all samples
    distances, valid_filenames = evaluate_mde_from_masks(
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
    
    # # Group by glacier (if available) - TODO: try to do this
    # glacier_groups = defaultdict(list)
    # for dist, meta in zip(distances, metadata_list):
    #     glacier_groups[meta['glacier']].append(dist)
    
    # for glacier, dists in glacier_groups.items():
    #     if glacier != 'unknown':
    #         results[f'glacier_{glacier}'] = {
    #             'mean': np.mean(dists),
    #             'std': np.std(dists),
    #             'median': np.median(dists),
    #             'n_samples': len(dists)
    #         }
    
    return results


def make_test_dataset_and_loader_with_filenames(parent_dir: str, 
                                                 batch_size: int) -> Tuple[torch.utils.data.Dataset, DataLoader]:
    """
    Extended version that ensures dataset returns filenames.

    """
    from data_processing.ice_data import IceDataset  
    
    test_datasets = IceDataset.create_test_datasets(parent_dir)
    test_dataset = list(test_datasets.values())[0]
    
    # Ensure the dataset returns filenames
    # You might need to add a get_filename() method to your dataset
    loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    return test_dataset, loader


def run_mde_evaluation(model: torch.nn.Module,
                      test_loader: DataLoader,
                      device: str = 'cuda',
                      output_dir: str = './results') -> Dict:
    """
    Full evaluation pipeline.
    
    Args:
        model: Trained segmentation model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        output_dir: Directory to save results
    
    Returns:
        Dictionary with all results
    """
    model.eval()
    
    all_pred_masks = []
    all_gt_masks = []
    all_filenames = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            # Adjust based on your dataset's return format
            if len(batch) == 3:
                images, masks, filenames = batch
            else:
                images, masks = batch
                filenames = [f"sample_{i}" for i in range(len(images))]
            
            images = images.to(device)
            outputs = model(images)
            
            # Convert to binary masks
            pred_masks = (torch.sigmoid(outputs) > 0.5).cpu().numpy().squeeze()
            gt_masks = masks.cpu().numpy().squeeze()
            
            all_pred_masks.append(pred_masks)
            all_gt_masks.append(gt_masks)
            all_filenames.extend(filenames)
    
    # Concatenate all batches
    all_pred_masks = np.concatenate(all_pred_masks, axis=0)
    all_gt_masks = np.concatenate(all_gt_masks, axis=0)
    
    print(f"\nCalculating MDE for {len(all_filenames)} samples...")
    
    # Calculate MDE with subsets
    results = calculate_mde_with_subsets(
        all_pred_masks,
        all_gt_masks,
        all_filenames,
        metric='mean'
    )
    
    # Print results
    print("\n" + "="*70)
    print("MDE EVALUATION RESULTS")
    print("="*70)
    
    for subset_name, metrics in results.items():
        print(f"\n{subset_name}:")
        print(f"  Mean Distance: {metrics['mean']:.2f} m")
        print(f"  Std Dev: {metrics['std']:.2f} m")
        print(f"  Median: {metrics['median']:.2f} m")
        print(f"  N Samples: {metrics['n_samples']}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed distances
    distances, valid_filenames = evaluate_mde_from_masks(
        all_pred_masks, all_gt_masks, all_filenames
    )
    np.savetxt(
        os.path.join(output_dir, 'distance_errors.txt'),
        distances
    )
    
    # Save filenames
    with open(os.path.join(output_dir, 'evaluated_files.txt'), 'w') as f:
        f.write('\n'.join(valid_filenames))
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'mde_summary.txt'), 'w') as f:
        for subset_name, metrics in results.items():
            f.write(f"{subset_name}:\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value}\n")
            f.write("\n")
    
    return results


# Example usage
if __name__ == "__main__":
    # Setup
    parent_dir = "/path/to/data"
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    # model = YourModel()
    # model.load_state_dict(torch.load('best_model.pth'))
    # model = model.to(device)
    
    # Create dataloader
    test_dataset, test_loader = make_test_dataset_and_loader_with_filenames(
        parent_dir, 
        batch_size
    )
    
    # Run evaluation
    # results = run_mde_evaluation(model, test_loader, device)