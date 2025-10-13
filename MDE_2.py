"""
Code to calculate the MDE of ground truth labels with predicted fronts

We use our segmentation masks to derive the front line.

Code inspired by Gourmelen et al. (2022)

Updated MDE evaluation that integrates all fixes:
1. Background detection and exclusion
2. Edge boundary exclusion
3. Straight edge detection
4. All models displayed in visualizations
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import gc
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from scipy.spatial.distance import directed_hausdorff, cdist
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
from omegaconf import DictConfig, OmegaConf
from load_functions import load_model
import json

@dataclass
class ModelSpec:
    arch: str
    name: str
    ckpt_path: str

def load_background_filters(base_dir: str) -> Dict[str, bool]:
    """
    Load all background JSON files and combine them into a single filter dictionary.
    
    Args:
        base_dir: Base directory containing background_info folder
    
    Returns:
        Dictionary mapping absolute file paths to background status (True/False)
    """
    background_info_dir = os.path.join(base_dir, "background_scenes", "background_info")
    
    json_files = [
        "Envisat_backgrounds.json",
        "ERS_backgrounds.json", 
        "Sentinel-1_backgrounds.json"
    ]
    
    combined_filters = {}
    
    for json_file in json_files:
        json_path = os.path.join(background_info_dir, json_file)
        
        if os.path.exists(json_path):
            print(f"Loading background filter: {json_file}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                combined_filters.update(data)
                
                # Count true/false
                true_count = sum(1 for v in data.values() if v)
                false_count = sum(1 for v in data.values() if not v)
                print(f"  {json_file}: {true_count} background, {false_count} valid")
        else:
            print(f"Warning: {json_path} not found")
    
    total_true = sum(1 for v in combined_filters.values() if v)
    total_false = sum(1 for v in combined_filters.values() if not v)
    print(f"\nTotal: {total_true} background (excluded), {total_false} valid (included)")
    
    return combined_filters


def get_valid_file_indices(dataset, background_filters: Dict[str, bool]) -> List[int]:
    """
    Get indices of valid files (marked as false in background JSON).
    Uses flexible path matching to handle different path formats.
    
    Args:
        dataset: IceDataset instance
        background_filters: Dictionary from load_background_filters
    
    Returns:
        List of valid dataset indices
    """
    valid_indices = []
    skipped_count = 0
    not_found_count = 0
    
    # Get all image files from dataset
    if hasattr(dataset, 'image_files'):
        image_files = dataset.image_files
    elif hasattr(dataset, 'image_paths'):
        image_files = dataset.image_paths
    else:
        print("Warning: Could not find image files in dataset")
        return []
    
    # Create a mapping of basenames to background status for efficient lookup
    basename_to_status = {}
    filter_path_examples = []
    
    for filter_path, is_background in background_filters.items():
        basename = os.path.basename(filter_path)
        basename_to_status[basename] = is_background
        if len(filter_path_examples) < 3:
            filter_path_examples.append(filter_path)
    
    print(f"\nDebug - Example filter paths:")
    for example in filter_path_examples:
        print(f"  {example}")
    
    print(f"\nDebug - Example dataset paths:")
    for i, path in enumerate(image_files[:3]):
        print(f"  {path}")
    
    # Try multiple matching strategies
    for idx, img_path in enumerate(image_files):
        matched = False
        match_method = None
        
        # Strategy 1: Direct path match (absolute)
        abs_path = os.path.abspath(img_path) if not os.path.isabs(img_path) else img_path
        if abs_path in background_filters:
            matched = True
            match_method = "absolute"
            is_background = background_filters[abs_path]
        
        # Strategy 2: Basename match
        elif not matched:
            basename = os.path.basename(img_path)
            if basename in basename_to_status:
                matched = True
                match_method = "basename"
                is_background = basename_to_status[basename]
        
        # Strategy 3: Try to construct expected path
        elif not matched:
            # Extract just the filename and try to match with filter paths
            filename = os.path.basename(img_path)
            for filter_path in background_filters.keys():
                if filter_path.endswith(filename):
                    matched = True
                    match_method = "filename_suffix"
                    is_background = background_filters[filter_path]
                    break
        
        if matched:
            if not is_background:
                valid_indices.append(idx)
                if len(valid_indices) <= 3:
                    print(f"  ✓ Match {len(valid_indices)}: {os.path.basename(img_path)} (method: {match_method})")
            else:
                skipped_count += 1
        else:
            not_found_count += 1
            if not_found_count <= 3:
                print(f"  ✗ No match: {img_path}")
    
    print(f"\nDataset filtering results:")
    print(f"  Total files: {len(image_files)}")
    print(f"  Valid (false): {len(valid_indices)}")
    print(f"  Skipped (true): {skipped_count}")
    print(f"  Not in filter: {not_found_count}")
    
    return valid_indices


def create_filtered_dataloader(dataset, valid_indices: List[int], batch_size: int = 8) -> DataLoader:
    """
    Create a DataLoader that only includes valid indices.
    
    Args:
        dataset: Original dataset
        valid_indices: List of valid indices from get_valid_file_indices
        batch_size: Batch size for DataLoader
    
    Returns:
        Filtered DataLoader
    """
    from torch.utils.data import Subset, DataLoader
    
    # Create a subset with only valid indices
    filtered_dataset = Subset(dataset, valid_indices)
    
    filtered_loader = DataLoader(
        filtered_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Created filtered DataLoader with {len(filtered_dataset)} samples")
    return filtered_loader

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
    

    mask = mask.squeeze()
    
    if mask.ndim > 2:
        # Common cases:
        # (batch=1, H, W) -> take [0]
        # (H, W, channels=1) -> take [..., 0]
        # (batch, channels, H, W) -> take [0, 0]
        
        if mask.ndim == 3:
            # 3D case - could be (B, H, W) or (H, W, C)
            if mask.shape[0] <= 4:  
                mask = mask[0]
            elif mask.shape[2] <= 4:  
                mask = mask[..., 0]
            else:
                mask = mask[0]
        elif mask.ndim == 4:
            # 4D case - likely (B, C, H, W)
            mask = mask[0, 0]
        else:
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


def detect_background_pixels(image: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """
    Detect background (zero/near-zero) pixels in the image.
    """
    # if image.ndim == 3:
    #     # For RGB, check if all channels are below threshold
    #     background_mask = np.all(image <= threshold, axis=-1)
    # else:
    #     # For grayscale, direct comparison
    #     background_mask = image <= threshold
    
    # # ✅ CRITICAL FIX: Dilate background regions to catch edge artifacts
    # from scipy.ndimage import binary_dilation
    # background_mask = binary_dilation(background_mask, iterations=5)
    
    # return background_mask
    return np.zeros_like(image, dtype=bool)


def find_ice_ocean_boundary_lines_only(mask: np.ndarray, image: np.ndarray, 
                                       background_threshold: float = 1e-6) -> Optional[np.ndarray]:

    if mask.dtype in [np.float32, np.float64]:
        mask = (mask > 0.5).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # if image is not None:
    #     background_mask = detect_background_pixels(image, background_threshold)
    # else:
    #     background_mask = np.zeros_like(mask, dtype=bool)
    
    # # Create ocean mask: non-ice pixels that are NOT background
    # ocean_mask = (mask == 0) & (~background_mask)
    # ice_mask = (mask == 1).astype(np.uint8)
    
    # if not np.any(ocean_mask):
    #     print("No valid ocean pixels found (all non-ice is background)")
    #     return None
    
    ocean_mask = (mask == 0)
    ice_mask = (mask == 1).astype(np.uint8)
    
    if not np.any(ocean_mask):
        print("No ocean pixels found")
        return None
    
    ice_mask_255 = (ice_mask * 255).astype(np.uint8)
    
    try:
        # Find contours
        contours, _ = cv2.findContours(ice_mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            return None
        
        # Get the longest contour (main ice boundary)
        longest_contour = max(contours, key=len)
        
        if len(longest_contour) < 10:
            print(f"Boundary too short: {len(longest_contour)} pixels")
            return None
        
        # Convert contour to coordinates
        boundary_coords = longest_contour.squeeze()
        
        if boundary_coords.ndim == 1:
            boundary_coords = boundary_coords.reshape(1, -1)
        
        # Convert from (x, y) to (row, col) format
        boundary_coords = boundary_coords[:, [1, 0]].astype(np.float32)
        valid_boundary_points = []
        
        
        for point in boundary_coords:
            row, col = int(point[0]), int(point[1])
            
            # Check smaller neighborhood for ocean
            found_ocean = False
            for dr in range(-1, 2):  # 3x3 instead of 5x5
                for dc in range(-1, 2):
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < ocean_mask.shape[0] and 
                        0 <= nc < ocean_mask.shape[1] and 
                        ocean_mask[nr, nc]):
                        found_ocean = True
                        break
                if found_ocean:
                    break
            
            if found_ocean:
                valid_boundary_points.append(point)
        
        # ✅ MORE LENIENT: Require fewer valid points for filtered data
        if len(valid_boundary_points) < 5:  # Reduced from 10
            print("No valid ice-ocean boundary found")
            return None
        
        return np.array(valid_boundary_points, dtype=np.float32)
        
    except cv2.error as e:
        print(f"OpenCV error in boundary extraction: {e}")
        return None

def detect_background_boundary(image: np.ndarray, mask: np.ndarray, threshold: float = 1e-6) -> bool:

    """

    Disable as dataset already filtered out background
    """
    
    # if image is None:
    #     return False

    # if image.ndim == 3:
    #     background_mask = np.all(np.abs(image) <= threshold, axis=-1)
    # else:
    #     background_mask = np.abs(image) <= threshold
  

    # mask_binary = (mask > 0).astype(np.uint8)
    # edges = ndimage.sobel(mask_binary)
    # edge_pixels = edges > 0
    
    # if not np.any(edge_pixels):
    #     return False
    
    # # Check if boundary pixels coincide with background
    # boundary_on_background = edge_pixels & background_mask
    # overlap_ratio = np.sum(boundary_on_background) / np.sum(edge_pixels)
    
    # If >30% of boundary is on background pixels, it's likely a satellite edge
    # return overlap_ratio > 0.3
    return False



def is_boundary_straight_line(boundary: np.ndarray, straightness_threshold: float = 0.95) -> bool:

    if boundary is None or len(boundary) < 10:
        return False
    
    try:
        # Fit line using least squares
        coords = boundary.astype(np.float64)
        mean_coord = np.mean(coords, axis=0)
        coords_centered = coords - mean_coord
        
        # SVD to find principal direction
        U, s, Vt = np.linalg.svd(coords_centered, full_matrices=False)
        
        # Ratio of singular values indicates linearity
        if len(s) > 1 and s[0] > 0:
            linearity = 1 - (s[1] / s[0])
            return linearity > straightness_threshold       
    except:
        pass
    
    rows, cols = boundary[:, 0], boundary[:, 1]
    if np.std(rows) < 3.0 or np.std(cols) < 3.0:
        return True
    
    return False


def is_boundary_on_patch_edge(boundary: np.ndarray, image_shape: tuple, 
                              edge_threshold: int = 8) -> bool:
    if boundary is None or len(boundary) == 0:
        return False
    
    height, width = image_shape
    rows, cols = boundary[:, 0], boundary[:, 1]
    
    # Count points near each edge
    near_top = np.sum(rows <= edge_threshold)
    near_bottom = np.sum(rows >= height - edge_threshold - 1)
    near_left = np.sum(cols <= edge_threshold)
    near_right = np.sum(cols >= width - edge_threshold - 1)
    
    total_points = len(boundary)
    
    # If >30% of points are near any edge, it's likely a patch boundary
    edge_ratios = [
        near_top / total_points,
        near_bottom / total_points,
        near_left / total_points,
        near_right / total_points
    ]
    return max(edge_ratios) > 0.3

def extract_ice_front_boundary_fixed(mask: np.ndarray, 
                                     image: np.ndarray,
                                     background_threshold: float = 1e-6,
                                     min_length: int = 50) -> Optional[np.ndarray]:
    """
    Extract ONLY the ice-ocean boundary line (single line, no double polygons).

    Args:
        mask: Binary mask where 1=ice, 0=ocean/background
        image: Unused (kept for API compatibility)
        background_threshold: Unused (kept for compatibility)
        min_length: Minimum boundary length in pixels

    Returns:
        Nx2 array of boundary line coordinates (row, col) or None
    """
    # --- 1. Threshold to binary ---
    if mask.dtype in [np.float32, np.float64]:
        mask_binary = (mask > 0.5).astype(np.uint8)
    else:
        mask_binary = (mask > 0).astype(np.uint8)

    if np.unique(mask_binary).size < 2:
        return None  # no clear separation

    # --- 2. Compute a thin boundary (no dilation to avoid double lines) ---
    # Morphological gradient = outer edge of ice region
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_binary, kernel, iterations=1)
    boundary_mask = cv2.subtract(mask_binary, eroded)

    if not np.any(boundary_mask):
        return None

    # --- 3. Find contours on this 1-px edge ---
    contours, _ = cv2.findContours(
        boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        return None

    # --- 4. Keep only meaningful, long contours ---
    valid_contours = [c for c in contours if len(c) >= min_length]
    if len(valid_contours) == 0:
        return None

    # --- 5. Choose the longest contour (main ice front) ---
    longest = max(valid_contours, key=len)
    coords = longest.squeeze()
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    # Convert from (x, y) → (row, col)
    boundary_line = coords[:, [1, 0]].astype(np.float32)

    # Remove duplicate closing point if contour is closed
    if np.allclose(boundary_line[0], boundary_line[-1]):
        boundary_line = boundary_line[:-1]

    return boundary_line



def extract_boundary_contour_v2(mask: np.ndarray,
                                image: Optional[np.ndarray] = None,
                                background_threshold: float = 1e-6,
                                morphological_iterations: int = 2,
                                min_contour_length: int = 50) -> Optional[np.ndarray]:
    """
    Alternative approach using contour finding on the ice-ocean interface only.
    """

    # Ensure binary
    if mask.dtype in [np.float32, np.float64]:
        mask_binary = (mask > 0.5).astype(np.uint8)
    else:
        mask_binary = mask.astype(np.uint8)
    
    # # Detect background
    # if image is not None:
    #     background_mask = detect_background_pixels(image, background_threshold)
    # else:
    #     background_mask = np.zeros_like(mask_binary, dtype=bool)
    
    # # Create valid ocean mask (non-ice AND non-background)
    # ocean_mask = (mask_binary == 0) & (~background_mask)
    
    # if not np.any(ocean_mask):
    #     return None
    ocean_mask = (mask_binary == 0)

    
    if morphological_iterations > 0:
        mask_binary = binary_opening(mask_binary, iterations=morphological_iterations).astype(np.uint8)
    
    # Dilate ocean slightly to ensure adjacency detection
    ocean_dilated = binary_dilation(ocean_mask, iterations=2).astype(np.uint8)
    
    # Find where ice meets dilated ocean
    ice_ocean_interface = mask_binary & ocean_dilated
    
    if not np.any(ice_ocean_interface):
        return None
    
    # Find contours on this interface
    ice_ocean_interface = (ice_ocean_interface * 255).astype(np.uint8)
    contours, _ = cv2.findContours(ice_ocean_interface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return None
    
    # Get longest contour
    valid_contours = [c for c in contours if len(c) >= min_contour_length]
    
    if len(valid_contours) == 0:
        return None
    
    longest = max(valid_contours, key=len)
    boundary = longest.squeeze()
    
    if boundary.ndim == 1:
        boundary = boundary.reshape(1, -1)
    
    # Convert from (x, y) to (row, col)
    boundary = boundary[:, [1, 0]].astype(np.float32)
    
    # Validation checks
    if is_boundary_straight_line(boundary):
        return None
    
    if is_boundary_on_patch_edge(boundary, mask.shape):
        return None
    
    return boundary



def calculate_boundary_distance_safe(pred_boundary: np.ndarray, 
                                     gt_boundary: np.ndarray,
                                     pred_image: Optional[np.ndarray],
                                     gt_image: Optional[np.ndarray],
                                     pixel_resolution_m: float,
                                     metric: str = 'mean') -> float:
    """
    Calculate distance 
    Args:
        pred_boundary: Predicted boundary coordinates
        gt_boundary: Ground truth boundary coordinates
        pred_image: Predicted image for background check
        gt_image: Ground truth image for background check
        pixel_resolution_m: Resolution in meters per pixel
        metric: Distance metric
    
    Returns:
        Distance in meters or np.nan if invalid
    """
    # Check boundaries exist
    if pred_boundary is None or gt_boundary is None:
        return np.nan
    
    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return np.nan
     # Check for straight edges
    if is_boundary_straight_line(pred_boundary) or is_boundary_straight_line(gt_boundary):
        return np.nan
    
    # Calculate distance
    if metric == 'hausdorff':
        d1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
        d2 = directed_hausdorff(gt_boundary, pred_boundary)[0]
        distance_pixels = max(d1, d2)
    else:
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
    filename_upper = filename.upper()
    if 'S1' in filename_upper:
        return 40.0
    elif 'ERS' in filename_upper or 'ENV' in filename_upper:
        return 30.0
    else:
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
    distances, valid_filenames = evaluate_mde_with_filtering(
        pred_masks, gt_masks, filenames, metric
    )
    if len(distances) == 0:
        print("Warning: No valid boundaries found")
        return {}
    
    metadata_list = [extract_metadata_from_filename(fn) for fn in valid_filenames]
    
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
    model_colors: Dict[str, str] = None
) -> None:
    """
    Visualize ONLY ice front boundary lines (no filled regions).
    
    Args:
        image: Satellite image (H, W) or (H, W, 3)
        gt_mask: Ground truth mask
        pred_masks: Dictionary of {model_name: prediction_mask}
        filename: Image filename
        save_dir: Directory to save visualization
        model_colors: Dictionary of model colors
    """
    # Default colors
    if model_colors is None:
        model_colors = {
            'ViT_best_iou': '#FF6B6B',
            'Unet_best_iou': "#984ECD",
            'DeepLabV3_best_iou': "#DFD87B",
            'FPN_best_iou': "#26C663",
            'DinoV3_best_iou': "#46C1EE"
        }
    
    # Extract GT boundary LINE
    gt_boundary = extract_ice_front_boundary_fixed(
        gt_mask, 
        image, 
        background_threshold=1e-6,
        min_length=50
    )
    
    if gt_boundary is None:
        print(f"Skipping {filename}: No valid GT ice-ocean boundary")
        return
    
    # Extract predicted boundaries
    pred_boundaries = {}
    
    print(f"\n=== {filename} ===")
    print(f"GT boundary: {len(gt_boundary)} points")
    
    for model_name, pred_mask in pred_masks.items():
        pred_boundary = extract_ice_front_boundary_fixed(
            pred_mask,
            image,
            background_threshold=1e-6,
            min_length=50
        )
        
        if pred_boundary is not None:
            pred_boundaries[model_name] = pred_boundary
            print(f"  {model_name}: ✓ {len(pred_boundary)} points")
        else:
            print(f"  {model_name}: ✗ No valid boundary")
    
    if len(pred_boundaries) == 0:
        print(f"Skipping {filename}: No valid predicted boundaries")
        return
    
    # Create visualization
    n_models = len(pred_boundaries)
    fig = plt.figure(figsize=(5 * (n_models + 2), 5))
    
    # Panel 1: Raw satellite image ONLY
    ax1 = plt.subplot(1, n_models + 2, 1)
    if image.ndim == 2:
        ax1.imshow(image, cmap='gray', vmin=0, vmax=1)
    else:
        ax1.imshow(image)
    ax1.set_title('Raw Satellite Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Image + GT boundary LINE
    ax2 = plt.subplot(1, n_models + 2, 2)
    if image.ndim == 2:
        ax2.imshow(image, cmap='gray', vmin=0, vmax=1)
    else:
        ax2.imshow(image)
    
    # Plot GT boundary as LINE (not filled)
    ax2.plot(gt_boundary[:, 1], gt_boundary[:, 0],
             color='lime', linewidth=2.5, alpha=0.9, label='GT Ice Front')
    
    ax2.set_title('Ground Truth\nIce-Ocean Boundary', fontsize=12, fontweight='bold')
    ax2.axis('off')
    ax2.legend(loc='upper right', fontsize=9)
    
    # Panels 3+: Individual model predictions
    for idx, (model_name, pred_boundary) in enumerate(pred_boundaries.items(), 3):
        ax = plt.subplot(1, n_models + 2, idx)
        
        # Background image
        if image.ndim == 2:
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(image)
        
        # GT boundary (faded)
        ax.plot(gt_boundary[:, 1], gt_boundary[:, 0],
                color='lime', linewidth=1.5, alpha=0.4, linestyle='--', label='GT')
        
        # Predicted boundary LINE
        color = model_colors.get(model_name, '#FF6B6B')
        ax.plot(pred_boundary[:, 1], pred_boundary[:, 0],
                color=color, linewidth=2.5, alpha=0.9, label='Prediction')
        
        # Calculate MDE
        from scipy.spatial.distance import cdist
        distances = cdist(pred_boundary, gt_boundary, metric='euclidean')
        mean_dist_px = np.mean(distances.min(axis=1))
        
        # Get resolution
        pixel_res = get_satellite_resolution(filename)
        mde_m = mean_dist_px * pixel_res
        
        # Title
        short_name = model_name.split('_')[0]
        ax.set_title(f'{short_name}\nMDE: {mde_m:.1f}m', 
                    fontsize=11, fontweight='bold')
        ax.axis('off')
        ax.legend(loc='upper right', fontsize=8)
    
    # Overall title
    fig.suptitle(f'Ice-Ocean Boundary Analysis: {filename}',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{os.path.splitext(filename)[0]}_fixed.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {save_path}")


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


# REPLACE the run_fixed_mde_evaluation function definition (around line 1223):

def run_fixed_mde_evaluation(
    models: Dict[str, torch.nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str = './mde_results_fixed',
    apply_morphological_filter: bool = True,
    filter_operation: str = 'opening',
    filter_iterations: int = 2,
    exclude_edges: bool = True,
    edge_width: int = 8,
    check_straightness: bool = True,
    original_filenames: Optional[List[str]] = None  # ✅ ADD this parameter
) -> Dict[str, Dict]:
    """
    FIXED: MDE evaluation with enhanced background and patch boundary detection.
    """
    all_results = {}
    
    # ✅ FIX: Use provided filenames if available, otherwise extract from dataset
    print("Setting up filenames for evaluation...")
    if original_filenames is not None:
        all_filenames = original_filenames.copy()
        print(f"Using provided filtered filenames: {len(all_filenames)} files")
    else:
        # Fallback to extracting from dataset
        all_filenames = []
        if hasattr(test_loader.dataset, 'image_files'):
            all_filenames = test_loader.dataset.image_files.copy()
        elif hasattr(test_loader.dataset, 'image_paths'):
            all_filenames = [os.path.basename(path) for path in test_loader.dataset.image_paths]
        else:
            all_filenames = [f"sample_{i}" for i in range(len(test_loader.dataset))]
        print(f"Extracted filenames from dataset: {len(all_filenames)} files")
    
    print(f"Example filenames: {all_filenames[:3] if len(all_filenames) > 0 else 'None'}")
    
    print(f"\nENHANCED Configuration:")
    print(f"  Background detection: ENABLED (threshold=1e-6, max_ratio=80%)")
    print(f"  Satellite edge detection: ENABLED")
    print(f"  Patch boundary detection: ENABLED")
    print(f"  Edge exclusion: {'ENABLED' if exclude_edges else 'DISABLED'}")
    if exclude_edges:
        print(f"  Edge width: {edge_width} pixels (INCREASED)")
    print(f"  Enhanced straight line detection: {'ENABLED' if check_straightness else 'DISABLED'}")
    
    # Process each model sequentially
    for model_idx, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n{'='*70}")
        print(f"Processing {model_idx}/{len(models)}: {model_name}")
        print('='*70)
        
        model = model.to(device)
        model.eval()
        
        all_distances = []
        all_valid_filenames = []
        skipped_background = 0
        skipped_satellite_edge = 0
        skipped_patch_boundary = 0
        skipped_straight_edge = 0
        skipped_no_boundary = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(test_loader)}")
                
                # ✅ FIX: Handle batch filename extraction properly
                if len(batch) == 3:
                    images, masks, batch_filenames = batch
                else:
                    images, masks = batch
                    # Calculate the correct indices for this batch
                    batch_start_idx = batch_idx * test_loader.batch_size
                    batch_filenames = []
                    for i in range(len(images)):
                        idx = batch_start_idx + i
                        if idx < len(all_filenames):
                            batch_filenames.append(all_filenames[idx])
                        else:
                            batch_filenames.append(f"sample_{idx}")
                
                # ✅ DEBUG: Print batch info occasionally
                if batch_idx < 3:  # First few batches
                    print(f"  Batch {batch_idx}: {len(batch_filenames)} files, examples: {batch_filenames[:2]}")
                
                # Move to device and run inference
                images_gpu = images.to(device)
                outputs = model(images_gpu)
                
                # Convert predictions and GT to numpy
                pred_masks_np = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                masks_np = process_mask(masks).cpu().numpy()
                images_np = images.cpu().numpy()
                
                # Handle model output shape
                if pred_masks_np.shape[1] == 2:
                    pred_masks_np = pred_masks_np[:, 1, :, :]
                elif pred_masks_np.shape[1] == 1:
                    pred_masks_np = pred_masks_np.squeeze(1)
                
                # Process each sample in batch
                actual_batch_size = len(batch_filenames)
                
                for i in range(actual_batch_size):
                    # Extract single sample
                    if pred_masks_np.ndim == 3:
                        pred_mask = pred_masks_np[i]
                    else:
                        pred_mask = pred_masks_np
                    
                    if masks_np.ndim == 3:
                        gt_mask = masks_np[i]
                    else:
                        gt_mask = masks_np
                    
                    if images_np.ndim == 4:
                        image = images_np[i]
                        # Convert from (C, H, W) to (H, W)
                        if image.shape[0] == 1:
                            image = image.squeeze(0)
                        elif image.shape[0] == 3:
                            image = np.transpose(image, (1, 2, 0))
                    else:
                        image = images_np
                    
                    filename = batch_filenames[i]
                
                    # Extract boundaries with all enhanced fixes
                    pred_boundary = extract_boundary_contour_v2(
                        pred_mask,
                        image=image,
                        background_threshold=1e-6,
                        morphological_iterations=filter_iterations,    
                        min_contour_length=50
                    )
                    
                    gt_boundary = extract_boundary_contour_v2(
                        gt_mask,
                        image=image,
                        background_threshold=1e-6,
                        morphological_iterations=filter_iterations, 
                        min_contour_length=50
                    )
                    
                    # Track specific skip reasons
                    if pred_boundary is None or gt_boundary is None:
                        # Additional debugging to understand why boundaries are None
                        # if image is not None:
                        #     if detect_background_boundary(image, pred_mask if pred_boundary is None else gt_mask):
                        #         skipped_satellite_edge += 1
                        #         continue
                        
                        if pred_boundary is None:
                            temp_boundary = extract_boundary_contour_v2(
                                pred_mask, 
                                image=None, 
                                background_threshold=1e-6, 
                                morphological_iterations=0, 
                                min_contour_length=10
                            )
                            if temp_boundary is not None:
                                if is_boundary_on_patch_edge(temp_boundary, pred_mask.shape):
                                    skipped_patch_boundary += 1
                                    continue
                                elif is_boundary_straight_line(temp_boundary):
                                    skipped_straight_edge += 1
                                    continue
                        
                        skipped_no_boundary += 1
                        continue
                    
                    # Calculate distance
                    pixel_res = get_satellite_resolution(filename)
                    distance = calculate_boundary_distance(
                        pred_boundary, gt_boundary, pixel_res, 'mean'
                    )
                    
                    if not np.isnan(distance):
                        all_distances.append(distance)
                        all_valid_filenames.append(filename)
                
                # Clear GPU memory
                del outputs, pred_masks_np, masks_np, images_np, images_gpu
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Move model back to CPU
        model = model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
        
        # ✅ ENHANCED: Check if we found any valid boundaries
        if len(all_distances) > 0:
            results = {
                'overall': {
                    'mean': float(np.mean(all_distances)),
                    'std': float(np.std(all_distances)),
                    'median': float(np.median(all_distances)),
                    'n_samples': len(all_distances),
                    'skipped_background': skipped_background,
                    'skipped_satellite_edge': skipped_satellite_edge,
                    'skipped_patch_boundary': skipped_patch_boundary,
                    'skipped_straight_edge': skipped_straight_edge,
                    'skipped_no_boundary': skipped_no_boundary
                }
            }
            
            all_results[model_name] = results
            
            # Print detailed results
            total_skipped = (skipped_background + skipped_satellite_edge + 
                           skipped_patch_boundary + skipped_straight_edge + skipped_no_boundary)
            
            print(f"\nResults for {model_name}:")
            print(f"  Mean Distance: {results['overall']['mean']:.2f} m")
            print(f"  Std Dev: {results['overall']['std']:.2f} m")
            print(f"  Median: {results['overall']['median']:.2f} m")
            print(f"  Valid Patches: {results['overall']['n_samples']}")
            print(f"  Success Rate: {(results['overall']['n_samples'] / (results['overall']['n_samples'] + total_skipped) * 100):.1f}%")
            
            # Save detailed filenames
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # ✅ NEW: Save the actual filenames used
            with open(os.path.join(model_dir, 'valid_filenames.txt'), 'w') as f:
                f.write(f"Valid filenames for {model_name} ({len(all_valid_filenames)} files):\n")
                f.write("-" * 50 + "\n")
                for filename in all_valid_filenames:
                    f.write(f"{filename}\n")
            
            print(f"  ✓ Saved {len(all_valid_filenames)} valid filenames to {model_dir}/valid_filenames.txt")
            
        else:
            print(f"\n{model_name}: No valid boundaries found!")
            all_results[model_name] = {
                'overall': {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'median': float('nan'),
                    'n_samples': 0,
                    'skipped_background': skipped_background,
                    'skipped_satellite_edge': skipped_satellite_edge,
                    'skipped_patch_boundary': skipped_patch_boundary,
                    'skipped_straight_edge': skipped_straight_edge,
                    'skipped_no_boundary': skipped_no_boundary
                }
            }
        
        print(f"✓ {model_name} completed")
    
    # ✅ ENHANCED: Print final summary
    print(f"\n{'='*60}")
    print("MODEL SUCCESS STATISTICS")
    print('='*60)
    print(f"{'Model':<20} {'Success Rate':<15} {'Avg MDE (m)':<15}")
    print('-' * 50)
    
    for model_name, results in all_results.items():
        if results['overall']['n_samples'] > 0:
            success_rate = results['overall']['n_samples']
            avg_mde = results['overall']['mean']
            print(f"{model_name:<20} {success_rate:<15} {avg_mde:<15.1f}")
        else:
            print(f"{model_name:<20} {'0.0':<15} {'nan':<15}")
    
    gc.collect()
    print(f"\n✓ All results saved to {output_dir}")
    return all_results

if __name__ == "__main__":
    from data_processing.ice_data import IceDataset
    from omegaconf import OmegaConf
    # from MDE import prepare_device, build_model_specs, load_models
    
    # Configuration
    parent_dir = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/ICE-BENCH"
    checkpoint_base = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/model_outputs"
    batch_size = 8
    device = prepare_device()
    
    cfg = OmegaConf.create({
        'model': {
            'name': 'Unet',
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
    
    iou_best_models = {
        "ViT_best_iou": "ViT_bs32_correct_labels_best_iou.pth",
        "Unet_best_iou": "Unet_bs32_correct_labels_latest_epoch.pth",
        "DeepLabV3_best_iou": "DeepLabV3_bs32_correct_labels_latest_epoch.pth",
        "FPN_best_iou": "FPN_bs32_correct_labels_best_iou.pth",
        "DinoV3_best_iou": "DinoV3_bs32_correct_labels_best_iou.pth",
    }
    
    model_specs = build_model_specs(checkpoint_base, iou_best_models)
    models = load_models(model_specs, cfg, device)
    
    #original test dataset
    test_datasets = IceDataset.create_test_datasets(parent_dir)
    test_dataset = list(test_datasets.values())[0]
    # test_loader = DataLoader(
    #     test_dataset, 
    #     batch_size=batch_size, 
    #     shuffle=False, 
    #     num_workers=0, 
    #     pin_memory=True
    # )
    
    # ============ NEW: FILTER BASED ON BACKGROUND JSON ============
    
    background_filters = load_background_filters(parent_dir)
    
    # Get valid indices (only files marked as false)
    valid_indices = get_valid_file_indices(test_dataset, background_filters)
    
    if len(valid_indices) == 0:
        print("ERROR: No valid files found after filtering!")
        exit(1)
    
    # Create filtered DataLoader
    test_loader = create_filtered_dataloader(test_dataset, valid_indices, batch_size)
    
    # Extract original filenames for the filtered indices
    if hasattr(test_dataset, 'image_files'):
        original_image_files = test_dataset.image_files
    elif hasattr(test_dataset, 'image_paths'):
        original_image_files = test_dataset.image_paths
    else:
        print("Warning: Could not extract original filenames from dataset")
        original_image_files = []
    
    # Store filtered filenames as basenames
    all_filenames = []
    for idx in valid_indices:
        if idx < len(original_image_files):
            filename = os.path.basename(original_image_files[idx])
            all_filenames.append(filename)
        else:
            all_filenames.append(f"sample_{idx}")
    
    print(f"\nFiltered filenames: {len(all_filenames)} files")

    print("="*70)

    
    results = run_fixed_mde_evaluation(
        models, 
        test_loader, 
        device,
        output_dir='./mde_results_filtered',  # Different output dir
        apply_morphological_filter=True,
        filter_operation='opening',
        filter_iterations=2,
        exclude_edges=True,
        edge_width=5,
        check_straightness=True,
        original_filenames=all_filenames  # NEW: Pass the filtered filenames
    )
     
    
    print("\n" + "="*70)
    print("FILTERED MDE EVALUATION COMPLETED")
    print("="*70)
    print(f"Processed {len(valid_indices)} valid samples (excluding background scenes)")
    print("Results saved to ./mde_results_filtered/")