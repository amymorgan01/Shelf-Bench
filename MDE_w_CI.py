"""
Code to calculate the MDE of ground truth labels with predicted fronts
We use our segmentation masks to derive the front line. Background and ocean are defined as 0 in masks, therefore scenes with backgrounds are pre-filtered out to avoid ice-background classifications
Code inspired by Gourmelen et al. (2022)

UPDATED: Now includes satellite-specific breakdown of MDE metrics
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import os
import gc
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from scipy.spatial.distance import directed_hausdorff, cdist
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from data_processing.ice_data import IceDataset
from omegaconf import OmegaConf
from pathlib import Path
from paths import ROOT_GWS, ROOT_LOCAL

# Import autocast from the correct location based on PyTorch version
try:
    from torch.cuda.amp import autocast
    USE_OLD_AUTOCAST = True
except ImportError:
    from torch.amp import autocast
    USE_OLD_AUTOCAST = False

@dataclass
class ModelSpec:
    arch: str
    name: str
    ckpt_path: str

# ============ FILE FILTERING ============

def load_background_filters(base_dir: str) -> Dict[str, bool]:
    background_info_dir = os.path.join(base_dir, "background_scenes", "background_info")
    json_files = ["Envisat_backgrounds.json", "ERS_backgrounds.json", "Sentinel-1_backgrounds.json"]
    
    combined_filters = {}
    for json_file in json_files:
        json_path = os.path.join(background_info_dir, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                combined_filters.update(json.load(f))
    
    return combined_filters

def get_valid_file_indices(dataset, background_filters: Dict[str, bool]) -> List[int]:
    valid_indices = []
    image_files = getattr(dataset, 'image_files', getattr(dataset, 'image_paths', []))
    
    basename_to_status = {os.path.basename(k): v for k, v in background_filters.items()}
    
    for idx, img_path in enumerate(image_files):
        basename = os.path.basename(img_path)
        if basename in basename_to_status and not basename_to_status[basename]:
            valid_indices.append(idx)
    
    print(f"Dataset filtering: {len(valid_indices)} valid, {len(image_files) - len(valid_indices)} skipped")
    return valid_indices

def create_filtered_dataloader(dataset, valid_indices: List[int], batch_size: int = 8,
                               num_workers: int = None, pin_memory: bool = True) -> DataLoader:
    if num_workers is None:
        try:
            num_workers = max(1, (os.cpu_count() or 4) - 1)
        except:
            num_workers = 4
    filtered_dataset = Subset(dataset, valid_indices)
    kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers,
                  pin_memory=pin_memory, persistent_workers=True, prefetch_factor=2)
    return DataLoader(filtered_dataset, **kwargs)

# ============ MASK PROCESSING ============

def normalize_mask_to_2d(mask: np.ndarray) -> np.ndarray:
    mask = mask.squeeze()
    while mask.ndim > 2:
        mask = mask[0]
    return mask.astype(np.uint8)

def apply_morphological_filter(mask: np.ndarray, operation: str = 'opening', iterations: int = 2) -> np.ndarray:
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    mask_binary = (mask > 0).astype(np.uint8)
    
    ops = {'erosion': binary_erosion, 'dilation': binary_dilation, 
           'opening': binary_opening, 'closing': binary_closing}
    
    return ops[operation](mask_binary, structure=structure, iterations=iterations).astype(np.uint8)

# ============ BOUNDARY EXTRACTION ============

def extract_boundary_contour(mask: np.ndarray, 
                            image: Optional[np.ndarray] = None,
                            morphological_iterations: int = 2,
                            min_contour_length: int = 50) -> Optional[np.ndarray]:
    mask_binary = (mask > 0.5).astype(np.uint8) if mask.dtype in [np.float32, np.float64] else mask.astype(np.uint8)
    
    if np.unique(mask_binary).size < 2:
        return None
    
    if morphological_iterations > 0:
        mask_binary = binary_opening(mask_binary, iterations=morphological_iterations).astype(np.uint8)
    
    ocean_mask = (mask_binary == 0)
    ocean_dilated = binary_dilation(ocean_mask, iterations=2).astype(np.uint8)
    ice_ocean_interface = mask_binary & ocean_dilated
    
    if not np.any(ice_ocean_interface):
        return None
    
    contours, _ = cv2.findContours((ice_ocean_interface * 255).astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    valid_contours = [c for c in contours if len(c) >= min_contour_length]
    if not valid_contours:
        return None
    
    longest = max(valid_contours, key=len)
    boundary = longest.squeeze()
    
    if boundary.ndim == 1:
        boundary = boundary.reshape(1, -1)
    
    boundary = boundary[:, [1, 0]].astype(np.float32)
    
    if is_boundary_straight_line(boundary) or is_boundary_on_patch_edge(boundary, mask.shape):
        return None
    
    return boundary

def is_boundary_straight_line(boundary: np.ndarray, threshold: float = 0.95) -> bool:
    if boundary is None or len(boundary) < 10:
        return False
    
    try:
        coords = boundary.astype(np.float64)
        coords_centered = coords - np.mean(coords, axis=0)
        _, s, _ = np.linalg.svd(coords_centered, full_matrices=False)
        if len(s) > 1 and s[0] > 0:
            return (1 - s[1] / s[0]) > threshold
    except:
        pass
    
    rows, cols = boundary[:, 0], boundary[:, 1]
    return np.std(rows) < 3.0 or np.std(cols) < 3.0

def is_boundary_on_patch_edge(boundary: np.ndarray, image_shape: tuple, edge_threshold: int = 8) -> bool:
    if boundary is None or len(boundary) == 0:
        return False
    
    height, width = image_shape
    rows, cols = boundary[:, 0], boundary[:, 1]
    
    edge_ratios = [
        np.sum(rows <= edge_threshold) / len(boundary),
        np.sum(rows >= height - edge_threshold - 1) / len(boundary),
        np.sum(cols <= edge_threshold) / len(boundary),
        np.sum(cols >= width - edge_threshold - 1) / len(boundary)
    ]
    
    return max(edge_ratios) > 0.3

# ============ DISTANCE CALCULATION ============

def calculate_boundary_distance(pred_boundary: np.ndarray, gt_boundary: np.ndarray,
                               pixel_resolution_m: float, metric: str = 'mean') -> float:
    if pred_boundary is None or gt_boundary is None or len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return np.nan
    
    if metric == 'hausdorff':
        d1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
        d2 = directed_hausdorff(gt_boundary, pred_boundary)[0]
        distance_pixels = max(d1, d2)
    else:
        distances = cdist(pred_boundary, gt_boundary, metric='euclidean')
        min_distances = distances.min(axis=1)
        distance_pixels = np.mean(min_distances) if metric == 'mean' else np.median(min_distances)
    
    return distance_pixels * pixel_resolution_m


def get_satellite_resolution(filename: str) -> float:
    """Get pixel resolution in meters for the satellite"""
    filename_upper = filename.upper()
    if 'S1' in filename_upper:
        return 40.0
    elif 'ERS' in filename_upper or 'ENV' in filename_upper:
        return 30.0
    return 30.0


def get_satellite_name(filename: str) -> str:
    """Extract satellite name from filename"""
    filename_upper = filename.upper()
    if 'S1' in filename_upper or 'SENTINEL' in filename_upper:
        return 'Sentinel-1'
    elif 'ERS' in filename_upper:
        return 'ERS'
    elif 'ENV' in filename_upper or 'ENVISAT' in filename_upper:
        return 'Envisat'
    return 'Unknown'

# ============ MODEL MANAGEMENT ============

def prepare_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    return device

def build_model_specs(base_path: str, ckpt_names: Dict[str, str]) -> List[ModelSpec]:
    specs = []
    for model_key, ckpt in ckpt_names.items():
        arch = model_key.split('_')[0]
        specs.append(ModelSpec(arch=arch, name=model_key, ckpt_path=os.path.join(base_path, arch, ckpt)))
    return specs

def load_models(model_specs: List[ModelSpec], cfg, device: torch.device) -> Dict[str, torch.nn.Module]:
    """Load models using the standardized load_full_model_state function"""
    from omegaconf import OmegaConf
    from load_functions import load_model, load_full_model_state  # Import the new function

    models = {}
    for spec in model_specs:
        try:
            cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            
            # Ensure segmentation_head exists
            if 'model' in cfg_copy and isinstance(cfg_copy.model, dict):
                cfg_copy.model = OmegaConf.create(cfg_copy.model)
            
            if not hasattr(cfg_copy, 'model'):
                cfg_copy.model = OmegaConf.create({})
            
            if not hasattr(cfg_copy.model, 'segmentation_head') or cfg_copy.model.get('segmentation_head') in [None, ""]:
                default_head = getattr(cfg, 'model', {}).get('segmentation_head', 'unet')
                cfg_copy.model.segmentation_head = default_head
            
            # Handle DinoV3 satellite weights
            if 'dino' in spec.arch.lower() or 'dinov3' in spec.arch.lower():
                if (not hasattr(cfg_copy.model, 'satellite_weights_path')) or (cfg_copy.model.get('satellite_weights_path') in [None, ""]):
                    if hasattr(cfg, 'model') and hasattr(cfg.model, 'satellite_weights_path'):
                        cfg_copy.model.satellite_weights_path = cfg.model.satellite_weights_path
                    else:
                        cfg_copy.model.satellite_weights_path = ""
            
            cfg_copy.model.name = spec.arch
            model = load_model(cfg_copy, device)
            
            # Load checkpoint
            ckpt = torch.load(spec.ckpt_path, map_location=device, weights_only=False)
            
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            
            # **NEW: Use the standardized loading function**
            model = load_full_model_state(model, state_dict, spec.name)
            
            model.to(device)
            model.eval()
            models[spec.name] = model
            
            del ckpt, state_dict
            gc.collect()
            
            print(f"✓ {spec.name} loaded and moved to {device}")
            
        except Exception as e:
            print(f"✗ Error loading {spec.name}: {e}")
            print(f"  Skipping this model and continuing with others...")
    
    return models

def process_mask(masks: torch.Tensor) -> torch.Tensor:
    if masks.max() > 1:
        masks = masks / 255.0
    if masks.dim() == 4 and masks.size(1) == 1:
        masks = masks.squeeze(1)
    return (1 - masks).long()

# ============ OPTIMIZED FUNCTIONS ============

def calculate_boundary_distance_gpu(pred_boundary: np.ndarray, 
                                    gt_boundary: np.ndarray,
                                    pixel_resolution_m: float, 
                                    metric: str = 'mean',
                                    device: torch.device = None) -> float:
    """GPU-accelerated boundary distance calculation"""
    if pred_boundary is None or gt_boundary is None or len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return np.nan
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        pred_t = torch.from_numpy(pred_boundary.astype(np.float32)).to(device)
        gt_t = torch.from_numpy(gt_boundary.astype(np.float32)).to(device)
        
        if metric == 'hausdorff':
            chunk_size = 1000
            max_dist_1 = 0.0
            max_dist_2 = 0.0
            
            for i in range(0, len(pred_t), chunk_size):
                pred_chunk = pred_t[i:i+chunk_size]
                dists = torch.cdist(pred_chunk.unsqueeze(0), gt_t.unsqueeze(0), p=2).squeeze(0)
                max_dist_1 = max(max_dist_1, dists.min(dim=1)[0].max().item())
            
            for i in range(0, len(gt_t), chunk_size):
                gt_chunk = gt_t[i:i+chunk_size]
                dists = torch.cdist(gt_chunk.unsqueeze(0), pred_t.unsqueeze(0), p=2).squeeze(0)
                max_dist_2 = max(max_dist_2, dists.min(dim=1)[0].max().item())
            
            distance_pixels = max(max_dist_1, max_dist_2)
        else:
            chunk_size = 1000
            min_distances = []
            
            for i in range(0, len(pred_t), chunk_size):
                pred_chunk = pred_t[i:i+chunk_size]
                dists = torch.cdist(pred_chunk.unsqueeze(0), gt_t.unsqueeze(0), p=2).squeeze(0)
                min_dists_chunk = dists.min(dim=1)[0]
                min_distances.append(min_dists_chunk)
            
            min_distances = torch.cat(min_distances)
            distance_pixels = min_distances.mean().item() if metric == 'mean' else min_distances.median().item()
        
        return distance_pixels * pixel_resolution_m
        
    except Exception as e:
        print(f"GPU calc failed, using CPU fallback: {e}")
        return calculate_boundary_distance(pred_boundary, gt_boundary, pixel_resolution_m, metric)

def extract_boundary_worker(args):
    """Worker for parallel boundary extraction"""
    mask, morphological_iterations, min_contour_length = args
    return extract_boundary_contour(mask, None, morphological_iterations, min_contour_length)

def extract_boundaries_parallel(masks: np.ndarray, 
                               morphological_iterations: int = 2,
                               min_contour_length: int = 50,
                               max_workers: int = None) -> List[Optional[np.ndarray]]:
    """Extract boundaries from multiple masks in parallel"""
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)
    
    args_list = [(mask, morphological_iterations, min_contour_length) for mask in masks]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        boundaries = list(executor.map(extract_boundary_worker, args_list))
    
    return boundaries

def bootstrap_ci_fast(data: np.ndarray,
                      stat_func=np.mean,
                      n_boot: int = 1000,
                      alpha: float = 0.05,
                      rng_seed: int = 0) -> Tuple[float, float]:
    """Vectorized bootstrap CI"""
    if data.size < 2:
        return (float('nan'), float('nan'))
    
    rng = np.random.default_rng(rng_seed)
    n = data.size
    
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_samples = data[indices]
    
    if stat_func == np.mean:
        boot_stats = np.mean(boot_samples, axis=1)
    elif stat_func == np.median:
        boot_stats = np.median(boot_samples, axis=1)
    else:
        boot_stats = np.array([stat_func(sample) for sample in boot_samples])
    
    boot_stats = boot_stats[np.isfinite(boot_stats)]
    
    if boot_stats.size == 0:
        return (float('nan'), float('nan'))
    
    lower = np.percentile(boot_stats, 100 * (alpha / 2.0))
    upper = np.percentile(boot_stats, 100 * (1.0 - alpha / 2.0))
    
    return (float(lower), float(upper))


def calculate_satellite_breakdown(distances_by_satellite: Dict[str, List[float]], 
                                 ci_n_boot: int = 1000, 
                                 ci_alpha: float = 0.05) -> Dict[str, Dict]:
    """Calculate MDE statistics for each satellite separately"""
    satellite_results = {}
    
    for sat_name, distances in distances_by_satellite.items():
        if len(distances) > 0:
            arr = np.asarray(distances, dtype=float)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            median_val = float(np.median(arr))
            n_samples = int(arr.size)
            
            # Calculate confidence intervals
            mean_ci_lower, mean_ci_upper = bootstrap_ci_fast(
                arr, stat_func=np.mean, n_boot=ci_n_boot, alpha=ci_alpha, rng_seed=0
            )
            median_ci_lower, median_ci_upper = bootstrap_ci_fast(
                arr, stat_func=np.median, n_boot=ci_n_boot, alpha=ci_alpha, rng_seed=1
            )
            
            satellite_results[sat_name] = {
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'n_samples': n_samples,
                'mean_ci': (mean_ci_lower, mean_ci_upper),
                'median_ci': (median_ci_lower, median_ci_upper)
            }
        else:
            satellite_results[sat_name] = {
                'mean': float('nan'),
                'std': float('nan'),
                'median': float('nan'),
                'n_samples': 0,
                'mean_ci': (float('nan'), float('nan')),
                'median_ci': (float('nan'), float('nan'))
            }
    
    return satellite_results


def run_mde_evaluation_optimized(models: Dict[str, torch.nn.Module],
                                test_loader: DataLoader,
                                device: torch.device,
                                output_dir: str = './mde_results',
                                filter_iterations: int = 2,
                                original_filenames: Optional[List[str]] = None,
                                ci_n_boot: int = 1000,
                                ci_alpha: float = 0.05,
                                use_gpu_distances: bool = True) -> Dict[str, Dict]:
    """Optimized MDE evaluation with satellite-specific breakdown"""
    all_results = {}
    os.makedirs(output_dir, exist_ok=True)

    if original_filenames is None:
        original_filenames = getattr(test_loader.dataset, 'image_files', 
                                    [f"sample_{i}" for i in range(len(test_loader.dataset))])

    for model_idx, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n{'='*60}")
        print(f"Processing {model_idx}/{len(models)}: {model_name}")
        print(f"{'='*60}")

        model = model.to(device)
        model.eval()

        all_distances = []
        all_valid_filenames = []
        distances_by_satellite = defaultdict(list)  # NEW: Track distances per satellite
        skipped_counts = defaultdict(int)
        total_batches = len(test_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx + 1}/{total_batches} processed...")
                
                if len(batch) == 3:
                    images, masks, batch_filenames = batch
                else:
                    images, masks = batch
                    batch_start = batch_idx * test_loader.batch_size
                    batch_filenames = [original_filenames[batch_start + i] 
                                     for i in range(len(images)) if batch_start + i < len(original_filenames)]

                images_gpu = images.to(device, non_blocking=True)
                
                # Fixed autocast usage - compatible with both old and new PyTorch
                if device.type == 'cuda':
                    if USE_OLD_AUTOCAST:
                        with autocast(enabled=True):
                            outputs = model(images_gpu)
                    else:
                        with torch.amp.autocast('cuda', enabled=True):
                            outputs = model(images_gpu)
                else:
                    outputs = model(images_gpu)

                pred_masks_np = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                masks_np = process_mask(masks).cpu().numpy()

                if pred_masks_np.ndim == 4 and pred_masks_np.shape[1] == 2:
                    pred_masks_np = pred_masks_np[:, 1, :, :]
                elif pred_masks_np.ndim == 4 and pred_masks_np.shape[1] == 1:
                    pred_masks_np = pred_masks_np.squeeze(1)
                elif pred_masks_np.ndim == 2:
                    pred_masks_np = pred_masks_np[np.newaxis, ...]

                pred_boundaries = extract_boundaries_parallel(
                    pred_masks_np, 
                    morphological_iterations=filter_iterations
                )
                gt_boundaries = extract_boundaries_parallel(
                    masks_np,
                    morphological_iterations=filter_iterations
                )

                for i, filename in enumerate(batch_filenames):
                    pred_boundary = pred_boundaries[i] if i < len(pred_boundaries) else None
                    gt_boundary = gt_boundaries[i] if i < len(gt_boundaries) else None

                    if pred_boundary is None or gt_boundary is None:
                        skipped_counts['no_boundary'] += 1
                        continue

                    pixel_res = get_satellite_resolution(filename)
                    satellite_name = get_satellite_name(filename)  # NEW: Get satellite name
                    
                    if use_gpu_distances and device.type == 'cuda':
                        distance = calculate_boundary_distance_gpu(
                            pred_boundary, gt_boundary, pixel_res, 'mean', device
                        )
                    else:
                        distance = calculate_boundary_distance(
                            pred_boundary, gt_boundary, pixel_res, 'mean'
                        )

                    if not np.isnan(distance):
                        all_distances.append(distance)
                        all_valid_filenames.append(filename)
                        distances_by_satellite[satellite_name].append(distance)  # NEW: Track by satellite

                del outputs, pred_masks_np, masks_np, images_gpu
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        model = model.to('cpu')
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        if len(all_distances) > 0:
            arr = np.asarray(all_distances, dtype=float)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            median_val = float(np.median(arr))
            n_samples = int(arr.size)

            print(f"\n  Computing bootstrap confidence intervals...")
            mean_ci_lower, mean_ci_upper = bootstrap_ci_fast(
                arr, stat_func=np.mean, n_boot=ci_n_boot, alpha=ci_alpha, rng_seed=0
            )
            median_ci_lower, median_ci_upper = bootstrap_ci_fast(
                arr, stat_func=np.median, n_boot=ci_n_boot, alpha=ci_alpha, rng_seed=1
            )

            # NEW: Calculate satellite-specific breakdown
            satellite_breakdown = calculate_satellite_breakdown(
                distances_by_satellite, ci_n_boot, ci_alpha
            )

            results = {
                'overall': {
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'n_samples': n_samples,
                    'mean_ci': (mean_ci_lower, mean_ci_upper),
                    'median_ci': (median_ci_lower, median_ci_upper)
                },
                'by_satellite': satellite_breakdown  # NEW: Add satellite breakdown
            }
            all_results[model_name] = results

            print(f"\n  ✓ Results for {model_name}:")
            print(f"    Overall Mean MDE: {mean_val:.2f} m (95% CI [{mean_ci_lower:.2f}, {mean_ci_upper:.2f}])")
            print(f"    Overall Median MDE: {median_val:.2f} m (95% CI [{median_ci_lower:.2f}, {median_ci_upper:.2f}])")
            print(f"    Valid samples: {n_samples}")
            
            # NEW: Print satellite breakdown
            print(f"\n  Satellite Breakdown:")
            for sat_name, sat_results in satellite_breakdown.items():
                if sat_results['n_samples'] > 0:
                    print(f"    {sat_name}:")
                    print(f"      Mean: {sat_results['mean']:.2f} m (95% CI [{sat_results['mean_ci'][0]:.2f}, {sat_results['mean_ci'][1]:.2f}])")
                    print(f"      Median: {sat_results['median']:.2f} m (95% CI [{sat_results['median_ci'][0]:.2f}, {sat_results['median_ci'][1]:.2f}])")
                    print(f"      N: {sat_results['n_samples']}")

            # Save individual results CSV
            with open(os.path.join(model_dir, 'MDE_results.csv'), 'w') as f:
                f.write("filename,satellite,distance_m\n")
                for fn, dist in zip(all_valid_filenames, all_distances):
                    sat = get_satellite_name(fn)
                    f.write(f"{fn},{sat},{dist:.4f}\n")
            
            # NEW: Save satellite breakdown CSV
            with open(os.path.join(model_dir, 'MDE_satellite_breakdown.csv'), 'w') as f:
                f.write("satellite,mean_m,std_m,median_m,n_samples,mean_ci_lower,mean_ci_upper,median_ci_lower,median_ci_upper\n")
                for sat_name, sat_results in satellite_breakdown.items():
                    if sat_results['n_samples'] > 0:
                        f.write(
                            f"{sat_name},"
                            f"{sat_results['mean']:.2f},"
                            f"{sat_results['std']:.2f},"
                            f"{sat_results['median']:.2f},"
                            f"{sat_results['n_samples']},"
                            f"{sat_results['mean_ci'][0]:.2f},"
                            f"{sat_results['mean_ci'][1]:.2f},"
                            f"{sat_results['median_ci'][0]:.2f},"
                            f"{sat_results['median_ci'][1]:.2f}\n"
                        )
        else:
            all_results[model_name] = {
                'overall': {'mean': float('nan'), 'n_samples': 0},
                'by_satellite': {}
            }
            print(f"  ✗ No valid samples for {model_name}")

    # Save overall summary CSVs
    with open(os.path.join(output_dir, 'MDE_model_comparison_090226_X100.csv'), 'w') as f:
        f.write("model,mean_mde_m,std_m,median_m,n_valid\n")
        for model_name, results in all_results.items():
            if results['overall']['n_samples'] > 0:
                r = results['overall']
                f.write(f"{model_name},{r['mean']:.2f},{r['std']:.2f},{r['median']:.2f},{r['n_samples']}\n")

    with open(os.path.join(output_dir, 'MDE_model_comparison_ci_090226_X100.csv'), 'w') as f:
        header = (
            "model,mean_mde_m,std_m,median_m,n_valid,"
            "mean_ci_lower_m,mean_ci_upper_m,median_ci_lower_m,median_ci_upper_m\n"
        )
        f.write(header)
        for model_name, results in all_results.items():
            r = results['overall']
            if r.get('n_samples', 0) > 0:
                mean_ci = r.get('mean_ci', (float('nan'), float('nan')))
                median_ci = r.get('median_ci', (float('nan'), float('nan')))
                f.write(
                    f"{model_name},{r['mean']:.2f},{r['std']:.2f},{r['median']:.2f},{r['n_samples']},"
                    f"{mean_ci[0]:.2f},{mean_ci[1]:.2f},{median_ci[0]:.2f},{median_ci[1]:.2f}\n"
                )
            else:
                f.write(f"{model_name},nan,nan,nan,0,nan,nan,nan,nan\n")

    # NEW: Save consolidated satellite breakdown across all models
    with open(os.path.join(output_dir, 'MDE_all_models_satellite_breakdown.csv'), 'w') as f:
        f.write("model,satellite,mean_m,median_m,n_samples,mean_ci_lower,mean_ci_upper,median_ci_lower,median_ci_upper\n")
        for model_name, results in all_results.items():
            if 'by_satellite' in results:
                for sat_name, sat_results in results['by_satellite'].items():
                    if sat_results['n_samples'] > 0:
                        f.write(
                            f"{model_name},{sat_name},"
                            f"{sat_results['mean']:.2f},"
                            f"{sat_results['median']:.2f},"
                            f"{sat_results['n_samples']},"
                            f"{sat_results['mean_ci'][0]:.2f},"
                            f"{sat_results['mean_ci'][1]:.2f},"
                            f"{sat_results['median_ci'][0]:.2f},"
                            f"{sat_results['median_ci'][1]:.2f}\n"
                        )

    print(f"\n{'='*60}")
    print(f"✓ All results saved to {output_dir}")
    print(f"  - Overall summaries: MDE_model_comparison_*.csv")
    print(f"  - Satellite breakdown: MDE_all_models_satellite_breakdown.csv")
    print(f"  - Individual model results in each model subdirectory")
    print(f"{'='*60}")
    return all_results


if __name__ == "__main__":
    
    parent_dir = ROOT_GWS / "benchmark_data_CB" / "ICE-BENCH"
    checkpoint_base = ROOT_GWS / "benchmark_data_CB" / "model_outputs"

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

    # iou_best_models = {
    #     "ViT_best_iou": "ViT_retrain_090226_X100_debug_best_iou.pth",
    #     "Unet_best_iou": "Unet_retrain_090226_X100_debug_best_iou.pth",
    #     "DeepLabV3_best_iou": "DeepLabV3_retrain_090226_X100_debug_best_iou.pth",
    #     "FPN_best_iou": "FPN_retrain_090226_X100_debug_best_iou.pth",
    #     "DinoV3_best_iou": "DinoV3_retrain_090226_X100_debug_best_iou.pth",
    # }
    
    # model_specs = build_model_specs(checkpoint_base, iou_best_models)
    # models = load_models(model_specs, cfg, device)
        # -------------------------
    # dynamic checkpoint discovery
    # -------------------------
    # Will discover .pth/.pt/.ckpt files below each architecture directory in checkpoint_base
    # and build names that preserve suffixes like "best_iou" or "best_loss".
    from pathlib import Path
    checkpoint_base_path = Path(checkpoint_base)

    # allowed checkpoint extensions
    ckpt_exts = {'.pth', '.pt', '.ckpt'}

    discovered_ckpts = {}
    for ckpt_path in checkpoint_base_path.rglob("*"):
        if ckpt_path.suffix.lower() in ckpt_exts and ckpt_path.is_file():
            # arch is the immediate parent directory name (e.g. 'ViT', 'Unet' etc)
            arch = ckpt_path.parent.name
            stem = ckpt_path.stem  # filename without extension

            # Try to extract suffix (e.g. 'best_iou', 'best_loss') from filename stem
            suffix = None
            for candidate in ['best_iou', 'best_loss', 'best_val', 'final', 'last']:
                if candidate in stem:
                    suffix = candidate
                    break
            # fallback: use last token after '_' if exists
            if suffix is None and '_' in stem:
                suffix = stem.split('_')[-1]
            if suffix is None:
                suffix = 'checkpoint'

            # build friendly model name like "ViT_best_iou"
            model_name = f"{arch}_{suffix}"

            # If there are multiple with same model_name, prefer explicit 'best' ones first,
            # otherwise keep the first found. You can tweak this logic if desired.
            if model_name not in discovered_ckpts:
                discovered_ckpts[model_name] = str(ckpt_path)

    if not discovered_ckpts:
        raise RuntimeError(f"No checkpoints found under {checkpoint_base_path} (checked extensions {ckpt_exts})")

    # Build specs from discovered dict
    model_specs = build_model_specs(str(checkpoint_base_path), discovered_ckpts)
    models = load_models(model_specs, cfg, device)

    
    test_datasets = IceDataset.create_test_datasets(parent_dir)
    test_dataset = list(test_datasets.values())[0]
    
    background_filters = load_background_filters(parent_dir)
    valid_indices = get_valid_file_indices(test_dataset, background_filters)
    test_loader = create_filtered_dataloader(test_dataset, valid_indices, batch_size,
                                             num_workers=6, pin_memory=True)
    
    original_filenames = [os.path.basename(test_dataset.image_files[idx]) for idx in valid_indices]
    
    results = run_mde_evaluation_optimized(
        models, 
        test_loader, 
        device, 
        './mde_results_filtered', 
        filter_iterations=2, 
        original_filenames=original_filenames,
        ci_n_boot=1000,
        use_gpu_distances=True
    )