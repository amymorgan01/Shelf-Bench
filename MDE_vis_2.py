
import os
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import cdist

# Import functions from MDE_2 module
from MDE_2 import (
    extract_boundary_contour_v2,
    extract_ice_front_boundary_fixed, 
    calculate_boundary_distance,
    get_satellite_resolution,
    process_mask,
    prepare_device,
    build_model_specs,
    load_models,
    load_background_filters,       
    get_valid_file_indices,        
    create_filtered_dataloader, 
)

MODEL_COLORS = {
    'ViT_best_iou': "#D84040",
    'Unet_best_iou': "#C950C9",
    'DeepLabV3_best_iou': "#DAD046",
    'FPN_best_iou': "#22B77B",
    'DinoV3_best_iou': "#2E70C6"
}

MODEL_SHORT_NAMES = {
    'ViT_best_iou': 'ViT',
    'Unet_best_iou': 'U-Net',
    'DeepLabV3_best_iou': 'DeepLabV3',
    'FPN_best_iou': 'FPN',
    'DinoV3_best_iou': 'DinoV3'
}

def get_model_display_info(model_name: str) -> Tuple[str, str]:
    """Get short name and color for a model."""
    short_name = MODEL_SHORT_NAMES.get(model_name, model_name.split('_')[0])
    color = MODEL_COLORS.get(model_name, '#FF6B6B')
    return short_name, color

def prepare_image_for_display(image: np.ndarray) -> np.ndarray:
    """Prepare image array for matplotlib display."""
    if image.ndim == 3 and image.shape[0] == 1:
        return image.squeeze(0)
    elif image.ndim == 3 and image.shape[0] == 3:
        return np.transpose(image, (1, 2, 0))
    return image

def calculate_mde_metrics(pred_boundary: np.ndarray, 
                         gt_boundary: np.ndarray,
                         pixel_res: float) -> Tuple[float, float]:
    """Calculate MDE in pixels and meters."""
    distances = cdist(pred_boundary, gt_boundary, metric='euclidean')
    mean_dist_px = np.mean(distances.min(axis=1))
    mde_m = mean_dist_px * pixel_res
    return mean_dist_px, mde_m

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

# def visualize_ice_front_fixed(
#     image: np.ndarray,
#     gt_mask: np.ndarray,
#     pred_masks: Dict[str, np.ndarray],
#     filename: str,
#     save_dir: str
# ) -> None:
#     """
#     Optimized visualization - FIXED to avoid double boundary plotting.
#     Individual model panels now show ONLY the prediction, not GT overlay.
#     """
    
#     # Extract GT boundary
#     gt_boundary = extract_ice_front_boundary_fixed(
#         gt_mask, image, background_threshold=1e-6, min_length=10
#     )
    
#     if gt_boundary is None:
#         print(f"Skipping {filename}: No valid GT ice-ocean boundary")
#         return
    
#     # Extract all predicted boundaries
#     pred_boundaries = {}
#     print(f"\n=== {filename} ===")
#     print(f"GT boundary: {len(gt_boundary)} points")
    
#     for model_name, pred_mask in pred_masks.items():
#         pred_boundary = extract_ice_front_boundary_fixed(
#             pred_mask, image, background_threshold=1e-6, min_length=50
#         )
#         pred_boundaries[model_name] = pred_boundary
        
#         status = f"✓ {len(pred_boundary)} points" if pred_boundary is not None else "✗ No valid boundary"
#         print(f"  {model_name}: {status}")
    
#     # Setup figure
#     n_models = len(pred_masks)
#     fig = plt.figure(figsize=(5 * (n_models + 2), 5))
    
#     # Prepare image for display
#     display_image = prepare_image_for_display(image)
#     is_grayscale = display_image.ndim == 2
    
#     # Panel 1: Raw satellite image
#     ax1 = plt.subplot(1, n_models + 2, 1)
#     if is_grayscale:
#         ax1.imshow(display_image, cmap='gray', vmin=0, vmax=1)
#     else:
#         ax1.imshow(display_image)
#     ax1.set_title('Raw Satellite Image', fontsize=12, fontweight='bold')
#     ax1.axis('off')
    
#     # Panel 2: Image + GT boundary ONLY
#     ax2 = plt.subplot(1, n_models + 2, 2)
#     if is_grayscale:
#         ax2.imshow(display_image, cmap='gray', vmin=0, vmax=1)
#     else:
#         ax2.imshow(display_image)
    
#     ax2.plot(gt_boundary[:, 1], gt_boundary[:, 0],
#              color='lime', linewidth=2.0, alpha=1.0)
#     ax2.set_title('Ground Truth\nIce-Ocean Boundary', fontsize=12, fontweight='bold')
#     ax2.axis('off')
    
#     # Get resolution once
#     pixel_res = get_satellite_resolution(filename)
    
#     # Panels 3+: Individual model predictions (PREDICTION ONLY - NO GT OVERLAY)
#     legend_elements = []
    
#     for idx, model_name in enumerate(pred_masks.keys(), 3):
#         pred_boundary = pred_boundaries[model_name]
#         short_name, color = get_model_display_info(model_name)
        
#         ax = plt.subplot(1, n_models + 2, idx)
        
#         # Display image
#         if is_grayscale:
#             ax.imshow(display_image, cmap='gray', vmin=0, vmax=1)
#         else:
#             ax.imshow(display_image)
        
#         # ✅ FIXED: Plot ONLY the predicted boundary (no GT overlay)
#         if pred_boundary is not None:
#             ax.plot(pred_boundary[:, 1], pred_boundary[:, 0],
#                     color=color, linewidth=2.0, alpha=1.0)
            
#             # Calculate MDE
#             _, mde_m = calculate_mde_metrics(pred_boundary, gt_boundary, pixel_res)
#             title = f'{short_name}\nMDE: {mde_m:.1f}m'
            
#             # Add to comprehensive legend
#             legend_elements.append(
#                 mpatches.Patch(color=color, label=f'{short_name}: {mde_m:.1f}m')
#             )
#         else:
#             # No valid boundary
#             ax.text(0.5, 0.5, 'Boundary\nExtraction\nFailed', 
#                    ha='center', va='center', transform=ax.transAxes,
#                    fontsize=12, color='red', fontweight='bold',
#                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#             title = f'{short_name}\nNo Valid Boundary'
            
#             legend_elements.append(
#                 mpatches.Patch(color=color, label=f'{short_name}: N/A')
#             )
        
#         ax.set_title(title, fontsize=11, fontweight='bold')
#         ax.axis('off')
    
#     # Overall title
#     fig.suptitle(f'Ice-Ocean Boundary Analysis: {filename}',
#                 fontsize=14, fontweight='bold', y=0.98)
    
#     # Add comprehensive legend at bottom
#     gt_patch = mpatches.Patch(color='lime', label='Ground Truth')
#     legend_elements.insert(0, gt_patch)
    
#     fig.legend(handles=legend_elements, 
#               loc='lower center', 
#               ncol=min(6, len(legend_elements)),
#               fontsize=10,
#               frameon=True,
#               fancybox=True,
#               shadow=True,
#               bbox_to_anchor=(0.5, -0.02))
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
#     # Save
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f'{os.path.splitext(filename)[0]}_fixed.png')
#     plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     print(f"✓ Saved: {save_path}")

def visualize_ice_front_fixed(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_masks: Dict[str, np.ndarray],
    filename: str,
    save_dir: str
) -> None:
    """
    Optimized visualization with comprehensive legend.
    Plots ice front boundary lines for all models.
    """
    
    # Extract GT boundary
    gt_boundary = extract_ice_front_boundary_fixed(
        gt_mask, image, background_threshold=0.05, min_length=50
    )
    
    if gt_boundary is None:
        print(f"Skipping {filename}: No valid GT ice-ocean boundary")
        return
    
    # Extract all predicted boundaries
    pred_boundaries = {}
    print(f"\n=== {filename} ===")
    print(f"GT boundary: {len(gt_boundary)} points")
    
    for model_name, pred_mask in pred_masks.items():
        pred_boundary = extract_ice_front_boundary_fixed(
            pred_mask, image, background_threshold=1e-6, min_length=50
        )
        pred_boundaries[model_name] = pred_boundary
        
        status = f"✓ {len(pred_boundary)} points" if pred_boundary is not None else "✗ No valid boundary"
        print(f"  {model_name}: {status}")
    
    # Setup figure
    n_models = len(pred_masks)
    fig = plt.figure(figsize=(5 * (n_models + 2), 5))
    
    # Prepare image for display
    display_image = prepare_image_for_display(image)
    is_grayscale = display_image.ndim == 2
    
    # Panel 1: Raw satellite image
    ax1 = plt.subplot(1, n_models + 2, 1)
    if is_grayscale:
        ax1.imshow(display_image, cmap='gray', vmin=0, vmax=1)
    else:
        ax1.imshow(display_image)
    ax1.set_title('Raw Satellite Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Image + GT boundary
    ax2 = plt.subplot(1, n_models + 2, 2)
    if is_grayscale:
        ax2.imshow(display_image, cmap='gray', vmin=0, vmax=1)
    else:
        ax2.imshow(display_image)
    
    ax2.plot(gt_boundary[:, 1], gt_boundary[:, 0],
             color='lime', linewidth=2.5, alpha=0.9, label='GT Ice Front')
    ax2.set_title('Ground Truth\nIce-Ocean Boundary', fontsize=12, fontweight='bold')
    ax2.axis('off')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.8)
    
    # Get resolution once
    pixel_res = get_satellite_resolution(filename)
    
    # Panels 3+: Individual model predictions
    legend_elements = []  # For comprehensive legend
    
    for idx, model_name in enumerate(pred_masks.keys(), 3):
        pred_boundary = pred_boundaries[model_name]
        short_name, color = get_model_display_info(model_name)
        
        ax = plt.subplot(1, n_models + 2, idx)
        
        # Display image
        if is_grayscale:
            ax.imshow(display_image, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(display_image)
        
        # GT boundary (faded)
        ax.plot(gt_boundary[:, 1], gt_boundary[:, 0],
                color='lime', linewidth=1.5, alpha=0.4, 
                linestyle='--', label='GT')
        
        # Predicted boundary
        if pred_boundary is not None:
            ax.plot(pred_boundary[:, 1], pred_boundary[:, 0],
                    color=color, linewidth=2.5, alpha=0.9, 
                    label='Prediction')
            
            # Calculate MDE
            _, mde_m = calculate_mde_metrics(pred_boundary, gt_boundary, pixel_res)
            title = f'{short_name}\nMDE: {mde_m:.1f}m'
            
            # Add to comprehensive legend
            legend_elements.append(
                mpatches.Patch(color=color, label=f'{short_name}')
            )
        else:
            # No valid boundary
            ax.text(0.5, 0.5, 'Boundary\nExtraction\nFailed', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            title = f'{short_name}\nNo Valid Boundary'
            
            # Add to legend with N/A
            legend_elements.append(
                mpatches.Patch(color=color, label=f'{short_name}: N/A')
            )
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    # Overall title
    fig.suptitle(f'Ice-Ocean Boundary Analysis: {filename}',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Add comprehensive legend at bottom
    fig.legend(handles=legend_elements, 
              loc='lower center', 
              ncol=min(5, len(legend_elements)),
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True,
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{os.path.splitext(filename)[0]}_fixed.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {save_path}")

def create_summary_comparison(
    results: Dict[str, List[float]],
    save_dir: str
) -> None:
    """
    Optimized summary plots with consistent styling and legend.
    REPLACES both create_summary_visualization and create_summary_comparison.
    """
    if not any(results.values()):
        print("No results to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get model info
    model_names = list(results.keys())
    short_names = [get_model_display_info(name)[0] for name in model_names]
    colors = [get_model_display_info(name)[1] for name in model_names]
    
    # 1. Box plot
    ax1 = axes[0, 0]
    data_to_plot = [results[name] for name in model_names]
    bp = ax1.boxplot(data_to_plot, labels=short_names, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('MDE (meters)', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance Distribution', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='-')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Mean MDE bar chart
    ax2 = axes[0, 1]
    means = [np.mean(results[name]) for name in model_names]
    stds = [np.std(results[name]) for name in model_names]
    
    bars = ax2.bar(short_names, means, yerr=stds, capsize=5, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, mean_val, std_val in zip(bars, means, stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.1f}m\n±{std_val:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Mean MDE (meters)', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Performance ± Std Dev', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Histogram comparison
    ax3 = axes[1, 0]
    for (model_name, distances), color, short_name in zip(results.items(), colors, short_names):
        if distances:
            ax3.hist(distances, bins=30, alpha=0.6, label=short_name, 
                    color=color, edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel('MDE (meters)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('MDE Distribution Histogram', fontsize=12, fontweight='bold')
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(alpha=0.3, linestyle='--')
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    for (model_name, distances), color, short_name in zip(results.items(), colors, short_names):
        if distances:
            sorted_data = np.sort(distances)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
            ax4.plot(sorted_data, cumulative, label=short_name, 
                    color=color, linewidth=2.5, alpha=0.8)
    
    ax4.set_xlabel('MDE (meters)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax4.legend(frameon=True, fancybox=True, shadow=True)
    ax4.grid(alpha=0.3, linestyle='--')
    
    # Add overall statistics in text box
    stats_text = "Summary Statistics:\n"
    for model_name, short_name in zip(model_names, short_names):
        if results[model_name]:
            mean_val = np.mean(results[model_name])
            median_val = np.median(results[model_name])
            stats_text += f"{short_name}: μ={mean_val:.1f}m, M={median_val:.1f}m\n"
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='bottom')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save
    save_path = os.path.join(save_dir, 'model_comparison_summary.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved summary visualization: {save_path}")

def run_batch_inference(
    models: Dict[str, torch.nn.Module],
    images: torch.Tensor,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Optimized batch inference - runs all models efficiently.
    """
    pred_masks_all = {}
    images_gpu = images.to(device)
    
    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            outputs = model(images_gpu)
            pred_masks = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            
            # Handle shape
            if pred_masks.shape[1] == 2:
                pred_masks = pred_masks[:, 1, :, :]
            elif pred_masks.shape[1] == 1:
                pred_masks = pred_masks.squeeze(1)
            
            pred_masks_all[model_name] = pred_masks
            del outputs
    
    del images_gpu
    torch.cuda.empty_cache()
    
    return pred_masks_all

def generate_fixed_visualizations(
    models: Dict[str, torch.nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str = './filtered_visualizations',  # ✅ Updated default name
    n_samples: int = 20,
    random_seed: Optional[int] = 42
) -> None:
    """
    Generate visualizations from FILTERED dataset (background scenes excluded).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Move all models to device once
    for model in models.values():
        model.to(device)
        model.eval()
    
    dataset_size = len(test_loader.dataset)  # This is the FILTERED size
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Sample from the filtered dataset
    sample_indices = np.random.choice(
        dataset_size, 
        size=min(n_samples, dataset_size), 
        replace=False
    )
    sample_indices = sorted(sample_indices)
    
    print(f"Sampling {len(sample_indices)} from {dataset_size} filtered samples...")
    
    # ✅ Get filenames from filtered dataset
    all_filenames = []
    
    # Handle different dataset types
    if hasattr(test_loader.dataset, 'dataset'):  # This is a Subset
        original_dataset = test_loader.dataset.dataset
        subset_indices = test_loader.dataset.indices
        
        if hasattr(original_dataset, 'image_files'):
            all_original_files = original_dataset.image_files
        elif hasattr(original_dataset, 'image_paths'):
            all_original_files = original_dataset.image_paths
        else:
            all_original_files = []
        
        # Get filenames for the filtered indices
        all_filenames = []
        for subset_idx in subset_indices:
            if subset_idx < len(all_original_files):
                filename = os.path.basename(all_original_files[subset_idx])
                all_filenames.append(filename)
            else:
                all_filenames.append(f"sample_{subset_idx}")
                
    else:  # Regular dataset
        if hasattr(test_loader.dataset, 'image_files'):
            all_filenames = [os.path.basename(f) for f in test_loader.dataset.image_files]
        elif hasattr(test_loader.dataset, 'image_paths'):
            all_filenames = [os.path.basename(p) for p in test_loader.dataset.image_paths]
    
    print(f"Available filenames: {len(all_filenames)}")
    if len(all_filenames) > 0:
        print(f"Example filenames: {all_filenames[:3]}")
    
    # Rest of the function stays the same...
    # Track results
    all_results = {name: [] for name in models.keys()}
    model_success_count = {name: 0 for name in models.keys()}
    total_processed = 0
    
    # Process batches
    samples_processed = 0
    current_idx = 0
    
    for batch_idx, batch in enumerate(test_loader):
        # Unpack batch
        if len(batch) == 3:
            images, masks, batch_filenames = batch
        else:
            images, masks = batch
            batch_start = batch_idx * test_loader.batch_size
            batch_filenames = []
            for i in range(len(images)):
                idx = batch_start + i
                if idx < len(all_filenames):
                    batch_filenames.append(all_filenames[idx])
                else:
                    batch_filenames.append(f"filtered_sample_{idx}")
        
        batch_size = len(images)
        batch_indices = range(current_idx, current_idx + batch_size)
        samples_to_viz = [i for i, idx in enumerate(batch_indices) if idx in sample_indices]
        
        if samples_to_viz:
            # Run inference once for all models
            pred_masks_all = run_batch_inference(models, images, device)
            
            # Process masks
            masks_np = process_mask(masks).cpu().numpy()
            
            # Visualize selected samples
            for i in samples_to_viz:
                image_np = prepare_image_for_display(images[i].cpu().numpy())
                gt_mask = masks_np[i]
                filename = batch_filenames[i]
                
                pred_masks_dict = {
                    name: preds[i].astype(np.uint8)
                    for name, preds in pred_masks_all.items()
                }
                
                # Visualize
                visualize_ice_front_fixed(
                    image_np, gt_mask, pred_masks_dict, 
                    filename, output_dir
                )
                
                # Calculate metrics for filtered data
                gt_boundary = extract_ice_front_boundary_fixed(
                    gt_mask, image_np, 
                    background_threshold=1e-6, min_length=10
                )
                
                if gt_boundary is not None:
                    pixel_res = get_satellite_resolution(filename)
                    
                    for model_name, pred_mask in pred_masks_dict.items():
                        pred_boundary = extract_ice_front_boundary_fixed(
                            pred_mask, image_np,
                            background_threshold=1e-6, min_length=10
                        )
                        
                        if pred_boundary is not None:
                            model_success_count[model_name] += 1
                            distance = calculate_boundary_distance(
                                pred_boundary, gt_boundary, pixel_res, 'mean'
                            )
                            if not np.isnan(distance):
                                all_results[model_name].append(distance)
                
                total_processed += 1
                samples_processed += 1
                print(f"Processed {samples_processed}/{len(sample_indices)}")
            
            del pred_masks_all, masks_np
        
        current_idx += batch_size
        
        if samples_processed >= n_samples:
            break
    
    # Print statistics
    print(f"\n{'='*70}")
    print("FILTERED DATASET - MODEL SUCCESS STATISTICS")
    print('='*70)
    print(f"{'Model':<25} {'Success Rate':<15} {'Avg MDE (m)':<15}")
    print("-" * 70)
    
    for model_name in models.keys():
        short_name = get_model_display_info(model_name)[0]
        success_rate = (model_success_count[model_name] / total_processed) * 100 if total_processed > 0 else 0
        avg_mde = np.mean(all_results[model_name]) if all_results[model_name] else np.nan
        print(f"{short_name:<25} {success_rate:>12.1f}% {avg_mde:>12.1f}")
    
    # Create summary
    if any(all_results.values()):
        create_summary_comparison(all_results, output_dir)
    
    print(f"\n✓ All FILTERED visualizations saved to {output_dir}")
    print(f"✓ Background scenes were automatically excluded")
    gc.collect()

if __name__ == "__main__":
    from data_processing.ice_data import IceDataset
    
    # Configuration
    parent_dir = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/ICE-BENCH"
    checkpoint_base = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/model_outputs"
    batch_size = 8
    n_visualizations = 200
    
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
    
    # # Create test dataloader
    # test_datasets = IceDataset.create_test_datasets(parent_dir)
    # test_dataset = list(test_datasets.values())[0]
    # test_loader = DataLoader(
    #     test_dataset, 
    #     batch_size=batch_size, 
    #     shuffle=False, 
    #     num_workers=0, 
    #     pin_memory=True
    # )
    
    test_datasets = IceDataset.create_test_datasets(parent_dir)
    test_dataset = list(test_datasets.values())[0]
    
    print("="*70)
    print("APPLYING BACKGROUND FILTERING TO VISUALIZATION DATASET")
    print("="*70)
    
    # Load background filters
    background_filters = load_background_filters(parent_dir)
    
    # Get valid indices (only files marked as false in background JSON)
    valid_indices = get_valid_file_indices(test_dataset, background_filters)
    
    if len(valid_indices) == 0:
        print("ERROR: No valid files found after filtering!")
        exit(1)
    
    # Create filtered DataLoader (same as MDE_2.py)
    test_loader = create_filtered_dataloader(test_dataset, valid_indices, batch_size)
    
    print(f"Filtered dataset: {len(valid_indices)} valid samples")
    print(f"Will generate visualizations from filtered dataset only")
    
    print("="*70)
    print("GENERATING OPTIMIZED ICE FRONT VISUALIZATIONS")
    print("="*70)
    
    generate_fixed_visualizations(
        models=models,
        test_loader=test_loader,
        device=device,
        output_dir='./filtered_ice_front_visualizations',
        n_samples=n_visualizations,
        random_seed=42
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION GENERATION COMPLETED")
    print("="*70)