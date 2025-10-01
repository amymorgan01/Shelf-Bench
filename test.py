"""
    
    Test file for ICE-BENCH trained models
    Evaluates 5 models (UNet, FPN, DeepLabV3, ViT, DinoV3)
    
"""

# imports - delete uneeded ones later
from data_processing.ice_data import IceDataset
from models.ViT import create_vit_large_16
from models.DinoV3 import DINOv3SegmentationModel
from load_functions import load_model, get_loss_function
from data_processing.ice_data import IceDataset
from metrics import calculate_metrics, calculate_iou_metrics, evaluate_model, calculate_pixel_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import warnings
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score
import logging
import gc
import hydra

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)

# metrics functions

def evaluate_single_model(model_path, test_loader, device, cfg, model_name, architecture):
    """
    Evaluate a single model and return metrics.
    Args:
        model_path: Path to the trained model checkpoint
        test_loader: Test data loader
        device: Device to run evaluation on
        cfg: Configuration object
        model_name: Name of the model for logging
        architecture: Model architecture name
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    log.info(f"Evaluating {architecture} model: {model_name}")
    log.info(f"Loading model from {model_path}")

    try:
        
        # Update config for current architecture
        cfg_copy = cfg.copy()
        cfg_copy.model.name = architecture 
        model = load_model(cfg_copy, device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Initialize metrics
        num_classes = cfg.model.classes
        running_precision = np.zeros(num_classes)
        running_recall = np.zeros(num_classes) 
        running_f1 = np.zeros(num_classes)
        running_loss = 0.0
        num_batches = 0
        
        # IoU tracking - accumulate intersections and unions
        class_intersection_totals = np.zeros(num_classes)
        class_union_totals = np.zeros(num_classes)
        
        # Pixel accuracy tracking
        total_correct_pixels = 0
        total_pixels = 0
        sklearn_predictions = []
        sklearn_targets = []
        sample_every_n_batches = max(1, len(test_loader) // 20)  # Sample 5% for sklearn

        loss_function = get_loss_function(cfg)

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                # **APPLY SAME PREPROCESSING AS IN TRAINING**
                # 1. Normalize from [0,255] to [0,1]
                if masks.max() > 1:
                    masks = masks / 255.0
                
                # 2. Remove channel dimension if present
                if masks.dim() == 4 and masks.size(1) == 1:
                    masks = masks.squeeze(1)
                
                # 3. Invert masks (same as training)
                masks = 1 - masks
                
                # 4. Ensure integer type for class indices
                masks = masks.long()
                
                # Debug logging for first batch
                if batch_idx == 0:
                    log.info(f"Debug - Batch {batch_idx}:")                      
                    log.info(f"  Original masks range: [0, 255] -> After processing: [{masks.min()}, {masks.max()}]")
                    log.info(f"  Masks shape: {masks.shape}")
                    log.info(f"  Masks unique values: {torch.unique(masks)}")
                    log.info(f"  Outputs shape: {outputs.shape if 'outputs' in locals() else 'Not computed yet'}")
                    log.info(f"  Num classes: {num_classes}")

                outputs = model(images)
                if batch_idx == 0:
                    log.info(f"  Outputs shape: {outputs.shape}")
                
                # **HANDLE LOSS CALCULATION FOR MONAI CombinedLoss**
                try:
                    # For MONAI losses with to_onehot_y=False, it should accept class indices
                    # But let's try both approaches
                    loss = loss_function(outputs, masks)
                    if batch_idx == 0:
                        log.info(f"  Loss computed successfully with class indices: {loss.item():.4f}")
                except Exception as e1:
                    try:
                        # Fallback: Try with one-hot encoding
                        masks_one_hot = F.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
                        loss = loss_function(outputs, masks_one_hot)
                        if batch_idx == 0:
                            log.info(f"  Loss computed successfully with one-hot format: {loss.item():.4f}")
                    except Exception as e2:
                        log.warning(f"Failed to compute loss: {e1}, {e2}")
                        loss = torch.tensor(0.0)


                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                # EFFICIENT: Calculate pixel accuracy directly
                correct_pixels = (preds == masks).sum().item()
                batch_pixels = masks.numel()
                total_correct_pixels += correct_pixels
                total_pixels += batch_pixels

                # Calculate batch metrics using your existing functions
                batch_precision, batch_recall, batch_f1 = calculate_metrics(
                    masks, preds, num_classes, device
                )
                
                # Accumulate metrics
                running_precision += batch_precision
                running_recall += batch_recall
                running_f1 += batch_f1

                # EFFICIENT IoU calculation - accumulate intersections and unions
                for cls in range(num_classes):
                    pred_cls = (preds == cls)
                    target_cls = (masks == cls)
                    intersection = (pred_cls & target_cls).sum().float()
                    union = (pred_cls | target_cls).sum().float()
                    
                    class_intersection_totals[cls] += intersection.item()
                    class_union_totals[cls] += union.item()

                # MEMORY OPTIMIZATION: Only sample some batches for sklearn
                if batch_idx % sample_every_n_batches == 0:
                    # Subsample pixels from this batch
                    preds_flat = preds.view(-1).cpu().numpy()
                    masks_flat = masks.view(-1).cpu().numpy()
                    
                    # Random subsample to reduce memory
                    batch_size = masks_flat.shape[0]
                    if batch_size > 10000:  # Subsample large batches
                        indices = np.random.choice(batch_size, 10000, replace=False)
                        sample_masks = masks_flat[indices]
                        sample_preds = preds_flat[indices]
                    else:
                        sample_masks = masks_flat
                        sample_preds = preds_flat
                    
                    sklearn_predictions.extend(sample_preds)
                    sklearn_targets.extend(sample_masks)
                    
                    # Limit total samples
                    if len(sklearn_predictions) > 200000:  # 200K pixels max
                        sklearn_predictions = sklearn_predictions[-100000:]
                        sklearn_targets = sklearn_targets[-100000:]

                num_batches += 1

        # Calculate final metrics
        avg_loss = running_loss / num_batches
        pixel_accuracy = total_correct_pixels / total_pixels
        avg_precision = running_precision / num_batches
        avg_recall = running_recall / num_batches
        avg_f1 = running_f1 / num_batches

        # EFFICIENT IoU calculation
        class_ious = []
        for cls in range(num_classes):
            if class_union_totals[cls] > 0:
                iou = class_intersection_totals[cls] / class_union_totals[cls]
            else:
                iou = 0.0
            class_ious.append(iou)
        
        class_ious = np.array(class_ious)
        mean_iou = class_ious.mean()

        # OPTIONAL: Calculate sklearn metrics on subset for verification
        sklearn_accuracy = sklearn_precision = sklearn_recall = sklearn_f1 = sklearn_jaccard = None
        if len(sklearn_predictions) > 0:
            sklearn_predictions = np.array(sklearn_predictions)
            sklearn_targets = np.array(sklearn_targets)
            
            sklearn_accuracy = accuracy_score(sklearn_targets, sklearn_predictions)
            sklearn_precision = precision_score(sklearn_targets, sklearn_predictions, average=None, zero_division=0)
            sklearn_recall = recall_score(sklearn_targets, sklearn_predictions, average=None, zero_division=0)
            sklearn_f1 = f1_score(sklearn_targets, sklearn_predictions, average=None, zero_division=0)
            sklearn_jaccard = jaccard_score(sklearn_targets, sklearn_predictions, average=None, zero_division=0)

        # Create metrics dictionary
        metrics = {
            "model_name": model_name,
            "architecture": architecture,
            "loss": avg_loss,
            "pixel_accuracy": pixel_accuracy,
            "mean_iou": mean_iou,
            "class_ious": class_ious.tolist(),  # Convert to list for CSV compatibility
            "mean_precision": avg_precision.mean(),
            "mean_recall": avg_recall.mean(),
            "mean_f1": avg_f1.mean(),
            "precision_per_class": avg_precision.tolist(),  # Convert to list for CSV compatibility
            "recall_per_class": avg_recall.tolist(),  # Convert to list for CSV compatibility
            "f1_per_class": avg_f1.tolist(),  # Convert to list for CSV compatibility
            # Sklearn verification metrics (sampled)
            "sklearn_accuracy": sklearn_accuracy,
            "sklearn_precision_per_class": sklearn_precision.tolist() if sklearn_precision is not None else None,
            "sklearn_recall_per_class": sklearn_recall.tolist() if sklearn_recall is not None else None,
            "sklearn_f1_per_class": sklearn_f1.tolist() if sklearn_f1 is not None else None,
            "sklearn_jaccard_per_class": sklearn_jaccard.tolist() if sklearn_jaccard is not None else None,
        }

        log.info(f"Completed evaluation for {architecture} - {model_name}")
        log.info(f"Mean IoU: {mean_iou:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}")
        
        # Clean up memory
        del model, checkpoint, outputs, preds
        torch.cuda.empty_cache()
        gc.collect()

        return metrics

    except Exception as e:
        log.error(f"Error evaluating {architecture} - {model_name}: {str(e)}")
        return None



def run_testing(cfg, class_names=["Ocean", "Ice"]):
    """
    Run comprehensive testing across all model architectures.

    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Model architectures and their base paths
    base_path = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/model_outputs"
    architectures = {
        "ViT": os.path.join(base_path, "ViT"),
        "Unet": os.path.join(base_path, "Unet"),
        "DeepLabV3": os.path.join(base_path, "DeepLabV3"),
        "FPN": os.path.join(base_path, "FPN"),
        "DinoV3": os.path.join(base_path, "DinoV3")
    }
    
    # test dataset loading
    parent_dir = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/ICE-BENCH"
    test_datasets = IceDataset.create_test_datasets(parent_dir)
    test_dataset = list(test_datasets.values())[0]
    
    log.info(f"Loaded unified test dataset with {len(test_dataset)} samples from all satellites")

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,  #cfg.training.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    log.info(f"Test dataset size: {len(test_dataset)}")
    log.info(f"Number of test batches: {len(test_loader)}")
    
    # Store all results
    all_results = []
    failed_models = []
    
    # Iterate through each architecture
    for arch_name, arch_path in architectures.items():
        log.info(f"\n{'='*40}")
        log.info(f"Testing {arch_name} models")
        log.info(f"{'='*40}")
        
        if not os.path.exists(arch_path):
            log.warning(f"Path does not exist: {arch_path}")
            continue
            
        #specify specific paths TODO: change this perhaps?
        model_files = []
        expected_patterns = [
            f"{arch_name}_bs32_correct_labels_latest_epoch.pth",
            f"{arch_name}_bs32_correct_labels_best_loss.pth", 
            f"{arch_name}_bs32_correct_labels_best_iou.pth"
        ]
        for pattern in expected_patterns:
            if os.path.exists(os.path.join(arch_path, pattern)):
                model_files.append(pattern)
        
        if not model_files:
            log.warning(f"No model files found in {arch_path}")
            continue
            
        log.info(f"Found {len(model_files)} model files: {model_files}")
        
        # Test each model in the architecture
        for model_file in sorted(model_files):
            model_path = os.path.join(arch_path, model_file)
            model_name = os.path.splitext(model_file)[0]
            
            metrics = evaluate_single_model(
                model_path, test_loader, device, cfg, model_name, arch_name
            )
            
            if metrics is not None:
                all_results.append(metrics)
            else:
                failed_models.append(f"{arch_name}/{model_name}")
                
            # Clean up GPU memory after each model
            torch.cuda.empty_cache()
            gc.collect()
                
    del test_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create comprehensive results summary
    if all_results:
        log.info(f"Processing {len(all_results)} results...")
        
    
        # Define output directory
        output_dir = Path("/home/users/amorgan/benchmark_CB_AM/visualisation_panels")
        output_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
        
        # Define output files with full paths
        all_detailed_csv = output_dir / f"detailed_model_comparison.csv"
        all_summary_csv = output_dir / f"architecture_summary.csv"
        
        try:
            # Create DataFrame with proper handling of numpy arrays
            results_df = pd.DataFrame(all_results)
            
            # Save detailed results with explicit error handling
            log.info(f"Saving detailed results to: {all_detailed_csv}")
            results_df.to_csv(all_detailed_csv, index=False)
            log.info(f"✓ Detailed results saved successfully to '{all_detailed_csv}'")
            
            # Create summary statistics
            summary_stats = []
            for arch in architectures.keys():
                arch_results = results_df[results_df['architecture'] == arch]
                if len(arch_results) > 0:
                    summary_stats.append({
                        'architecture': arch,
                        'num_models': len(arch_results),
                        'mean_iou_avg': arch_results['mean_iou'].mean(),
                        'mean_iou_std': arch_results['mean_iou'].std(),
                        'pixel_accuracy_avg': arch_results['pixel_accuracy'].mean(),
                        'pixel_accuracy_std': arch_results['pixel_accuracy'].std(),
                        'mean_f1_avg': arch_results['mean_f1'].mean(),
                        'mean_f1_std': arch_results['mean_f1'].std(),
                        'best_model': arch_results.loc[arch_results['mean_iou'].idxmax(), 'model_name'],
                        'best_iou': arch_results['mean_iou'].max()
                    })
        
            # Save summary with error handling
            if summary_stats:
                summary_df = pd.DataFrame(summary_stats)
                log.info(f"Saving summary results to: {all_summary_csv}")
                summary_df.to_csv(all_summary_csv, index=False)
                log.info(f"✓ Summary results saved successfully to '{all_summary_csv}'")

                # Print summary to console
                print("\n" + "="*80)
                print("ARCHITECTURE COMPARISON SUMMARY")
                print("="*80)
                print(summary_df.round(4))
            else:
                log.warning("No summary statistics to save")
                    
        except Exception as e:
            log.error(f"Error saving CSV files: {str(e)}")
            log.info("Attempting to save with alternative method...")
            
            try:
                # Alternative save method with explicit encoding
                results_df.to_csv(str(all_detailed_csv), index=False, encoding='utf-8')
                log.info(f"✓ Alternative save successful for detailed results")
                
                if summary_stats:
                    summary_df = pd.DataFrame(summary_stats)
                    summary_df.to_csv(str(all_summary_csv), index=False, encoding='utf-8')
                    log.info(f"✓ Alternative save successful for summary results")
                    
            except Exception as e2:
                log.error(f"Alternative save method also failed: {str(e2)}")
                log.info("Printing results to console instead...")
                
                print("\n" + "="*80)
                print("DETAILED RESULTS (CSV save failed)")
                print("="*80)
                print(results_df.to_string())
                
                
        # Print best model overall (if we have results)
        if len(results_df) > 0:
            best_model_idx = results_df['mean_iou'].idxmax()
            best_model = results_df.loc[best_model_idx]
            print(f"\nBEST MODEL OVERALL:")
            print(f"Architecture: {best_model['architecture']}")
            print(f"Model: {best_model['model_name']}")
            print(f"Mean IoU: {best_model['mean_iou']:.4f}")
            print(f"Pixel Accuracy: {best_model['pixel_accuracy']:.4f}")
            print(f"Mean F1: {best_model['mean_f1']:.4f}")
            
            # Print class-wise performance for best model
            print(f"\nCLASS-WISE PERFORMANCE (Best Model):")
            class_ious = best_model['class_ious']
            precision_per_class = best_model['precision_per_class']
            recall_per_class = best_model['recall_per_class']
            f1_per_class = best_model['f1_per_class']
            
            for i, class_name in enumerate(class_names):
                print(f"{class_name}:")
                print(f"  IoU: {class_ious[i]:.4f}")
                print(f"  Precision: {precision_per_class[i]:.4f}")
                print(f"  Recall: {recall_per_class[i]:.4f}")
                print(f"  F1: {f1_per_class[i]:.4f}")
    else:
        log.warning("No results to save - all model evaluations failed")
    
    # Report failed models
    if failed_models:
        log.warning(f"Failed to evaluate {len(failed_models)} models:")
        for failed_model in failed_models:
            log.warning(f"  - {failed_model}")
    #     # Print summary to console
    #     print("\n" + "="*80)
    #     print("ARCHITECTURE COMPARISON SUMMARY")
    #     print("="*80)
    #     print(summary_df.round(4))
        
    #     # Print best model overall
    #     best_model_idx = results_df['mean_iou'].idxmax()
    #     best_model = results_df.loc[best_model_idx]
    #     print(f"\nBEST MODEL OVERALL:")
    #     print(f"Architecture: {best_model['architecture']}")
    #     print(f"Model: {best_model['model_name']}")
    #     print(f"Mean IoU: {best_model['mean_iou']:.4f}")
    #     print(f"Pixel Accuracy: {best_model['pixel_accuracy']:.4f}")
    #     print(f"Mean F1: {best_model['mean_f1']:.4f}")
        
    #     # Print class-wise performance for best model
    #     print(f"\nCLASS-WISE PERFORMANCE (Best Model):")
    #     for i, class_name in enumerate(class_names):
    #         print(f"{class_name}:")
    #         print(f"  IoU: {best_model['class_ious'][i]:.4f}")
    #         print(f"  Precision: {best_model['precision_per_class'][i]:.4f}")
    #         print(f"  Recall: {best_model['recall_per_class'][i]:.4f}")
    #         print(f"  F1: {best_model['f1_per_class'][i]:.4f}")
    
    # # Report failed models
    # if failed_models:
    #     log.warning(f"Failed to evaluate {len(failed_models)} models:")
    #     for failed_model in failed_models:
    #         log.warning(f"  - {failed_model}")
    
    # log.info("\nTesting completed!")
    
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    """
    Main function to run the comprehensive model testing.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # # Initialize wandb if needed (optional)
    # if cfg.get('use_wandb', False):
    #     wandb.init(project="ice-segmentation-evaluation", config=cfg)
    
    # Run comprehensive testing
    run_testing(cfg, class_names=["Ocean", "Ice"])
    
    # # Finish wandb if initialized
    # if cfg.get('use_wandb', False):
    #     wandb.finish()


if __name__ == "__main__":
    main()