"""Main Train file for Shelf-BENCH, including HYDRA IMPLEMENTATION and wandb logging.

To run all models: uv run continuous_train.py -m model.name=Unet,FPN,ViT,DeepLabV3 other parameters...

"""

import os
import torch
import gc
import wandb
import hydra
import logging
from omegaconf import DictConfig
from misc_functions import set_seed, init_wandb, save_model
from load_functions import (
    get_data_loaders,
    load_model,
    load_full_model_state,  # Changed from update_segmentation_head_weights_only
    get_optimizer,
    get_scheduler,
    get_loss_function,
)
from train_functions import train_one_epoch, validate_with_metrics
from metrics import evaluate_model
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image

log = logging.getLogger(__name__)

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# continuous wandb training from previous runs
# model_run_ids = {
#     "FPN": "f9nhdf5r",
#     "Unet": "tsab9f1v",
#     "DeepLabV3": "isst5072",
#     "DinoV3": "1ycmbstp",
#     "ViT": "g2gdkxkp"
#     }

# training from new wandb run
model_run_ids = None

def log_visualisations_to_wandb(model, val_loader, device, num_samples=2, denormalise=True):
    """Log sample predictions to wandb for visualization."""
    try:
        model.eval()
        images_to_log = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']

                # Get predictions
                preds = torch.argmax(outputs, dim=1)

                # Move to CPU
                images = images[:num_samples].cpu()
                masks = masks[:num_samples].cpu()
                preds = preds[:num_samples].cpu()

                for i in range(min(num_samples, len(images))):
                    img = images[i]
                    mask = masks[i]
                    pred = preds[i]

                    # Denormalise image
                    if denormalise:
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img = img * std + mean
                        img = torch.clamp(img, 0.0, 1.0)

                    # Convert to numpy (H, W, C) format for wandb
                    img_np = img.permute(1, 2, 0).numpy()
                    
                    # Remove channel dimension if present and convert to uint8
                    mask_np = mask.squeeze().numpy() if mask.dim() == 3 else mask.numpy()
                    pred_np = pred.squeeze().numpy() if pred.dim() == 3 else pred.numpy()
                    
                    # Normalize masks to 0-255 range for visualization
                    mask_np = (mask_np * 255 / max(mask_np.max(), 1)).astype(np.uint8)
                    pred_np = (pred_np * 255 / max(pred_np.max(), 1)).astype(np.uint8)
                    
                    # Convert grayscale masks to RGB for easier viewing
                    mask_rgb = np.stack([mask_np]*3, axis=-1)
                    pred_rgb = np.stack([pred_np]*3, axis=-1)

                    # Log images
                    images_to_log.append(wandb.Image(img_np, caption=f"Sample {i+1}: Image"))
                    images_to_log.append(wandb.Image(mask_rgb, caption=f"Sample {i+1}: Ground Truth"))
                    images_to_log.append(wandb.Image(pred_rgb, caption=f"Sample {i+1}: Prediction"))

                break  # Only one batch
        
        model.train()
        return images_to_log
        
    except Exception as e:
        log.error(f"Visualization error: {e}", exc_info=True)
        model.train()
        return []


# continuous wandb training from previous runs
# model_run_ids = {
#     "FPN": "f9nhdf5r",
#     "Unet": "tsab9f1v",
#     "DeepLabV3": "isst5072",
#     "DinoV3": "1ycmbstp",
#     "ViT": "g2gdkxkp"
#     }

# training from new wandb run
model_run_ids = None

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    print("Config keys:", list(cfg.keys()))
    print("Training config:", cfg.training)
    print("Model config:", cfg.model)

    # Set random seed
    set_seed(cfg["seed"])
    job_id = None
    if model_run_ids is not None:
        job_id = model_run_ids.get(cfg.model.name)

    init_wandb(cfg, job_id=job_id)
    print("wandb init done")
    print(60*"=")
    print(f"model name: {cfg.model.name}, job_id: {job_id}")
    print(60*"=")

    # Force CUDA device if available
    if torch.cuda.is_available():
        cfg.device = "cuda"
        torch.cuda.empty_cache()
    else:
        print("WARNING: CUDA not available, using CPU")
        cfg.device = "cpu"
        

    # Set device
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # save models
    base_save_dir = cfg.save_dir
    model_specific_dir = os.path.join(base_save_dir, cfg.model.name)
    os.makedirs(model_specific_dir, exist_ok=True)
    # save models with specific names
    model_name_prefix = f"{cfg['model']['name']}_retrain_090226_debug"
    best_loss_model_path = os.path.abspath(
        os.path.join(model_specific_dir, f"{model_name_prefix}_best_loss.pth")
    )
    best_iou_model_path = os.path.abspath(
        os.path.join(model_specific_dir, f"{model_name_prefix}_best_iou.pth")
    )
    metric_path = os.path.abspath(
        os.path.join(model_specific_dir, f"{model_name_prefix}_metrics.csv")
    )

    checkpoint_path = os.path.abspath(
        os.path.join(model_specific_dir, f"{model_name_prefix}_latest_epoch.pth")
    )
    print(f"Does checkpoint exist? {os.path.exists(checkpoint_path)}")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(cfg)
    log.info("After DataLoader creation")

    # Load the model
    print("Loading model...")
    model = load_model(cfg, device)
    model = model.to(device)
   
    # Load loss function, optimizer, and scheduler
    loss_function = get_loss_function(cfg)
    if hasattr(loss_function, 'to'):
        loss_function = loss_function.to(device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    # Check if checkpoint exists and load if specified
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_iou = 0.0
    epochs_without_improvement = 0
    early_stopping_patience = cfg.get("early_stopping_patience", None)
    early_stopping_metric = cfg.get("early_stopping_metric", "val_loss")  # or "val_iou"
    
   
    if cfg.get("load_path", False) and os.path.exists(checkpoint_path):
        log.info(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        
        print(f"Checkpoint keys: {list(checkpoint.keys()) if checkpoint else 'None'}")
        
        # Load FULL model state (decoder + segmentation head)
        model = load_full_model_state(model, checkpoint["model_state_dict"], cfg["model"]["name"])
        
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif scheduler is None:
            log.info("Scheduler is None - skipping scheduler state loading")
            
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_val_iou = checkpoint.get("best_val_iou", 0.0)
        
        log.info(f"Full model state loaded successfully. Resuming from epoch {start_epoch}")
        print(f"Full model state loaded successfully. Resuming from epoch {start_epoch}")
    else:
        log.info("No valid load_path specified or file does not exist. Training from scratch.")
        
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        print(f"\n{'='*10} Epoch {epoch + 1}/{cfg['training']['epochs']} {'='*10}")
        print(f"DEBUG: start_epoch={start_epoch}, training.epochs={cfg['training']['epochs']}")
        
        # Train one epoch
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
            cfg,
            log,
            epoch=epoch,
        )
        print(f"train_one_epoch returned successfully. Loss: {train_loss:.4f}")
        print("About to call validate_with_metrics...")

        val_metrics = validate_with_metrics(
            model, val_loader, loss_function, device, cfg, log, epoch=epoch
        )
        print(f"validate_with_metrics returned successfully.")
        val_loss = val_metrics["val_loss"]
        val_iou = val_metrics["val_iou"]
        print(f"Epoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val IoU: {val_iou:.4f}")
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        if cfg.get("use_wandb", False):
            wandb_metrics = {
                "epoch": epoch,
                "train_epoch_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_iou": val_iou,
                "val_mean_precision": val_metrics["mean_precision"],
                "val_mean_recall": val_metrics["mean_recall"],
                "val_mean_f1": val_metrics["mean_f1"],
            }
            
            # Add per-class metrics if available
            if "class_ious" in val_metrics and cfg.get("class_names"):
                class_ious = val_metrics["class_ious"]
                class_names = cfg["class_names"]
                for i, class_name in enumerate(class_names):
                    wandb_metrics[f"val_iou_{class_name}"] = class_ious[i].item() if hasattr(class_ious[i], 'item') else class_ious[i]
                    if "precision_per_class" in val_metrics:
                        wandb_metrics[f"val_precision_{class_name}"] = val_metrics["precision_per_class"][i]
                    if "recall_per_class" in val_metrics:
                        wandb_metrics[f"val_recall_{class_name}"] = val_metrics["recall_per_class"][i]
            
            wandb.log(wandb_metrics)
            
            # Log visualisations every 15 epochs
            if (epoch + 1) % 15 == 0:
                print(f"Logging visualisations to wandb for epoch {epoch + 1}...")
                try:
                    log.info(f"Starting visualization logging for epoch {epoch + 1}")
                    vis_images = log_visualisations_to_wandb(
                        model, val_loader, device, 
                        num_samples=2, 
                        denormalise=True 
                    )
                    
                    if vis_images:
                        log.info(f"Successfully created {len(vis_images)} visualization images, logging to wandb...")
                        wandb.log({"validation_predictions": vis_images}, step=epoch)
                        print(f"Successfully logged {len(vis_images)} visualisation images to wandb")
                    else:
                        log.warning("No visualization images were created")
                        print("Warning: No visualization images were created")
                    
                    # Clean up
                    del vis_images
                    torch.cuda.empty_cache()
                except Exception as e:
                    log.error(f"Failed to log visualisations: {e}", exc_info=True)
                    print(f"Error: Failed to log visualisations: {e}")
                    # Don't fail the entire training loop for visualization issues
            
        if os.path.exists(metric_path):
            with open(metric_path, "a") as f:
                f.write(
                    f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{val_iou:.4f}\n"
                )
        else:
            with open(metric_path, "w") as f:
                f.write("epoch,train_loss,val_loss,val_iou\n")
                f.write(
                    f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{val_iou:.4f}\n"
                )

        # Track saved a best model this epoch
        saved_best_model = False
        
        # Check and save ONLY if new best loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset early stopping counter
            print(f" NEW BEST LOSS! Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            save_model(
                best_loss_model_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                best_val_iou,
                cfg,
            )
            print(f"Best loss model saved/updated: {best_loss_model_path}")
            saved_best_model = True

        # Check and save ONLY if new best IoU  
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_without_improvement = 0  # Reset early stopping counter
            print(f" NEW BEST IoU! Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            save_model(
                best_iou_model_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                best_val_iou,
                cfg,
            )
            print(f"Best IoU model saved/updated: {best_iou_model_path}")
            saved_best_model = True

        save_model(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            best_val_iou,
            cfg,
        )
        print(f" Latest checkpoint updated: {checkpoint_path}")

        # Print saving summary for this epoch
        if saved_best_model:
            print(" This epoch produced a new best model!")
        else:
            print("No new best model this epoch (normal)")

        # Early stopping check
        if early_stopping_patience is not None:
            if not saved_best_model:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")
            
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING TRIGGERED!")
                print(f"No improvement in {early_stopping_metric} for {early_stopping_patience} epochs")
                print(f"Best {early_stopping_metric}: {best_val_loss if early_stopping_metric == 'val_loss' else best_val_iou:.4f}")
                print(f"{'='*60}")
                break

        torch.cuda.empty_cache()
        gc.collect()
        print(f"EPOCH {epoch + 1} COMPLETED SUCCESSFULLY")
        print(f"About to continue to next epoch...")

    # Final evaluation
    print(f"\n{'='*20} Final Evaluation {'='*20}")

    if os.path.exists(best_loss_model_path):
        print("Evaluating best loss model...")
        best_loss_metrics = evaluate_model(
            best_loss_model_path, val_loader, device, cfg, log
        )

    # Evaluate best IoU model
    if os.path.exists(best_iou_model_path):
        print("\nEvaluating best IoU model...")
        best_iou_metrics = evaluate_model(
            best_iou_model_path, val_loader, device, cfg, log
        )

    # Final wandb logging
    if cfg.get("use_wandb", False):
        final_wandb_metrics = {
            "final_best_val_loss": best_val_loss,
            "final_best_val_iou": best_val_iou,
            "total_epochs_trained": cfg["training"]["epochs"],
        }

        if "best_loss_metrics" in locals():
            final_wandb_metrics.update(
                {
                    f"final_best_loss_model_{k}": v
                    for k, v in best_loss_metrics.items()
                    if isinstance(v, (int, float))  # Only log scalar values to wandb
                }
            )

        if "best_iou_metrics" in locals():
            final_wandb_metrics.update(
                {
                    f"final_best_iou_model_{k}": v
                    for k, v in best_iou_metrics.items()
                    if isinstance(v, (int, float))  # Only log scalar values to wandb
                }
            )

        wandb.log(final_wandb_metrics)
        wandb.finish()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY:")
    print("=" * 60)
    print(f"Total epochs trained: {cfg['training']['epochs']}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Best loss model saved at: {best_loss_model_path}")
    print(f"Best IoU model saved at: {best_iou_model_path}")
    print("=" * 60)

    return best_val_loss, best_val_iou


if __name__ == "__main__":
    main()