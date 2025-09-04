""" Main Train file for ICE-BENCH, including HYDRA IMPLEMENTATION

To run all models: uv run ideal_train_file.py -m model.name=Unet,FPN,ViT,DeepLabV3 other parameters...

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
    get_optimizer,
    get_scheduler,
    get_loss_function,
)
from train_functions import train_one_epoch, validate_with_metrics
from metrics import calculate_metrics, calculate_iou_metrics, evaluate_model, 


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Print the configurationprint("MAIN FUNCTION STARTED")
    print("Config keys:", list(cfg.keys()))
    print("Training config:", cfg.training)
    print("Model config:", cfg.model)

    # Set random seed
    set_seed(cfg["seed"])

    # TODO: sort out wandb 
    init_wandb(cfg)
    # Initialize wandb with sweep configuration

    print("wandb init done")

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
    os.makedirs(cfg["save_dir"], exist_ok=True)
    # save models with specific names
    model_name_prefix = f"{cfg['model']['name']}_bs{cfg['training']['batch_size']}"
    best_loss_model_path = os.path.join(cfg["save_dir"], f"{model_name_prefix}_best_loss.pth")
    best_iou_model_path = os.path.join(cfg["save_dir"], f"{model_name_prefix}_best_iou.pth")
    checkpoint_path = os.path.join(cfg["save_dir"], f"{model_name_prefix}_latest_epoch.pth")


    # Get data loaders
    train_loader, val_loader = get_data_loaders(cfg)
    log.info("After DataLoader creation")

    # Load the model
    model = load_model(cfg, device)

    # Load loss function, optimizer, and scheduler
    loss_function = get_loss_function(cfg)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

     # Check if checkpoint exists and load if specified
    start_epoch = 0
    if cfg.get("load_path", False) and os.path.exists(checkpoint_path):
        log.info(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1  # Resume from next epoch
        log.info(f"Model weights loaded successfully. Resuming from epoch {start_epoch}")
        print(f"Model weights loaded successfully. Resuming from epoch {start_epoch}")
    else:
        log.info(
            "No valid load_path specified or file does not exist. Training from scratch."
        )

    best_val_loss = float("inf")
    best_val_iou = 0.0

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        print(f"\n{'='*10} Epoch {epoch + 1}/{cfg['training']['epochs']} {'='*10}")

        # Train one epoch
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
            cfg,  # used to be class_names
            log,
            epoch=epoch,
        )

        val_metrics = validate_with_metrics(
            model, val_loader, loss_function, device, cfg, log, epoch=epoch
        )

        val_loss = val_metrics['val_loss']
        val_iou = val_metrics['val_iou']

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        if cfg.get("use_wandb", False):
            wandb_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_iou": val_iou,
                "val_mean_precision": val_metrics['mean_precision'],
                "val_mean_recall": val_metrics['mean_recall'],
                "val_mean_f1": val_metrics['mean_f1']
            }
            
            # # Add class-wise metrics if class names are available
            # if hasattr(cfg, 'class_names') and cfg.class_names:
            #     for i, class_name in enumerate(cfg.class_names):
            #         wandb_metrics[f"val_iou_{class_name}"] = val_metrics['class_ious'][i]
            #         wandb_metrics[f"val_precision_{class_name}"] = val_metrics['precision_per_class'][i]
            #         wandb_metrics[f"val_recall_{class_name}"] = val_metrics['recall_per_class'][i]
            #         wandb_metrics[f"val_f1_{class_name}"] = val_metrics['f1_per_class'][i]
            
            wandb.log(wandb_metrics)
        


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best loss! Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            save_model(best_loss_model_path, model, optimizer, scheduler, epoch, val_loss, val_iou, cfg, log)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            print(f"New best IoU! Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            save_model(best_iou_model_path, model, optimizer, scheduler, epoch, val_loss, val_iou, cfg, log)

        # Periodic checkpoints
        if epoch % cfg.get("save_freq", 10) == 0:
            checkpoint_save_path = os.path.join(cfg["save_dir"], f"model_epoch_{epoch}.pth")
            save_model(checkpoint_save_path, model, optimizer, scheduler, epoch, val_loss, val_iou, cfg, log)
            print(f"Checkpoint saved at epoch {epoch}")

        # Always save latest
        save_model(checkpoint_path, model, optimizer, scheduler, epoch, val_loss, val_iou, cfg, log)


        torch.cuda.empty_cache()
        gc.collect()

        # Final evaluation
    print(f"\n{'='*20} Final Evaluation {'='*20}")
    
    if os.path.exists(best_loss_model_path):
        print("Evaluating best loss model...")
        best_loss_metrics = evaluate_model(best_loss_model_path, val_loader, device, cfg, log)
        
        print(f"Best Loss Model Final Metrics:")
        print(f"  Loss: {best_loss_metrics['loss']:.4f}")
        print(f"  Mean IoU: {best_loss_metrics['mean_iou']:.4f}")
        print(f"  Mean Precision: {best_loss_metrics['mean_precision']:.4f}")
        print(f"  Mean Recall: {best_loss_metrics['mean_recall']:.4f}")
        print(f"  Mean F1: {best_loss_metrics['mean_f1']:.4f}")

    # Evaluate best IoU model
    if os.path.exists(best_iou_model_path):
        print("\nEvaluating best IoU model...")
        best_iou_metrics = evaluate_model(best_iou_model_path, val_loader, device, cfg, log)
        
        print(f"Best IoU Model Final Metrics:")
        print(f"  Loss: {best_iou_metrics['loss']:.4f}")
        print(f"  Mean IoU: {best_iou_metrics['mean_iou']:.4f}")
        print(f"  Mean Precision: {best_iou_metrics['mean_precision']:.4f}")
        print(f"  Mean Recall: {best_iou_metrics['mean_recall']:.4f}")
        print(f"  Mean F1: {best_iou_metrics['mean_f1']:.4f}")

    # Final wandb logging
    if cfg.get("use_wandb", False):
        final_wandb_metrics = {
            "final_best_val_loss": best_val_loss,
            "final_best_val_iou": best_val_iou,
            "total_epochs_trained": cfg["training"]["epochs"],
        }
        
        if 'best_loss_metrics' in locals():
            final_wandb_metrics.update({
                f"final_best_loss_model_{k}": v for k, v in best_loss_metrics.items()
                if isinstance(v, (int, float))  # Only log scalar values to wandb
            })
        
        if 'best_iou_metrics' in locals():
            final_wandb_metrics.update({
                f"final_best_iou_model_{k}": v for k, v in best_iou_metrics.items()
                if isinstance(v, (int, float))  # Only log scalar values to wandb
            })
        
        wandb.log(final_wandb_metrics)
        wandb.finish()


    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY:")
    print("="*60)
    print(f"Total epochs trained: {cfg['training']['epochs']}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Best loss model saved at: {best_loss_model_path}")
    print(f"Best IoU model saved at: {best_iou_model_path}")
    print("="*60)

    return best_val_loss, best_val_iou


if __name__ == "__main__":
    main()