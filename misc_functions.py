
import random
from typing import Optional
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import logging
import torch.nn as nn
import torch.optim as optim
import wandb


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_wandb(cfg: DictConfig, job_id: Optional[str] = None):
    if wandb.run is None:
        wandb.init(
            project="Shelf-Bench", 
            # name="trial_initial_runs",
            entity="amy-morgan-university-of-oxford", #change to your wandb username
            settings=wandb.Settings(start_method="thread"),
            job_type="training",
            id=job_id,
            resume="allow",
            save_code=False,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        # Update config with sweep parameters
        if wandb.config is not None:
            # Update training parameters
            for key, value in wandb.config.training.items():
                if hasattr(cfg.training, key):
                    setattr(cfg.training, key, value)

            # Update model parameters
            for key, value in wandb.config.model.items():
                if hasattr(cfg.model, key):
                    setattr(cfg.model, key, value)

            # Update device
            if hasattr(wandb.config, "device"):
                cfg.device = wandb.config.device


def save_model(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    val_iou: float,
    cfg: DictConfig,
):
    """
    Save full model state including decoder and segmentation head.
    The encoder weights are not saved since they remain frozen with pretrained weights.
    """
    
    # Save the FULL model state (decoder + segmentation head)
    # This works for all model types
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),  # Save everything
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
        "best_val_loss": val_loss,
        "best_val_iou": val_iou,
        "config": cfg,
    }
    
    torch.save(checkpoint, path)
    
    # Log what we saved
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Saved model to {path}")
    print(f"  Total params in state_dict: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    