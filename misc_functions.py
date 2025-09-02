"""


"""


import random
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


def init_wandb(cfg: DictConfig):
    if wandb.run is None:
        wandb.init(
            project="ICE-BENCH",
            # name="trial_initial_runs",
            entity="amy-morgan-university-of-oxford",
            settings=wandb.Settings(start_method="thread"),
            job_type="training",
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
    cfg: DictConfig,
    log: logging.Logger,
):
    # Save model checkpoint with more information
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
            "val_loss": val_loss,
            "config": cfg,
        },
        path,
    )
