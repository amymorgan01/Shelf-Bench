
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
from pathlib import Path

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
            entity="amy-morgan-university-of-oxford",
            settings=wandb.Settings(start_method="thread"),
            job_type="training",
            id=job_id,
            resume="allow",
            save_code=False,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    sweep_cfg = dict(wandb.config)

    flat_map = {
        # training params
        "learning_rate":    ("training", "learning_rate"),
        "batch_size":       ("training", "batch_size"),
        "weight_decay":     ("training", "weight_decay"),
        "loss_function":    ("training", "loss_function"),
        "optimizer":        ("training", "optimizer"),
        # model params
        "model_name":       ("model", "name"),
        "freeze_backbone":  ("model", "freeze_backbone"),
        "encoder_name":     ("model", "encoder_name"),
    }

    for sweep_key, (section, cfg_key) in flat_map.items():
        if sweep_key in sweep_cfg:
            try:
                OmegaConf.update(cfg, f"{section}.{cfg_key}", sweep_cfg[sweep_key])
                print(f"Sweep override: {section}.{cfg_key} = {sweep_cfg[sweep_key]}")
            except Exception as e:
                print(f"Warning: could not set {section}.{cfg_key}: {e}")



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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Make state dict CPU-safe
        model_state = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        }

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_loss": float(val_loss),
            "best_val_iou": float(val_iou),
            "cfg": OmegaConf.to_container(cfg, resolve=True),  # plain dict, not DictConfig
        }

        torch.save(checkpoint, str(path))
        print(f"Saved model to {path}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params in state_dict: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

    except Exception:
        print(f"ERROR: failed to save checkpoint to {path}")
        raise