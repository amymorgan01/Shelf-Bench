"""
sweep_train.py — W&B Sweep entry point for Shelf-BENCH.
1. Register the sweep with W&B (run once — gives you a SWEEP_ID)
wandb sweep sweep_config.yaml --project SET_PROJECT_NAME
2. Start an agent on each node/GPU
wandb agent <YOUR_USERNAME>/SET_PROJECT_NAME/<SWEEP_ID>
To run multiple agents in parallel (one per GPU):
CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID> &
CUDA_VISIBLE_DEVICES=1 wandb agent <SWEEP_ID> &
"""

import os
import torch
import gc
import wandb
import logging
from omegaconf import OmegaConf, DictConfig
from paths import ROOT_GWS

from misc_functions import set_seed, save_model
from load_functions import (
    get_data_loaders,
    load_model,
    get_optimizer,
    get_scheduler,
    get_loss_function,
)
from train_functions import train_one_epoch, validate_with_metrics
from metrics import evaluate_model

log = logging.getLogger(__name__)

SWEEP_BASE_SAVE_DIR = str(ROOT_GWS / "benchmark_data_CB" / "model_outputs" / "wandb_sweeps")


def get_sweep_model_paths(cfg: DictConfig, run) -> dict:

    sweep_id  = run.sweep_id  
    run_id    = run.id     
    model_name = cfg.model.name  

    model_dir = os.path.join(SWEEP_BASE_SAVE_DIR, sweep_id, model_name)
    os.makedirs(model_dir, exist_ok=True)

    prefix = f"{model_name}_run_{run_id}"
    return {
        "model_dir":  model_dir,
        "best_loss":  os.path.join(model_dir, f"{prefix}_best_loss.pth"),
        "best_iou":   os.path.join(model_dir, f"{prefix}_best_iou.pth"),
    }


def load_base_config(config_path: str = "conf/config.yaml") -> DictConfig:
    cfg = OmegaConf.load(config_path)
    return cfg


def override_cfg_with_sweep(cfg: DictConfig) -> DictConfig:

    sweep_cfg = wandb.config 


    if "model_name" in sweep_cfg:         
        cfg.model.name = sweep_cfg.model_name
    if "freeze_backbone" in sweep_cfg:
        cfg.model.freeze_backbone = sweep_cfg.freeze_backbone
    if "epochs" in sweep_cfg:
        cfg.training.epochs = int(sweep_cfg.epochs)
    if "learning_rate" in sweep_cfg:
        cfg.training.learning_rate = sweep_cfg.learning_rate
    if "weight_decay" in sweep_cfg:
        cfg.training.weight_decay = sweep_cfg.weight_decay
    if "batch_size" in sweep_cfg:
        cfg.training.batch_size = sweep_cfg.batch_size
    if "loss_function" in sweep_cfg:
        cfg.training.loss_function = sweep_cfg.loss_function
    if "optimizer" in sweep_cfg:
        cfg.training.optimizer = sweep_cfg.optimizer

    return cfg


def train():
    """Single sweep run — called once per agent trial."""

    cfg = load_base_config("conf/config.yaml")
    run = wandb.init(project="ice_bench_seg_sweep")
    cfg = override_cfg_with_sweep(cfg)

    
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    print(f"Using device: {device}")

  
    paths = get_sweep_model_paths(cfg, run)
    print(f"Sweep outputs → {paths['model_dir']}")
    print(f"  best_loss : {paths['best_loss']}")
    print(f"  best_iou  : {paths['best_iou']}")


    wandb.config.update(
        {"save_dir": paths["model_dir"], "sweep_id": run.sweep_id},
        allow_val_change=True,
    )


    train_loader, val_loader = get_data_loaders(cfg)


    model = load_model(cfg, device)
    loss_function = get_loss_function(cfg)
    if hasattr(loss_function, "to"):
        loss_function = loss_function.to(device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    best_val_loss = float("inf")
    best_val_iou = -1.0   
    early_stopping_patience = cfg.get("early_stopping_patience", None)
    epochs_without_improvement = 0

    for epoch in range(cfg["training"]["epochs"]):
        print(f"\n{'='*10} Epoch {epoch+1}/{cfg['training']['epochs']} {'='*10}")

        train_loss = train_one_epoch(
            model, train_loader, loss_function, optimizer, device, cfg, log, epoch=epoch
        )

        val_metrics = validate_with_metrics(
            model, val_loader, loss_function, device, cfg, log, epoch=epoch
        )
        print(f"[DEBUG] raw val_loss: {val_metrics['val_loss']}, val_iou: {val_metrics['val_iou']}")


        val_loss = float(val_metrics["val_loss"])
        val_iou  = float(val_metrics["val_iou"])

        if not torch.isfinite(torch.tensor(val_loss)):
            raise ValueError(f"Non-finite val_loss at epoch {epoch+1}: {val_loss}")
        if not torch.isfinite(torch.tensor(val_iou)):
            raise ValueError(f"Non-finite val_iou at epoch {epoch+1}: {val_iou}")

        if scheduler is not None:
            scheduler.step()

  
        wandb.log({
            "epoch":              epoch,
            "train_loss":         train_loss,
            "val_loss":           val_loss,
            "val_iou":            val_iou,
            "val_mean_f1":        val_metrics["mean_f1"],
            "val_mean_precision": val_metrics["mean_precision"],
            "val_mean_recall":    val_metrics["mean_recall"],
        })


        improved = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
            save_model(
                paths["best_loss"],
                model, optimizer, scheduler,
                epoch, best_val_loss, best_val_iou, cfg,
            )
            print(f"  ✓ NEW BEST LOSS {val_loss:.4f} → {paths['best_loss']}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            improved = True
            save_model(
                paths["best_iou"],
                model, optimizer, scheduler,
                epoch, best_val_loss, best_val_iou, cfg,
            )
            print(f"  ✓ NEW BEST IoU  {val_iou:.4f} → {paths['best_iou']}")

        # Early stopping 
        if early_stopping_patience is not None:
            if not improved:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0

            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        torch.cuda.empty_cache()
        gc.collect()

    # Summary metrics visible in sweep dashboard
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["best_val_iou"]  = best_val_iou
    wandb.summary["best_loss_path"] = paths["best_loss"]
    wandb.summary["best_iou_path"]  = paths["best_iou"]

    run.finish()


if __name__ == "__main__":
    train()