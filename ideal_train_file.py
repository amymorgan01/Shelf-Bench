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
from train_functions import train_one_epoch, validate

""" 
HYDRA IMPLEMENTATION
"""
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Print the configurationprint("MAIN FUNCTION STARTED")
    print("Config keys:", list(cfg.keys()))
    print("Training config:", cfg.training)
    print("Model config:", cfg.model)

    # Set random seed
    set_seed(cfg["seed"])

    # TODO: it was signing me in as Jowan so removed it
    # init_wandb(cfg)
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

    # Get data loaders
    train_loader, val_loader = get_data_loaders(cfg)
    log.info("After DataLoader creation")

    # Load the model
    model = load_model(cfg, device)

    # Load loss function, optimizer, and scheduler
    loss_function = get_loss_function(cfg)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    best_val_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):

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

        val_loss = validate(
            model, val_loader, loss_function, device, cfg, log, epoch=epoch
        )
        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        if cfg.get("use_wandb", False):
            wandb.log({"train_loss": train_loss, "epoch": epoch})

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(cfg["save_dir"], "best_model.pth")
            save_model(path, model, optimizer, scheduler, epoch, val_loss, cfg, log)

        if epoch % cfg.save_freq == 0:
            path = os.path.join(cfg["save_dir"], f"model_epoch_{epoch}.pth")
            save_model(path, model, optimizer, scheduler, epoch, val_loss, cfg, log)

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
