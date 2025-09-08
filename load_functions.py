
""""
Loading function for ICE-BENCH: trainloader, valloader, models loaded, optimisers and schedulers

"""

import os
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from data_processing.ice_data import IceDataset
import torch.nn as nn
import torch.optim as optim
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from combined_loss import CombinedLoss
from models.ViT import create_vit_large_16
from models.DinoV3 import DINOv3SegmentationModel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from omegaconf import DictConfig
from typing import Tuple


def get_data_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    parent_dir = cfg["data"]["parent_dir"]

    # Load datasets
    train_dataset = IceDataset(mode="train", parent_dir=parent_dir, augment=True)
    val_dataset = IceDataset(mode="val", parent_dir=parent_dir, augment=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        # persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        # persistent_workers=True,
    )

    return train_loader, val_loader


def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    model_name = cfg["model"]["name"]
    in_channels = cfg["model"]["in_channels"]
    classes = cfg["model"]["classes"]

    if model_name == "Unet":
        encoder_name = cfg["model"]["encoder_name"]
        encoder_weights = cfg["model"]["encoder_weights"]
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == "FPN":
        encoder_name = cfg["model"]["encoder_name"]
        encoder_weights = cfg["model"]["encoder_weights"]
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    elif model_name == "DeepLabV3":
        encoder_name = cfg["model"]["encoder_name"]
        encoder_weights = cfg["model"]["encoder_weights"]
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    elif model_name == "ViT":
        img_size = cfg["model"]["img_size"]
        #pretrained_path = cfg["model"]["pretrained_path"]
        model = create_vit_large_16(
            num_classes=classes, 
            img_size=img_size, 
            use_pretrained=True,
            in_channels=in_channels
            #pretrained_path=pretrained_path
        )
        encoder_name = "ViT-Large"

       
        #TODO: check this works
    elif model_name == "DinoV3":
        img_size = cfg["model"]["img_size"]
        satellite_weights_path = cfg["model"]["satellite_weights_path"]
        segmentation_head = cfg["model"]["segmentation_head"]
        freeze_backbone = cfg["model"]["freeze_backbone"]

        model = DINOv3SegmentationModel(
            num_classes=classes,
            img_size=img_size,
            satellite_weights_path=satellite_weights_path,
            segmentation_head=segmentation_head,
            freeze_backbone=freeze_backbone
            )
        encoder_name = "DINOv3-ViT-L/16"

    else:
        raise ValueError(f"Model {model_name} not recognized.")

    model = model.to(device)
    print(f"Model {model_name} with {encoder_name} loaded on {device}.")
    return model


def get_loss_function(cfg: DictConfig) -> nn.Module:
    loss_name = cfg["training"]["loss_function"]
    if loss_name == "DiceLoss":
        return DiceLoss(to_onehot_y=False, softmax=True)
    elif loss_name == "DiceCELoss":
        return DiceCELoss(to_onehot_y=False, softmax=True)
    elif loss_name == "FocalLoss":
        return FocalLoss(to_onehot_y=False, softmax=True)
    elif loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss_name == "CombinedLoss":
        return CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    else:
        raise ValueError(f"Loss function {loss_name} not recognized.")


def get_optimizer(cfg: DictConfig, model: nn.Module) -> optim.Optimizer:
    optimizer_name = cfg["training"]["optimizer"]
    learning_rate = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))

    if optimizer_name == "Adam":
        return optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        momentum = cfg["training"].get("momentum", 0.9)
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")


def get_scheduler(cfg: DictConfig, optimizer: optim.Optimizer):
    scheduler_name = cfg["training"].get("scheduler", None)
    if scheduler_name == "CosineAnnealingWarmRestarts":
        T_0 = float(cfg["training"].get("T_0", 10))
        T_mult = float(cfg["training"].get("T_mult", 1))
        eta_min = float(cfg["training"].get("eta_min", 1e-6))
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    elif scheduler_name is None:
        return None
    else:
        raise ValueError(f"Scheduler {scheduler_name} not recognized.")
