import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import torch.optim as optim
import logging
import wandb
from omegaconf import DictConfig
from typing import Optional
import gc
from metrics import accumulate_confusion_matrix, calculate_metrics_from_confusion_matrix
from metrics import accumulate_iou_components, calculate_iou_from_components


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    log: logging.Logger,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    for batch_idx, (image, mask) in enumerate(
        tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")
    ):
        optimizer.zero_grad(set_to_none=True)
        image = image.to(device)
        mask = mask.to(device)

        mask = mask / 255
        # mask = F.one_hot(mask.long(), num_classes=2).squeeze(1).permute(0, 3, 1, 2)   old line
        mask = F.one_hot(mask.long(), num_classes=cfg.model.classes).squeeze(1).permute(0, 3, 1, 2).float()

        optimizer.zero_grad()

        prediction = model(image)

        loss = loss_function(prediction, mask)
        loss.backward()
        optimizer.step()
        if cfg.get("use_wandb", False):
            wandb.log({"train_loss": loss.item()})

        running_loss += loss.item()

        if batch_idx % cfg["training"].get("log_interval", 10) == 0:
            log.info(
                f"Train Epoch: {epoch+1} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    epoch_loss = running_loss / len(train_loader)
    log.info(f"Train Epoch: {epoch+1} Average Loss: {epoch_loss:.6f}")

    return epoch_loss


def validate_with_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    cfg: DictConfig,
    log: logging.Logger,
    epoch: Optional[int] = None,
):
    """
    Enhanced validation function that calculates comprehensive metrics.
    """
    model.eval()
    val_loss = 0.0
    num_classes = cfg.model.classes
    
    # Initialize confusion matrix components for proper metric calculation
    confusion_matrix = {
        'tp': torch.zeros(num_classes, device=device),
        'fp': torch.zeros(num_classes, device=device),
        'fn': torch.zeros(num_classes, device=device),
        'tn': torch.zeros(num_classes, device=device)
    }
    
    # Initialize IoU components
    iou_components = {
        'intersection': torch.zeros(num_classes, device=device),
        'union': torch.zeros(num_classes, device=device)
    }
    
    # Initialize pixel accuracy components
    total_correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            # CONSISTENT with training: Always normalize to [0,1] then convert to class indices
            if masks.max() > 1:
                masks = masks / 255.0
            
            # Convert to class indices (0 or 1 for binary segmentation)
            # For binary segmentation with values in [0,1], round to nearest class
            masks_for_metrics = torch.round(masks).long()           #change jan 2026
            # masks_for_metrics = (masks * (num_classes - 1)).long()
            
            # Convert to one-hot encoding for loss calculation
            one_hot_masks = (
                F.one_hot(masks_for_metrics, num_classes=num_classes)
                .squeeze(1)
                .permute(0, 3, 1, 2)
                .float()
            )

            outputs = model(images)

            loss = loss_function(outputs, one_hot_masks)
            val_loss += loss.item()

            # Get predictions (convert back to class indices)
            preds = torch.argmax(outputs, dim=1)

            # Accumulate confusion matrix components
            confusion_matrix = accumulate_confusion_matrix(
                masks_for_metrics, preds, num_classes, confusion_matrix
            )
            
            # Accumulate IoU components
            iou_components = accumulate_iou_components(
                masks_for_metrics, preds, num_classes, iou_components
            )
            
            # Accumulate pixel accuracy
            total_correct_pixels += (masks_for_metrics == preds).sum().item()
            total_pixels += masks_for_metrics.numel()

    # Calculate final metrics from accumulated components
    avg_val_loss = val_loss / len(val_loader)
    
    precision, recall, f1 = calculate_metrics_from_confusion_matrix(
        confusion_matrix, num_classes, device
    )
    
    class_ious, mean_iou = calculate_iou_from_components(
        iou_components, num_classes, device
    )
    
    pixel_accuracy = total_correct_pixels / total_pixels if total_pixels > 0 else 0.0

    # Log detailed metrics
    if epoch is not None:
        log.info(f"Epoch {epoch + 1} Validation Results:")
        log.info(f"  Loss: {avg_val_loss:.4f}")
        log.info(f"  Mean IoU: {mean_iou:.4f}")
        log.info(f"  Pixel Accuracy: {pixel_accuracy:.4f}")
        log.info(f"  Mean Precision: {precision.mean():.4f}")
        log.info(f"  Mean Recall: {recall.mean():.4f}")
        log.info(f"  Mean F1: {f1.mean():.4f}")
        
        # Log class-wise metrics if class names are available
        if hasattr(cfg, "class_names") and cfg.class_names:
            for i, class_name in enumerate(cfg.class_names):
                log.info(
                    f"  {class_name} - IoU: {class_ious[i]:.4f}, "
                    f"Precision: {precision[i]:.4f}, "
                    f"Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}"
                )

    metrics = {
        "val_loss": avg_val_loss,
        "val_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "mean_precision": precision.mean(),
        "mean_recall": recall.mean(),
        "mean_f1": f1.mean(),
        "class_ious": class_ious,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
    }
    
    return metrics