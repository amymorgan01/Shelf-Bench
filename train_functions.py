import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import torch.optim as optim
import logging
import wandb
from omegaconf import DictConfig


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
        image = image.to(device)
        mask = mask.to(device)

        # TODO: Check if the mask really is inverted
        mask = 1 - (mask / 255)
        mask = F.one_hot(mask.long(), num_classes=2).squeeze(1).permute(0, 3, 1, 2)

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

    epoch_loss = running_loss / len(train_loader)
    log.info(f"Train Epoch: {epoch+1} Average Loss: {epoch_loss:.6f}")

    return epoch_loss

def validate_with_metrics(model, val_loader, loss_function, device, cfg, log, epoch=None):
    """
    Enhanced validation function that calculates comprehensive metrics.
    """
    model.eval()
    val_loss = 0.0
    num_classes = cfg.model.classes
    
    # Initialize metric accumulators
    total_precision = torch.zeros(num_classes).to(device)
    total_recall = torch.zeros(num_classes).to(device)
    total_f1 = torch.zeros(num_classes).to(device)
    total_class_ious = torch.zeros(num_classes).to(device)
    
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            loss = loss_function(outputs, masks)
            val_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate metrics for this batch
            batch_precision, batch_recall, batch_f1 = calculate_metrics(
                masks, preds, num_classes, device
            )
            batch_class_ious, batch_mean_iou = calculate_iou_metrics(
                masks, preds, num_classes, device
            )
            
            # Accumulate metrics
            total_precision += torch.tensor(batch_precision).to(device)
            total_recall += torch.tensor(batch_recall).to(device)
            total_f1 += torch.tensor(batch_f1).to(device)
            total_class_ious += torch.tensor(batch_class_ious).to(device)
            
            num_batches += 1
    
    # Calculate average metrics
    avg_val_loss = val_loss / len(val_loader)
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches
    avg_class_ious = total_class_ious / num_batches
    mean_iou = avg_class_ious.mean().item()
    
    # Log detailed metrics
    if epoch is not None:
        log.info(f"Epoch {epoch + 1} Validation Results:")
        log.info(f"  Loss: {avg_val_loss:.4f}")
        log.info(f"  Mean IoU: {mean_iou:.4f}")
        log.info(f"  Mean Precision: {avg_precision.mean().item():.4f}")
        log.info(f"  Mean Recall: {avg_recall.mean().item():.4f}")
        log.info(f"  Mean F1: {avg_f1.mean().item():.4f}")
        
        # Log class-wise metrics if class names are available
        if hasattr(cfg, 'class_names') and cfg.class_names:
            for i, class_name in enumerate(cfg.class_names):
                log.info(f"  {class_name} - IoU: {avg_class_ious[i]:.4f}, "
                        f"Precision: {avg_precision[i]:.4f}, "
                        f"Recall: {avg_recall[i]:.4f}, F1: {avg_f1[i]:.4f}")
    
    metrics = {
        'val_loss': avg_val_loss,
        'mean_iou': mean_iou,
        'mean_precision': avg_precision.mean().item(),
        'mean_recall': avg_recall.mean().item(),
        'mean_f1': avg_f1.mean().item(),
        'class_ious': avg_class_ious.cpu().numpy(),
        'precision_per_class': avg_precision.cpu().numpy(),
        'recall_per_class': avg_recall.cpu().numpy(),
        'f1_per_class': avg_f1.cpu().numpy()
    }
    
    return metrics

# Old working val function w/o metrics - keep for reference

# def validate(
#     model: nn.Module,
#     val_loader: DataLoader,
#     loss_function: nn.Module,
#     device: torch.device,
#     cfg: DictConfig,
#     log: logging.Logger,
#     epoch: int,
# ) -> float:
#     model.eval()
#     val_loss = 0.0

#     with torch.no_grad():
#         for image, mask in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
#             image = image.to(device)
#             mask = mask.to(device)

#             mask = 1 - (mask / 255)
#             mask = F.one_hot(mask.long(), num_classes=2).squeeze(1).permute(0, 3, 1, 2)

#             prediction = model(image)

#             loss = loss_function(prediction, mask)
#             val_loss += loss.item()

#     val_loss /= len(val_loader)

#     log.info(f"Validation Epoch: {epoch+1} Average Loss: {val_loss:.6f}")

#     return val_loss
