import os
import torch
import gc
import wandb
import hydra
import logging
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score



log = logging.getLogger(__name__)


def calculate_metrics(targets, predictions, num_classes, device):
    """
    Calculate precision, recall, and F1 score for each class.
    Args:
        targets: Ground truth masks (B, H, W)
        predictions: Predicted masks (B, H, W)
        num_classes: Number of classes
        device: Device to run calculations on
    Returns:
        precision, recall, f1 arrays for each class
    """
    precision = torch.zeros(num_classes, device=device)
    recall = torch.zeros(num_classes, device=device)
    f1 = torch.zeros(num_classes, device=device)
    
    for cls in range(num_classes):
        pred_cls = (predictions == cls)
        target_cls = (targets == cls)
        
        # True positives, false positives, false negatives
        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()
        
        # Calculate metrics with epsilon to avoid division by zero
        eps = 1e-8
        precision[cls] = tp / (tp + fp + eps)
        recall[cls] = tp / (tp + fn + eps)
        f1[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls] + eps)
    
    return precision.cpu().numpy(), recall.cpu().numpy(), f1.cpu().numpy()


def calculate_iou_metrics(targets, predictions, num_classes, device):
    """
    Calculate IoU for each class and mean IoU.
    Args:
        targets: Ground truth masks (B, H, W)
        predictions: Predicted masks (B, H, W)
        num_classes: Number of classes
        device: Device to run calculations on
    Returns:
        class_ious: IoU for each class
        mean_iou: Mean IoU across all classes
    """
    class_ious = torch.zeros(num_classes, device=device)
    
    for cls in range(num_classes):
        pred_cls = (predictions == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        # Calculate IoU with epsilon to avoid division by zero
        iou = intersection / (union + 1e-8)
        class_ious[cls] = iou
    
    mean_iou = class_ious.mean()
    return class_ious.cpu().numpy(), mean_iou.item()





def evaluate_model(model_path, val_loader, device, cfg, log):
    """
    Comprehensive model evaluation function.
    Args:
        model_path: Path to the trained model checkpoint
        val_loader: Validation data loader
        device: Device to run evaluation on
        cfg: Configuration object
        log: Logger instance
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    log.info(f"Loading model from {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = load_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Initialize metrics
    num_classes = cfg.model.classes
    total_precision = torch.zeros(num_classes).to(device)
    total_recall = torch.zeros(num_classes).to(device)
    total_f1 = torch.zeros(num_classes).to(device)
    total_class_ious = torch.zeros(num_classes).to(device)
    total_loss = 0.0
    
    loss_function = get_loss_function(cfg)
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            loss = loss_function(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate metrics
            batch_precision, batch_recall, batch_f1 = calculate_metrics(
                masks, preds, num_classes, device
            )
            batch_class_ious, _ = calculate_iou_metrics(
                masks, preds, num_classes, device
            )
            
            # Accumulate metrics
            total_precision += torch.tensor(batch_precision).to(device)
            total_recall += torch.tensor(batch_recall).to(device)
            total_f1 += torch.tensor(batch_f1).to(device)
            total_class_ious += torch.tensor(batch_class_ious).to(device)
            
            num_batches += 1
    
    # Calculate final metrics
    avg_loss = total_loss / len(val_loader)
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches
    avg_class_ious = total_class_ious / num_batches
    mean_iou = avg_class_ious.mean().item()
    
    # Create comprehensive metrics dictionary
    metrics = {
        "loss": avg_loss,
        "mean_iou": mean_iou,
        "class_ious": avg_class_ious.cpu().numpy(),
        "mean_precision": avg_precision.mean().item(),
        "mean_recall": avg_recall.mean().item(),
        "mean_f1": avg_f1.mean().item(),
        "precision_per_class": avg_precision.cpu().numpy(),
        "recall_per_class": avg_recall.cpu().numpy(),
        "f1_per_class": avg_f1.cpu().numpy(),
    }
    
    return metrics
