import os
import torch
import torch.nn.functional as F
import gc
import wandb
import hydra
import logging
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from load_functions import load_model, get_loss_function, load_full_model_state

log = logging.getLogger(__name__)

def accumulate_confusion_matrix(targets, predictions, num_classes, confusion_matrix):

    for cls in range(num_classes):
        pred_cls = predictions == cls
        target_cls = targets == cls

        # True positives, false positives, false negatives, true negatives
        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()
        tn = (~pred_cls & ~target_cls).sum().float()

        confusion_matrix['tp'][cls] += tp
        confusion_matrix['fp'][cls] += fp
        confusion_matrix['fn'][cls] += fn
        confusion_matrix['tn'][cls] += tn
    
    return confusion_matrix


def calculate_metrics_from_confusion_matrix(confusion_matrix, num_classes, device):
    """
    Calculate precision, recall, and F1 score from confusion matrix.
    """
    precision = torch.zeros(num_classes, device=device)
    recall = torch.zeros(num_classes, device=device)
    f1 = torch.zeros(num_classes, device=device)

    eps = 1e-8
    
    for cls in range(num_classes):
        tp = confusion_matrix['tp'][cls]
        fp = confusion_matrix['fp'][cls]
        fn = confusion_matrix['fn'][cls]
        
        precision[cls] = tp / (tp + fp + eps)
        recall[cls] = tp / (tp + fn + eps)
        f1[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls] + eps)

    return precision.cpu().numpy(), recall.cpu().numpy(), f1.cpu().numpy()


def accumulate_iou_components(targets, predictions, num_classes, iou_components):
    """
    IoU calculation.
    """
    for cls in range(num_classes):
        pred_cls = predictions == cls
        target_cls = targets == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        iou_components['intersection'][cls] += intersection
        iou_components['union'][cls] += union
    
    return iou_components


def calculate_iou_from_components(iou_components, num_classes, device):
    """
    Calculate IoU for each class and mean IoU 
    """
    class_ious = torch.zeros(num_classes, device=device)
    eps = 1e-8
    
    for cls in range(num_classes):
        intersection = iou_components['intersection'][cls]
        union = iou_components['union'][cls]
        class_ious[cls] = intersection / (union + eps)
    
    mean_iou = class_ious.mean()
    return class_ious.cpu().numpy(), mean_iou.item()


def evaluate_model(model_path, val_loader, device, cfg, log):
    """
    model evaluation function
    """
    log.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = load_model(cfg, device)
    model = load_full_model_state(model, checkpoint["model_state_dict"], cfg["model"]["name"])
    model.eval()

    num_classes = cfg.model.classes

    confusion_matrix = {
        'tp': torch.zeros(num_classes, device=device),
        'fp': torch.zeros(num_classes, device=device),
        'fn': torch.zeros(num_classes, device=device),
        'tn': torch.zeros(num_classes, device=device)
    }

    iou_components = {
        'intersection': torch.zeros(num_classes, device=device),
        'union': torch.zeros(num_classes, device=device)
    }
    
    total_loss = 0.0
    total_correct_pixels = 0
    total_pixels = 0

    loss_function = get_loss_function(cfg)
    if hasattr(loss_function, 'to'):
        loss_function = loss_function.to(device)

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # CONSISTENT with training: Always normalize to [0,1] then convert to class indices
            if masks.max() > 1:
                masks = masks / 255.0
            
            # Convert to class indices
            masks_for_metrics = (masks * (num_classes - 1)).long()
            
            # Convert to one-hot for loss
            one_hot_masks = (
                F.one_hot(masks_for_metrics, num_classes=num_classes)
                .squeeze(1)
                .permute(0, 3, 1, 2)
                .float()
            )

            outputs = model(images)
            # previously 
            # loss = loss_function(outputs, masks)
            loss = loss_function(outputs, one_hot_masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            confusion_matrix = accumulate_confusion_matrix(
                masks_for_metrics, preds, num_classes, confusion_matrix
            )
            
            iou_components = accumulate_iou_components(
                masks_for_metrics, preds, num_classes, iou_components
            )
            
            total_correct_pixels += (masks_for_metrics == preds).sum().item()
            total_pixels += masks_for_metrics.numel()

    avg_loss = total_loss / len(val_loader)
    
    precision, recall, f1 = calculate_metrics_from_confusion_matrix(
        confusion_matrix, num_classes, device
    )
    
    class_ious, mean_iou = calculate_iou_from_components(
        iou_components, num_classes, device
    )
    
    pixel_accuracy = total_correct_pixels / total_pixels if total_pixels > 0 else 0.0

    metrics = {
        "loss": avg_loss,
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "class_ious": class_ious,
        "mean_precision": np.mean(precision),
        "mean_recall": np.mean(recall),
        "mean_f1": np.mean(f1),
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
    }

    return metrics

def calculate_pixel_accuracy(targets, predictions):
    """
    Calculate pixel-wise accuracy.
    """
    correct_pixels = (targets == predictions).sum().float()
    total_pixels = targets.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy.item()