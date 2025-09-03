"""
quick check to see how models have done
"""
from PIL import Image
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms
import cv2
from data_processing.ice_data import IceDataset


model_path = '/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/model_outputs/FPN_model_epoch_49.pth'
parent_dir = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/preprocessed_data/"  # Update with your data path
output_dir = '/home/users/amorgan/benchmark_CB_AM/figures/'

CLASS_NAMES = ["Other", "Land ice"]
N_CLASSES = len(CLASS_NAMES)
from matplotlib.colors import ListedColormap
COLORMAP = ListedColormap(['blue', 'lightgray'])

# Load  model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_prediction(image, mask, pred, class_names=None, save_path=None, sample_idx=0):
    """
    Visualize the original image, ground truth mask, and prediction
    Args:
        image: normalized tensor image
        mask: ground truth mask tensor
        pred: prediction mask tensor
        class_names: list of class names for legend
        save_path: path to save the visualization
        sample_idx: index of the sample being visualized
    """
    print(f"Sample {sample_idx} - Image shape: {image.shape}")
    print(f"Sample {sample_idx} - Mask shape: {mask.shape}")
    print(f"Sample {sample_idx} - Prediction shape: {pred.shape}")
  
    # Convert tensors to numpy arrays
    image = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    print(f"Sample {sample_idx} - After squeeze - Image: {image.shape}, Mask: {mask.shape}, Pred: {pred.shape}")
    
    # Clip to [0, 1] range
    image = np.clip(image, 0, 1)
    
    # Create figure with better spacing
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Original Image (Sample {sample_idx})', fontsize=14)
    axes[0].axis('off')
    
    # Plot ground truth mask
    im1 = axes[1].imshow(mask, cmap=COLORMAP, vmin=0, vmax=N_CLASSES-1)
    axes[1].set_title('Ground Truth Mask', fontsize=14)
    axes[1].axis('off')
    
    # Plot prediction
    im2 = axes[2].imshow(pred, cmap=COLORMAP, vmin=0, vmax=N_CLASSES-1)
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].axis('off')

    # Add colorbar
    plt.colorbar(im2, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
    
    # Add legend with class names
    if class_names:
        import matplotlib.patches as mpatches
        patches = []
        for i, name in enumerate(class_names):
            color = COLORMAP(i / (N_CLASSES-1))
            patches.append(mpatches.Patch(color=color, label=f'{i}: {name}'))
        fig.legend(handles=patches, loc='lower center', ncol=N_CLASSES, 
                  bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def calculate_class_metrics(pred, mask, num_classes=2):
    """Calculate IoU and accuracy for each class"""
    ious = []
    accuracies = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        mask_cls = (mask == cls)
        
        intersection = (pred_cls & mask_cls).sum()
        union = (pred_cls | mask_cls).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
        
        ious.append(iou)
        
        # Class-specific accuracy
        if mask_cls.sum() > 0:
            acc = (pred_cls & mask_cls).sum() / mask_cls.sum()
        else:
            acc = 1.0 if pred_cls.sum() == 0 else 0.0
        
        accuracies.append(acc)
    
    return ious, accuracies

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model - you may need to adjust architecture parameters
    try:
        model = smp.FPN(
            encoder_name="resnet50", 
            encoder_weights="imagenet",
            in_channels=1,
            classes=2,
        )
        
        # Load the trained weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from checkpoint with 'model_state_dict' key")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded model from checkpoint with 'state_dict' key")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")
            
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please verify your model architecture matches the saved checkpoint")
        return
    
    # Create test dataset
    try:
        val_dataset = IceDataset(mode='val', parent_dir=parent_dir)
        print(f"Test dataset loaded with {len(val_dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Generate predictions for samples
    num_samples = min(10, len(val_dataset))  # Visualize up to 10 samples
    print(f"Generating predictions for {num_samples} test samples...")
    
    # Store metrics for analysis
    all_ious = []
    all_pixel_accuracies = []
    all_class_ious = {i: [] for i in range(N_CLASSES)}
    
    with torch.no_grad():
        for i in range(num_samples):
            try:
                # Get a sample
                image, mask = val_dataset[i]
                
                # Add batch dimension and move to device
                image_batch = image.unsqueeze(0).to(device)
                mask = mask.to(device)
                
                # Get prediction
                output = model(image_batch)
                
                # Convert output probabilities to class predictions
                pred = torch.argmax(output, dim=1).squeeze(0)
                # FIX: Flip the predicted classes (0 becomes 1, 1 becomes 0)
                pred = 1 - pred  # This flips 0->1 and 1->0


                # Move back to CPU for visualization
                pred_np = pred.cpu().numpy()
                mask_np = mask.cpu().numpy()
                
                print(f"Sample {i} - Unique classes in mask: {np.unique(mask_np)}")
                print(f"Sample {i} - Unique classes in pred: {np.unique(pred_np)}")
                
                # Calculate metrics
                pixel_accuracy = (pred_np == mask_np).mean()
                all_pixel_accuracies.append(pixel_accuracy)
                
                # Calculate IoU per class
                class_ious, class_accs = calculate_class_metrics(pred_np, mask_np)
                for cls, iou in enumerate(class_ious):
                    all_class_ious[cls].append(iou)
                
                mean_iou = np.mean(class_ious)
                all_ious.append(mean_iou)
                
                print(f"Sample {i} - IoU: {mean_iou:.4f}, Pixel Acc: {pixel_accuracy:.4f}")
                
                # Visualize every sample (or adjust frequency as needed)
                save_path = os.path.join(output_dir, f'prediction_sample_{i}.png')
                visualize_prediction(image, mask, pred, CLASS_NAMES, save_path, i)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    print(f"Visualizations saved to {output_dir}")
    
    # Print comprehensive metrics
    if all_ious:
        avg_iou = np.mean(all_ious)
        avg_pixel_accuracy = np.mean(all_pixel_accuracies)
        
        print(f"\n=== Overall Metrics ===")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Pixel Accuracy: {avg_pixel_accuracy:.4f}")
        
        print(f"\n=== Per-Class IoU ===")
        for cls, class_name in enumerate(CLASS_NAMES):
            if all_class_ious[cls]:
                cls_iou = np.mean(all_class_ious[cls])
                print(f"{class_name}: {cls_iou:.4f}")
        
        # Plot metrics distribution
        plt.figure(figsize=(15, 10))
        
        # IoU distribution
        plt.subplot(2, 2, 1)
        plt.hist(all_ious, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(avg_iou, color='r', linestyle='--', linewidth=2, 
                   label=f'Mean IoU: {avg_iou:.4f}')
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        plt.title('IoU Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Pixel accuracy distribution
        plt.subplot(2, 2, 2)
        plt.hist(all_pixel_accuracies, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(avg_pixel_accuracy, color='r', linestyle='--', linewidth=2,
                   label=f'Mean Accuracy: {avg_pixel_accuracy:.4f}')
        plt.xlabel('Pixel Accuracy')
        plt.ylabel('Frequency')
        plt.title('Pixel Accuracy Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Per-class IoU comparison
        plt.subplot(2, 2, 3)
        class_mean_ious = [np.mean(all_class_ious[i]) if all_class_ious[i] else 0 
                          for i in range(N_CLASSES)]
        bars = plt.bar(CLASS_NAMES, class_mean_ious, alpha=0.7, color='coral', edgecolor='black')
        plt.ylabel('Mean IoU')
        plt.title('Per-Class IoU Performance')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, class_mean_ious):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # IoU vs Pixel Accuracy scatter
        plt.subplot(2, 2, 4)
        plt.scatter(all_ious, all_pixel_accuracies, alpha=0.6, color='purple')
        plt.xlabel('IoU Score')
        plt.ylabel('Pixel Accuracy')
        plt.title('IoU vs Pixel Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_metrics.png'), dpi=150, bbox_inches='tight')
        print(f"Comprehensive metrics plot saved to {os.path.join(output_dir, 'comprehensive_metrics.png')}")
        plt.close()

if __name__ == "__main__":
    main()
