"""
Calculate mean and standard deviation of training dataset for normalization.

This script:
1. Loads all training images from preprocessed data
2. Normalizes them to [0-1] range (matches how the z-score normalization expects data)
3. Calculates dataset-wide mean and std
4. Outputs values for use in ice_data.py

Usage:
    python calculate_dataset_statistics.py --data_dir /path/to/preprocessed_data
"""

import os
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from argparse import ArgumentParser
from pathlib import Path
import tqdm


def calculate_statistics(data_dir, split="train"):
    """
    Calculate mean and standard deviation of all images in a split.
    
    Args:
        data_dir: Path to preprocessed_data directory
        split: "train" or "val"
    
    Returns:
        mean (float): Mean pixel value [0-1]
        std (float): Standard deviation of pixel values [0-1]
        total_pixels (int): Total number of pixels processed
    """
    
    image_dir = os.path.join(data_dir, split, "images")
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return None, None, None
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".png", ".tif"))])
    print(f"Found {len(image_files)} images in {split} split")
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return None, None, None
    
    # Method 1: Load all images and calculate statistics
    print(f"\nCalculating statistics from {split} data...")
    all_pixels = []
    
    for img_name in tqdm.tqdm(image_files, desc=f"Processing {split} images"):
        image_path = os.path.join(image_dir, img_name)
        
        # Load as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not load {img_name}")
            continue
        
        # Convert to float and normalize to [0-1]
        image_normalized = image.astype(np.float32) / 255.0
        
        # Collect pixels (flatten)
        all_pixels.append(image_normalized.flatten())
    
    # Concatenate all pixels
    all_pixels = np.concatenate(all_pixels)
    total_pixels = len(all_pixels)
    
    # Calculate mean and std
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)
    
    print(f"\n{'='*60}")
    print(f"Statistics for {split} split:")
    print(f"{'='*60}")
    print(f"Total pixels processed: {total_pixels:,}")
    print(f"Mean:                   {mean:.16f}")
    print(f"Std Dev:                {std:.16f}")
    print(f"Min pixel value:        {np.min(all_pixels):.6f}")
    print(f"Max pixel value:        {np.max(all_pixels):.6f}")
    print(f"25th percentile:        {np.percentile(all_pixels, 25):.6f}")
    print(f"Median (50th):          {np.percentile(all_pixels, 50):.6f}")
    print(f"75th percentile:        {np.percentile(all_pixels, 75):.6f}")
    
    return mean, std, total_pixels


def calculate_statistics_torch(data_dir, split="train"):
    """
    Alternative method using PyTorch tensors (like your original script).
    
    Args:
        data_dir: Path to preprocessed_data directory
        split: "train" or "val"
    
    Returns:
        mean (float): Mean pixel value [0-1]
        std (float): Standard deviation of pixel values [0-1]
    """
    
    image_dir = os.path.join(data_dir, split, "images")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".png", ".tif"))])
    
    print(f"PyTorch method: Processing {len(image_files)} images...")
    list_imgs = []
    
    for img_name in tqdm.tqdm(image_files, desc=f"Loading {split} images"):
        image = cv2.imread(os.path.join(image_dir, img_name), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            continue
        
        # Convert to [0-1] float tensor
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        
        # Flatten and append
        list_imgs.append(torch.flatten(image_tensor))
    
    # Concatenate all
    torch_imgs = torch.cat(list_imgs, dim=0)
    
    mean = torch.mean(torch_imgs)
    std = torch.std(torch_imgs)
    
    print(f"\n{'='*60}")
    print(f"Statistics (PyTorch method) for {split} split:")
    print(f"{'='*60}")
    print(f"Total pixels: {len(torch_imgs):,}")
    print(f"Mean:         {float(mean):.16f}")
    print(f"Std Dev:      {float(std):.16f}")
    
    return float(mean), float(std)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/ICE-BENCH/preprocessed_data",
        help="Path to preprocessed_data directory containing train/val splits"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val","test"],
        help="Which split to calculate statistics for"
    )
    parser.add_argument(
        "--method",
        default="numpy",
        choices=["numpy", "torch"],
        help="Calculation method"
    )
    
    args = parser.parse_args()
    
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Method: {args.method}")
    print()
    
    if args.method == "numpy":
        mean, std, total_pixels = calculate_statistics(args.data_dir, args.split)
    else:
        mean, std = calculate_statistics_torch(args.data_dir, args.split)
    
    if mean is not None:
        print(f"\n{'='*60}")
        print("USAGE IN ice_data.py:")
        print(f"{'='*60}")
        print()
        print("Replace line 91 in data_processing/ice_data.py:")
        print()
        print(f"self.normalize = A.Normalize(mean={mean}, std={std})")
        print()
