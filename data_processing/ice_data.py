"""
DataLoader for ICE-Bench

"""

from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import tqdm
import numpy as np
import albumentations as A
import pandas as pd


class IceDataset(Dataset):
    """"
   
    This class loads a dataset of SAR images and their corresponding masks produced by C. Baumhoer.

    Args:
        - mode (str): 'train', 'val', or 'test'
        - parent_dir (str): Root directory containing 'images', 'masks' folders.
        - augment: whether to apply data augmentation (true for train, false for val/test)
    """

    def __init__(self, mode, parent_dir, augment=True):
        print(f"Initializing IceDataset in {mode} mode...")
        
        self.mode = mode
        self.parent_dir = parent_dir

        if augment is None:
            self.augment = (mode == 'train')
        else:
            self.augment = augment

        self.image_dir = os.path.join(parent_dir, mode, "images")
        self.mask_dir = os.path.join(parent_dir, mode, "masks")

        # Get all image files (lazy loading - just get count and verify structure)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                  if f.endswith(('.png'))])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) 
                                 if f.endswith(('.png'))])
        
        print(f"Found {len(self.image_files)} images and {len(self.mask_files)} masks")
        
        if len(self.image_files) != len(self.mask_files):
            print(f"Warning: Mismatch in number of images ({len(self.image_files)}) and masks ({len(self.mask_files)})")
            # Take the minimum to avoid index errors
            min_count = min(len(self.image_files), len(self.mask_files))
            self.image_files = self.image_files[:min_count]
            self.mask_files = self.mask_files[:min_count]
            print(f"Using {min_count} image-mask pairs")

        # Data augmentation transforms (only applied during training)
        if self.augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                ], p=0.3),
                # Optional: Add more SAR-specific augmentations
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            ])
        else:
            # No transforms for validation
            self.transform = A.Compose([])
        

       # TODO: Check this 
        # Separate normalization for image only (after joint transformations)
        self.normalize = A.Normalize(mean=0.3047126829624176, std=0.32187142968177795)
       
    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """"
        Loads a SAR image, and its corresponding label
        """
        image_name = self.image_files[idx]
        mask_name = self.mask_files[idx]
        
        # Load image with OpenCV and convert to PIL Image
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir,mask_name)

        # Load image and mask
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image_np is None:
            raise ValueError(f"Could not load image: {image_path}")
        if mask_np is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        #TODO: up to here (2.7.25), check tensor and normalisation things


        # Apply same spatial transformations to both augmentation
        transformed = self.transform(image=image_np, mask=mask_np)
        image_transformed, mask_transformed = transformed['image'], transformed['mask']
        
        #TODO: I have already normalised the images so decide whether to keep
        # Apply normalization only to image
        image_normalized = self.normalize(image=image_transformed)['image']

        image_tensor = torch.from_numpy(image_normalized).float().unsqueeze(0)

        # mask already black and white
        mask_tensor = torch.from_numpy(mask_transformed).float().unsqueeze(0)

        return image_tensor, mask_tensor