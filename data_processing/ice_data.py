"""
DataLoader for SHELF-BENCH

"""

import os
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from pathlib import Path


class IceDataset(Dataset):

    def __init__(self, mode, parent_dir, satellite=None, augment=True):
        print(f"Initializing IceDataset in {mode} mode...")

        self.mode = mode
        self.parent_dir = parent_dir
        self.satellite = satellite

        if augment is None:
            self.augment = mode == "train"
        else:
            self.augment = augment

        if mode == "test":
            # Test data is in ICE-BENCH subdirectory
            base_path = os.path.dirname(os.path.dirname(parent_dir))  # Go up two levels from preprocessed_data/train
            self.image_dir = os.path.join(
                base_path, "ICE-BENCH", "preprocessed_data", "test", "images"
            )
            self.mask_dir = os.path.join(
                base_path, "ICE-BENCH", "preprocessed_data", "test", "masks"
            )

        else:
            # train/val
            self.image_dir = os.path.join(parent_dir, mode, "images")
            self.mask_dir = os.path.join(parent_dir, mode, "masks")

        # Verify directories exist
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Mask directory does not exist: {self.mask_dir}")

        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith((".png"))]
        )
        self.mask_files = sorted(
            [f for f in os.listdir(self.mask_dir) if f.endswith((".png"))]
        )

        print(f"Found {len(self.image_files)} images and {len(self.mask_files)} masks")
        print(f"{mode} {satellite or ''}: {len(self.image_files)} samples")

        if len(self.image_files) != len(self.mask_files):
            print(
                f"Warning: Mismatch in number of images ({len(self.image_files)}) and masks ({len(self.mask_files)})"
            )
            # Take the minimum to avoid index errors
            min_count = min(len(self.image_files), len(self.mask_files))
            self.image_files = self.image_files[:min_count]
            self.mask_files = self.mask_files[:min_count]
            print(f"Using {min_count} image-mask pairs")

        # Data augmentation transforms (only applied during training)
        if self.augment:
            self.transform = A.Compose(
                [
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=0.2, contrast_limit=0.2, p=1.0
                            ),
                            A.GaussNoise(),  # var_limit=(10.0, 50.0), p=1.0),
                        ],
                        p=0.3,
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
                ]
            )
        else:
            # No transforms for validation
            self.transform = A.Compose([])

        self.normalize = A.Normalize(mean=0.3047126829624176, std=0.32187142968177795)

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Loads a SAR image, and its corresponding label
        """
        image_name = self.image_files[idx]
        mask_name = self.mask_files[idx]

        # Load image with OpenCV and convert to PIL Image
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image and mask
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image_np is None:
            raise ValueError(f"Could not load image: {image_path}")
        if mask_np is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        transformed = self.transform(image=image_np, mask=mask_np)
        image_transformed, mask_transformed = transformed["image"], transformed["mask"]
        image_normalized = self.normalize(image=image_transformed)["image"]

        image_tensor = torch.from_numpy(image_normalized).float().unsqueeze(0)

        mask_tensor = torch.from_numpy(mask_transformed).float().unsqueeze(0)

        return image_tensor, mask_tensor

    @staticmethod
    def create_test_datasets(parent_dir):
        """
        Create test dataset
        """

        expected_test_dir = os.path.join(parent_dir, "preprocessed_data", "test")
        expected_images_dir = os.path.join(expected_test_dir, "images")
        expected_masks_dir = os.path.join(expected_test_dir, "masks")

        # print(f"Debug: Looking for test data at:")
        # print(f"  Images: {expected_images_dir}")
        # print(f"  Masks: {expected_masks_dir}")
        # print(f"  Images exists: {os.path.exists(expected_images_dir)}")
        # print(f"  Masks exists: {os.path.exists(expected_masks_dir)}")

        if os.path.exists(expected_images_dir):
            image_files = os.listdir(expected_images_dir)
            png_files = [f for f in image_files if f.endswith(".png")]
            print(f"  All files in images: {len(image_files)}")
            print(f"  PNG files in images: {len(png_files)}")
            if len(image_files) > 0:
                print(f"  Sample files: {image_files[:5]}")

        if os.path.exists(expected_masks_dir):
            mask_files = os.listdir(expected_masks_dir)
            png_mask_files = [f for f in mask_files if f.endswith(".png")]
            print(f"  All files in masks: {len(mask_files)}")
            print(f"  PNG files in masks: {len(png_mask_files)}")

        test_datasets = {}

        try:
            # Create a single test dataset
            print("Creating test dataset...")
            test_dataset = IceDataset("test", parent_dir, satellite=None, augment=False)

            # Only create satellite references if the dataset has data
            if len(test_dataset) > 0:
                satellites = ["ERS", "Envisat", "Sentinel-1"]
                for satellite in satellites:
                    test_datasets[satellite] = test_dataset
                print(
                    f"Successfully created test dataset with {len(test_dataset)} samples"
                )
            else:
                print("Warning: Test dataset is empty")

        except Exception as e:
            print(f"Could not load test dataset: {e}")
            import traceback

            traceback.print_exc()

        return test_datasets
