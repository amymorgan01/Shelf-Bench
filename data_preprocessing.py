"""
Steps involved for preprocessing the Sentinel-1, ERS and Envisat scenes:
1. Set a naming convention: in rename_all_files.py [SAT]_[YYYYMMDD]_[POLARISATION]_[SCENE_ID] 
2. Keep patches at their original res: Envisat and ERS are 30 m, Sentinel-1 is 40 m. 
 - create a val dataset with 10% of the data from the training set
3. Images are greyscale, masks are greyscale
4. Patch the images and masks

Code inspired by CAFFE - Gourmelon et al. 2022

BAND_MAPPING = {
    "Sentinel-1": {
        "HH": 1,  # First band in Sentinel-1 data is HH polarization
        "HV": 2,  # Second band is HV polarization
        "DEM": 3, # Third band is DEM (Digital Elevation Model)
        "RATIO": 4 # Fourth band is HH/HV ratio
    },
    "ERS": {
        "VV": 1,  # First band in ERS data is VV polarization
        "DEM": 2  # Second band is DEM
    },
    "Envisat": {
        "VV": 1,  # First band in Envisat data is VV polarization
        "DEM": 2  # Second band is DEM
    }
}
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
import threading
import rasterio
from pathlib import Path
from paths import ROOT_GWS, ROOT_LOCAL

# set the path to the data
"""Structure of data:

benchmark_data_CB
-- Sentinel-1
------ masks
------ scenes
--------- test_s1
-------------- masks
-------------- scenes
-- ERS
------ masks
------ scenes
------ vectors
--------- test_ERS
-------------- masks
-------------- scenes
-- Envisat
------masks
------ scenes
------ vectors
--------- test_envisat
-------------- masks
-------------- scenes
"""

# OLD all paths
# parent_dir = ROOT_GWS / "benchmark_data_CB"
# S1_dir = os.path.join(parent_dir, "Sentinel-1")
# ERS_dir = os.path.join(parent_dir, "ERS")
# Envisat_dir = os.path.join(parent_dir, "Envisat")

# NEW - JAN 2026, /gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/ICE-BENCH/preprocessed_data
parent_dir = ROOT_GWS / "ICE-BENCH" / "benchmark_data_CB"
S1_dir = os.path.join(parent_dir, "Sentinel-1")
ERS_dir = os.path.join(parent_dir, "ERS")
Envisat_dir = os.path.join(parent_dir, "Envisat")

class SatellitePreprocessor:
    def __init__(self, base_dir, output_dir, patch_size = 256, overlap_train=0, overlap_val =0,
                create_trainval: bool = True, create_test: bool = True):

        """ 
        A class to preprocess satellite data for training and validation.
            base_dir: Path to benchmark_data_CB directory
            output_dir: Path where processed data will be saved
            patch_size: Size of extracted patches (default: 256)
            overlap_train: Overlap for training patches (default: 0)
            overlap_val: Overlap for validation patches (default: 128)
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.overlap_train = overlap_train
        self.overlap_val = overlap_val
       
        self.satellite_dirs = {
            'Sentinel-1': self.base_dir / 'Sentinel-1',
            'ERS': self.base_dir / 'ERS',
            'Envisat': self.base_dir / 'Envisat'
        }

        self._create_output_structure(create_trainval=create_trainval, create_test=create_test)

    def _create_output_structure(self,create_trainval: bool = True, create_test: bool = True):
        """
        Create the output directory structure for processed data. 
        Need to create a val dataset from train
        Updated to match dataloader expectations:
        - train/val: output_dir/train/images, output_dir/train/masks
        - test: output_dir/satellite/test_satellite/scenes, output_dir/satellite/test_satellite/masks

        """
        if create_trainval:
            for split in ['train', 'val']:
                for data_type in ['images', 'masks']:
                    output_path = self.output_dir / split / data_type
                    output_path.mkdir(parents=True, exist_ok=True)
        if create_test:
            for data_type in ['images', 'masks']:
                test_path = self.output_dir / 'test' / data_type
                test_path.mkdir(parents=True, exist_ok=True)

        (self.output_dir /'data_splits').mkdir(exist_ok=True)

        
    def _resize_image(self,image,satellite):
        """
        Option to resize
        """
        if satellite in ['ERS', 'Envisat']:

            scale_factor = 1
            new_height = int(image.shape[0] * scale_factor)
            new_width = int(image.shape[1] * scale_factor)
            resized = cv2.resize(image, (new_width, new_height), interpolation =cv2.INTER_AREA)
            return resized
        elif satellite == 'Sentinel-1':
            # Sentinel-1 is already at 40m resolution
            return image

    def _normalise_mask_vals(self, mask):
        """
        Normalize mask values from 0-1 range to 0-255 range
        Assumes mask is already grayscale (single channel)
        """
        if mask.max() <= 1:
            # Convert from 0-1 to 0-255
            return (mask * 255).astype(np.uint8)
        else:
            # Already in proper range
            return mask.astype(np.uint8)

    def _pad_image(self, image, patch_size, overlap):
        """
        Pad image for patch extraction

        image: input image
        patch_size: size of patches
        overlap: overlap set between patches

        returns: padded image, bottom padding, right padding

        """
        h,w = image.shape[:2]
        stride = patch_size - overlap

        if overlap == 0:
            pad_h = (stride - h % stride) % stride
            pad_w = (stride - w % stride) % stride
        else:
            pad_h = (stride - ((h - patch_size) % stride)) % stride if h > patch_size else patch_size - h
            pad_w = (stride - ((w - patch_size) % stride)) % stride if w > patch_size else patch_size - w
        
    # Apply padding
        if len(image.shape) == 3:
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        else:
            padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        return padded, pad_h, pad_w
    
    def _extract(self, image, patch_size, overlap):
        """
        Extract patches from image using sliding window
        
        returns: array of extracted patches, and coordinates of each patch
        
        """

        stride = patch_size - overlap
        h, w = image.shape[:2]
        patches = []
        coords = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                if len(image.shape) == 3:
                    patch = image[y:y + patch_size, x:x + patch_size, :]
                else:
                    patch = image[y:y + patch_size, x:x + patch_size]
                patches.append(patch)
                coords.append((y, x))
        return np.array(patches), coords
    
    def _convert_to_db(self, input_image):
        input_image = np.maximum(input_image, 1e-10)  # Avoid log(0)
        return 10 * np.log10(input_image)
    
    def _normalise_scale(self, input_image,percentile_clip=True):

        if percentile_clip:
                    # Use percentile clipping to avoid extreme values dominating normalization
            p2, p98 = np.nanpercentile(input_image, [2, 98])
            input_image = np.clip(input_image, p2, p98)
                
        image_max = np.nanmax(input_image)
        image_min = np.nanmin(input_image)
                
                # Avoid division by zero
        if image_max == image_min:
            return input_image
        elif image_max != image_min:
            #return np.zeros_like(input_image)
            out_image = (input_image - image_min) / (image_max - image_min)
            
        return out_image

     
    def _normalise_image(self, image, satellite_type):
        """ 
        Normalise image on individual patches

        """
        if satellite_type in ['Sentinel-1', 'ERS', 'Envisat']:
            # Create a copy to avoid modifying original
            image_copy = image.copy().astype(np.float64)
            
            # Convert the 0 values (background) to nans
            image_copy[image_copy == 0] = np.nan
            
            # Convert to decibel scale
            decibel_image = self._convert_to_db(image_copy)
            
            # Normalise the decibel scale (using patch-specific min/max)
            normalised_image = self._normalise_scale(decibel_image,percentile_clip=True)

            # Convert NaNs back to 0 for background
            normalised_image[np.isnan(normalised_image)] = 0
             
            # Convert to integer values
            normalised_image = (normalised_image * 255).astype(np.uint8)
            return normalised_image
        
        return image


    def _process_satellite_data(self, satellite, split_type, file_pairs, overlap):
        """
        Process satellite imagery for a specific split i.e train or val

        split_type; 'train' or 'val'
        file_pairs: List of (image_path, mask_path) tuples

        Images are tiff, therefore using rasterio
        """

        print(f"Processing {len(file_pairs)} {satellite} files for {split_type}")
        
        for img_path, mask_path in file_pairs:
            try:
                # load image using rasterio
                with rasterio.open(img_path) as src_img:
                    image = src_img.read(1)  # Read first band as grayscale
                    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
                    print(f"Image value range: {image.min():.3f} to {image.max():.3f}")
                    # Check for valid data
                    valid_pixels = image[image != 0]
                    if len(valid_pixels) == 0:
                        print(f"Warning: No valid pixels in {img_path.name}")
                        continue
                    
                    print(f"Valid pixel range: {valid_pixels.min():.3f} to {valid_pixels.max():.3f}")

                with rasterio.open(mask_path) as src_mask:
                    if src_mask.count == 1:
                        mask = src_mask.read(1)
                        
                    else:
                        mask = np.transpose(src_mask.read(), (1, 2, 0))  # (bands, h, w) -> (h, w, bands)
                print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")


                if image is None or mask is None:
                    print(f"Error loading image or mask: {img_path}, {mask_path}")
                    continue
            
                ################################################
                #        Resize to 40m resolution          #
                ################################################
                # image = self._resize_image(image, satellite)
                # mask = self._resize_image(mask, satellite)

                ################################################
                #        Convert mask to greyscale          #
                ################################################
                mask = self._normalise_mask_vals(mask)
                ################################################

                ########################################################
                #        Pad image and mask for patch extraction       #
                ########################################################
                image_padded, pad_h, pad_w = self._pad_image(image, self.patch_size, overlap) # for whole image use image_uint8
                mask_padded, _, _ = self._pad_image(mask, self.patch_size, overlap)

                ########################################################
                #        Extract patches from padded image and mask    #
                ########################################################
                image_patches, coords = self._extract(image_padded, self.patch_size, overlap) #patches from normalised image
                mask_patches, _ = self._extract(mask_padded, self.patch_size, overlap)

                ########################################################
                #        Save patches to output directory          #
                #######################################################    

                for i, (img_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):
                # choose output directories once (satellite-specific for test)
                    if split_type == 'test':
                        img_output_dir = self.output_dir / 'test' / 'images'
                        mask_output_dir = self.output_dir / 'test' / 'masks'
                    else:
                        img_output_dir = self.output_dir / split_type / 'images'
                        mask_output_dir = self.output_dir / split_type / 'masks'

                base_filename = img_path.stem

                for i, (img_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):
                    y,x = coords[i]

                    background_threshold = 1e-6
                    background_mask = np.abs(img_patch) <= background_threshold
                    
                    # Skip patches that are all background or greater than 80% background
                    if np.all(background_mask):
                        continue  # Skip saving this patch
                    if np.mean(background_mask) > 0.8:
                        continue

                    normalised_patch = self._normalise_image(img_patch, satellite)
                    
                    # Additional check after normalization - sometimes normalization can create all-black patches
                    if np.all(normalised_patch == 0) or np.mean(normalised_patch == 0) > 0.8:
                        continue

                    patch_name = f"{base_filename}__{pad_h}_{pad_w}_{i}_{y}_{x}.png"
                    
                    img_output_path = img_output_dir / patch_name
                    cv2.imwrite(str(img_output_path), normalised_patch)
                    
                    # Save mask patch
                    mask_output_path = mask_output_dir / patch_name
                    cv2.imwrite(str(mask_output_path), mask_patch)

                
                print(f"✓ Processed {base_filename}: {len(image_patches)} patches")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    def _get_file_pairs(self, satellite_dir, mode="trainval", satellite_name=None):
        """
        Get paired image and mask files

        returns a list of (image_path, mask_path) tuples
        
        mode: "trainval" (default) or "test"

        """
        
        if mode == "trainval":
            
            scenes_dir = satellite_dir / 'scenes'
            masks_dir = satellite_dir / 'masks'
            
        elif mode == "test":
                # Handle satellite-specific test folder names
            test_folder_map = {
                "Sentinel-1": "test_s1",
                "ERS": "test_ERS",
                "Envisat": "test_envisat",
            }
            test_subdir = test_folder_map.get(satellite_name)
            if test_subdir is None:
                print(f"No test mapping found for {satellite_name}")
                return []
            scenes_dir = satellite_dir / test_subdir / "scenes"
            masks_dir = satellite_dir / test_subdir / "masks"
        else:
            raise ValueError(f"Unknown mode {mode}")
          
        if not scenes_dir.exists() or not masks_dir.exists():
            print(f"Scenes or masks directory does not exist for {satellite_dir} ({mode})")
            return []
            
        image_files = list(scenes_dir.glob('*.tif'))
        mask_files = list(masks_dir.glob('*.tif'))
        
        print(f"Found {len(image_files)} image files and {len(mask_files)} mask files")
        
        file_pairs = []
        unmatched_images = []
        unmatched_masks = []
        
        # Create a set of mask stems for quick lookup
        mask_stems = {mask_file.stem for mask_file in mask_files}
        
        for img_path in image_files:
            if img_path.stem in mask_stems:
                mask_path = masks_dir / f"{img_path.stem}.tif"
                file_pairs.append((img_path, mask_path))
            else:
                unmatched_images.append(img_path.name)
        
        # Find unmatched masks
        image_stems = {img_file.stem for img_file in image_files}
        for mask_file in mask_files:
            if mask_file.stem not in image_stems:
                unmatched_masks.append(mask_file.name)
        
        # Log unmatched files
        if unmatched_images:
            print(f"WARNING: {len(unmatched_images)} unmatched image files:")
            for img_name in unmatched_images[:5]:  # Show first 5
                print(f"  - {img_name}")
            if len(unmatched_images) > 5:
                print(f"  ... and {len(unmatched_images) - 5} more")
        
        if unmatched_masks:
            print(f"WARNING: {len(unmatched_masks)} unmatched mask files:")
            for mask_name in unmatched_masks[:5]:  # Show first 5
                print(f"  - {mask_name}")
            if len(unmatched_masks) > 5:
                print(f"  ... and {len(unmatched_masks) - 5} more")
        
        return file_pairs
    
    def _create_data_splits(self, all_file_pairs,validation_split=0.1, random_seed=42):
        """
        Create train/val splits 
        
        Args:
            all_file_pairs: list of all file pairs
            val_splits: fraction for val
            random_seed
        
        Returns:
            train_pairs,val_pairs
        
        """
        if len(all_file_pairs) == 0:
            print("No file pairs found for splitting.")
            return [], []
        
        indices = np.arange(len(all_file_pairs))
        train_idx, val_idx = train_test_split(
            indices,
            test_size= 0.1,   #applied after filtering blank images 
            random_state=random_seed,
            shuffle=True
        )
        train_pairs = [all_file_pairs[i] for i in train_idx]
        val_pairs = [all_file_pairs[i] for i in val_idx]

        #save splits for reproducibility
        splits_dir = self.output_dir / 'data_splits'
        splits_dir.mkdir(parents=True, exist_ok=True)
        with open(splits_dir/ 'train_idx.pkl', 'wb') as f:
            pickle.dump(train_idx, f)
        with open(splits_dir/ 'val_idx.pkl', 'wb') as f:
            pickle.dump(val_idx, f)
        return train_pairs, val_pairs
    
    def process_all(self, process_trainval=True, process_test=True):
        """
        Main processing function

        """
        print("=" * 60)
        print("SATELLITE IMAGERY PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Base data directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Patch size: {self.patch_size}")
        print(f"Training overlap: {self.overlap_train}")
        print(f"Validation overlap: {self.overlap_val}")
        print(f"Process train/val: {process_trainval}")
        print(f"Process test: {process_test}")
        print("=" * 60)
        
        all_threads = []

        for satellite, sat_dir in self.satellite_dirs.items():
            print(f"Processing {satellite} data...")
           
            if not sat_dir.exists():
                print(f"Directory {sat_dir} does not exist. Skipping {satellite}.")
                continue
            print(f"\nProcessing {satellite} data from {sat_dir}...")

            # trainval mode

            if process_trainval:
                file_pairs = self._get_file_pairs(sat_dir, mode="trainval", satellite_name=satellite)
                if file_pairs:
                    train_pairs, val_pairs = self._create_data_splits(file_pairs)
                    if train_pairs:
                        t = threading.Thread(
                            target=self._process_satellite_data,
                            args=(satellite, "train", train_pairs, self.overlap_train),
                        )
                        all_threads.append(t)
                        t.start()
                    if val_pairs:
                        t = threading.Thread(
                            target=self._process_satellite_data,
                            args=(satellite, "val", val_pairs, self.overlap_val),
                        )
                        all_threads.append(t)
                        t.start()
                    
            # test mode
            if process_test:
                test_pairs = self._get_file_pairs(sat_dir, mode="test", satellite_name=satellite)
                if test_pairs:
                    t = threading.Thread(
                        target=self._process_satellite_data,
                        args=(satellite, "test", test_pairs, 0),  # no overlap for test
                    )
                    all_threads.append(t)
                    t.start()
            
        for thread in all_threads:
            thread.join()

        print("All processing threads completed.")
        self._print_summary()


    def _print_summary(self):
        """
        Print a processing summary
        """
        print("\nProcessing Summary:")
        print("-" *40)

        for split in ['train', 'val']:
            img_dir = self.output_dir / split / 'images'
            mask_dir = self.output_dir / split / 'masks'

            if img_dir.exists():
                img_count = len(list(img_dir.glob('*.png')))
                mask_count = len(list(mask_dir.glob('*.png')))
                print(f"{split.capitalize()} images: {img_count}, masks: {mask_count}")
                
        scenes_dir = self.output_dir / 'test' / 'images'
        masks_dir  = self.output_dir / 'test' / 'masks'
        if scenes_dir.exists():
            scene_count = len(list(scenes_dir.glob('*.png')))
            mask_count  = len(list(masks_dir.glob('*.png')))
            print(f"Test images: {scene_count}, masks: {mask_count}")


#Main configuration 
if __name__ == "__main__":
    BASE_DATA_DIR = ROOT_GWS / "benchmark_data_CB" / "ICE-BENCH" #old name - change to Shelf-Bench
    OUTPUT_DIR = ROOT_GWS / "benchmark_data_CB" / "ICE-BENCH" / "preprocessed_data"
    PATCH_SIZE = 256
    OVERLAP_TRAIN = 0 
    OVERLAP_VAL = 0
    
    # Create preprocessor and run
    preprocessor = SatellitePreprocessor(
        base_dir=BASE_DATA_DIR,
        output_dir=OUTPUT_DIR,
        patch_size=PATCH_SIZE,
        overlap_train=OVERLAP_TRAIN,
        overlap_val=OVERLAP_VAL,
        create_trainval=True,   # important
        create_test=False
    )
    
    preprocessor.process_all(process_trainval=True, process_test=False)