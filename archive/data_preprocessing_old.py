"""
Steps involved for preprocessing the Sentinel-1, ERS and Envisat scenes:
1. Set a naming convention: [SAT]_[YYYYMMDD]_[POLARISATION]_[EXTRA]  ⭐️
2. Keep patches at their original res: Envisat and ERS are 30 m, Sentinel-1 is 40 m). 
 - create a val dataset with 10% of the data (some hard some easy scenes) from the training set
3. Images are greyscale, masks are RBG - convert masks to greyscale as only 2 classes
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

# import libraries
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
import threading
import rasterio

# set the path to the data
"""Structure of data:

benchmark_data_CB
-- Sentinel-1
------ masks
------ scenes
-- ERS
------ masks
------ scenes
------ vectors
-- Envisat
------masks
------ scenes
------ vectors
"""

# all paths
parent_dir = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB"
S1_dir = os.path.join(parent_dir, "Sentinel-1")
ERS_dir = os.path.join(parent_dir, "ERS")
Envisat_dir = os.path.join(parent_dir, "Envisat")

class SatellitePreprocessor:
    def __init__(self, base_dir, output_dir, patch_size = 256, overlap_train=0, overlap_val =0):

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

        self._create_output_structure()

    def _create_output_structure(self):
        """
        Create the output directory structure for processed data. 
        Need to create a val dataset from train
        """
        for split in ['train', 'val']:
            for data_type in ['images', 'masks']:
                output_path = self.output_dir / split / data_type
                output_path.mkdir(parents=True, exist_ok=True)
        
        (self.output_dir /'data_splits').mkdir(exist_ok=True)

    @staticmethod
    def convert_to_db(sigma0):
        """Convert sigma0 to decibels (dB)"""
        # Replace invalid values (≤ 0) with NaN
        sigma0 = np.where(sigma0 > 0, sigma0, np.nan)
        # Convert to dB
        return 10 * np.log10(sigma0)
    
    def _normalise_image(self, image, satellite_type):
        """
        Normalise image data to uint8 range (0-255) for PNG saving.
        Handles different satellite data ranges appropriately.
        """
        # Convert SAR data to dB scale first
        if satellite_type in ['Sentinel-1', 'ERS', 'Envisat']:
            print(f"Converting {satellite_type} sigma0 to dB scale")
            original_zeros = image == 0.0  # Keep track of background pixels
            image_db = self.convert_to_db(image)
            # # # Replace NaN values with 0 for background
            # # image = np.nan_to_num(image_db, nan=0.0, posinf=0.0, neginf=0.0)
            # # Restore original background pixels
            # image[original_zeros] = 0.0
            print(f"dB conversion complete. Range: {image[image != 0].min():.2f} to {image[image != 0].max():.2f} dB")
        
        # Get valid (non-zero/non-background) pixels for normalisation
        valid_mask = image != 0.0
        
        if not np.any(valid_mask):
            print(f"Warning: Image contains only background values")
            return np.zeros_like(image, dtype=np.uint8)
        
        valid_pixels = image[valid_mask]
        
        # Calculate percentile-based normalisation to handle outliers
        p2, p98 = np.percentile(valid_pixels, [2, 98])
        
        # Avoid division by zero
        if p98 == p2:
            p2 = valid_pixels.min()
            p98 = valid_pixels.max()
            if p98 == p2:
                return np.zeros_like(image, dtype=np.uint8)
        
        print(f"Normalising using {p2:.2f} to {p98:.2f} range")
        
        # Normalize to 0-255 range
        normalised = np.clip((image - p2) / (p98 - p2) * 255, 0, 255)
        
        # Keep background as 0
        normalised[~valid_mask] = 0
        
        return normalised.astype(np.uint8)

    def _detect_background_value(self, image: np.ndarray, sample_size: int=1000) -> float:
        """
        Detect the background/nodata value in the image by sampling edge pixels.
        For this dataset, background value is always 0.0.
        """
        # For this dataset, background/nodata is always 0.0
        background_val = 0.0
        print(f"Detected background value: {background_val}")
        return background_val

        
    def _resize_image(self,image,satellite):
        """
        IGNORE THIS FUNCTION


        Resize images all to 40m resolution, sentinel-1 is already 40m, so 
        only need to downscale ERS and Envisat images.

        Inputs include image array and satellite name,
        returns resized images

        """
        if satellite in ['ERS', 'Envisat']:
            #downscale to 40m res, i.e 30/40 = 0.75
            # dont resize for now
            scale_factor = 1
            new_height = int(image.shape[0] * scale_factor)
            new_width = int(image.shape[1] * scale_factor)
            resized = cv2.resize(image, (new_width, new_height), interpolation =cv2.INTER_AREA)
            return resized
        elif satellite == 'Sentinel-1':
            # Sentinel-1 is already at 40m resolution
            return image

    def _convert_mask_to_greyscale(self, mask):
        """
        Convert RGB masks to greyscale binary mask.
        """
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            # Convert to grayscale
            grey_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            # Create binary mask - assume non-zero values are the class of interest
            binary_mask = (grey_mask > 0).astype(np.uint8) * 255
            return binary_mask
        elif len(mask.shape) == 2:
            # Already grayscale, just ensure it's binary
            binary_mask = (mask > 0).astype(np.uint8) * 255
            return binary_mask
        
        return mask
    

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
                        # mask = np.stack([mask]*3, axis=-1)  # Convert to 3-channel if needed
                    else:
                        mask = np.transpose(src_mask.read(), (1, 2, 0))  # (bands, h, w) -> (h, w, bands)
                print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")


                if image is None or mask is None:
                    print(f"Error loading image or mask: {img_path}, {mask_path}")
                    continue
            
                # Normalise image to uint8 for PNG saving
                image_uint8 = self._normalise_image(image, satellite)
                print(f"Normalised image range: {image_uint8.min()} to {image_uint8.max()}")

                ################################################
                #        Resize to 40m resolution          #
                ################################################
                # image = self._resize_image(image, satellite)
                # mask = self._resize_image(mask, satellite)

                ################################################
                #        Convert mask to greyscale          #
                ################################################
                mask = self._convert_mask_to_greyscale(mask)
                ################################################

                ########################################################
                #        Pad image and mask for patch extraction       #
                ########################################################
                image_padded, pad_h, pad_w = self._pad_image(image_uint8, self.patch_size, overlap)
                mask_padded, _, _ = self._pad_image(mask, self.patch_size, overlap)

                ########################################################
                #        Extract patches from padded image and mask    #
                ########################################################
                image_patches, coords = self._extract(image_padded, self.patch_size, overlap)
                mask_patches, _ = self._extract(mask_padded, self.patch_size, overlap)

                ########################################################
                #        Save patches to output directory          #
                ########################################################
                base_filename = img_path.stem

                for i, (img_patch, mask_patch) in enumerate(zip(image_patches, mask_patches)):
                    y,x = coords[i]

                     # Skip patches that are all background
                    if np.all(img_patch == 0.0):
                        continue  # Skip saving this patch

                    #create a patch name
                    patch_name = f"{base_filename}__{pad_h}_{pad_w}_{i}_{y}_{x}.png"

                    # Save image patch
                    img_output_path = self.output_dir / split_type / 'images' / patch_name
                    cv2.imwrite(str(img_output_path), img_patch)
                    
                    # Save mask patch
                    mask_output_path = self.output_dir / split_type / 'masks' / patch_name
                    cv2.imwrite(str(mask_output_path), mask_patch)
                
                print(f"✓ Processed {base_filename}: {len(image_patches)} patches")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    def _get_file_pairs(self, satellite_dir):
        """
        Get paired image and mask files

        returns a list of (image_path, mask_path) tuples

        """
        scenes_dir = satellite_dir / 'scenes'
        masks_dir = satellite_dir / 'masks'
        if not scenes_dir.exists() or not masks_dir.exists():
            print(f"Scenes or masks directory does not exist for {satellite_dir}")
            return []
        image_files = list(scenes_dir.glob('*.tif'))
        file_pairs = []
        for img_path in image_files:
            mask_candidates = [
             
                masks_dir / f"{img_path.stem}.tif"
            ]
            mask_path = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path:
                file_pairs.append((img_path, mask_path))
            else:
                print(f"No matching mask found for {img_path}")
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
            test_size= validation_split,
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
    
    def process_all(self):
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
        print("=" * 60)
        
        all_threads = []

        for satellite, sat_dir in self.satellite_dirs.items():
            print(f"Processing {satellite} data...")
           
            if not sat_dir.exists():
                print(f"Directory {sat_dir} does not exist. Skipping {satellite}.")
                continue
            print(f"\nProcessing {satellite} data from {sat_dir}...")

            # Get file pairs for images and masks
            file_pairs = self._get_file_pairs(sat_dir)

            if not file_pairs:
                print(f"No valid file pairs found for {satellite}. Skipping.")
                continue

            print(f"Found {len(file_pairs)} file pairs for {satellite}.")

            # Create train/val splits
            train_pairs, val_pairs = self._create_data_splits(file_pairs)
            print(f"Split: {len(train_pairs)} training, {len(val_pairs)} validation")
            
            if train_pairs:
                thread = threading.Thread(
                    target=self._process_satellite_data,
                    args=(satellite, 'train', train_pairs, self.overlap_train)
                )
                all_threads.append(thread)
                thread.start()

            if val_pairs:
                thread = threading.Thread(
                    target=self._process_satellite_data,
                    args=(satellite, 'val', val_pairs, self.overlap_val)
                )
                all_threads.append(thread)
                thread.start()
            
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

#Main configuration 
if __name__ == "__main__":
    BASE_DATA_DIR = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB"
    OUTPUT_DIR = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/preprocessed_data"
    PATCH_SIZE = 256
    OVERLAP_TRAIN = 0 
    OVERLAP_VAL = 0
    
    # Create preprocessor and run
    preprocessor = SatellitePreprocessor(
        base_dir=BASE_DATA_DIR,
        output_dir=OUTPUT_DIR,
        patch_size=PATCH_SIZE,
        overlap_train=OVERLAP_TRAIN,
        overlap_val=OVERLAP_VAL
    )
    
    preprocessor.process_all()
