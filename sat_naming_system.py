""" 
Set a naming convention for the Sentinel-1, ERS and Envisat scenes
Current naming convention:
Sentinel-1 masks: benchmark_data_CB/Sentinel-1/masks/S1A_EW_GRDM_1SDH_20170609T151156_20170609T151300_016958_01C3A7_84C0_Orb_NR_Cal_TC.tif
Sentinel-1 scenes: benchmark_data_CB/Sentinel-1/scenes/S1A_EW_GRDM_1SDH_20170609T151156_20170609T151300_016958_01C3A7_84C0_Orb_NR_Cal_TC.tif

ERS masks: benchmark_data_CB/ERS/masks/CG_prepro_SAR_IMP_1PNESA19961029_142649_00000018A016_00110_07980_0000_cropped.tif
    benchmark_data_CB/ERS/masks/PIG_prepro_SAR_IMP_1PNESA19950122_141125_00000018F142_01617_18418_0000_cropped.tif
    benchmark_data_CB/ERS/masks/CIS_prepro_SAR_IMP_1PNESA19960214_143603_00000016A008_00425_04287_0000_cropped.tif
    benchmark_data_CB/ERS/masks/PIG02_prepro_SAR_IMP_1PNESA20020105_135932_00000017A070_00167_35091_0000_cropped.tif
    benchmark_data_CB/ERS/masks/prepro_SAR_IMP_1PNESA19911203_091555_00000017A045_00007_01996_0000_cropped.tif
    benchmark_data_CB/ERS/masks/TG_prepro_SAR_IMP_1PNESA19951107_141549_00000018G151_00024_22557_0000_cropped.tif
ERS scenes: same filenames as masks but stored in scenes folder
ERS vectors: only has the associated date with the scene/mask as .shp, .prj, .dbf, .shx .qmd, .cpg

Envisat masks: benchmark_data_CB/Envisat/masks/prepro_ASA_IMP_1PNESA20060205_133956_000000182044_00482_20576_0000_cropped.tif
Envisat scenes: same
Envisat vectors: only has the associated date with the scene/mask as .shp, .prj, .dbf, .shx .qmd, .cpg

Names will follow:
[SATELLITE-ID]_[YYYYMMDD]_[polarization]_[extra]

"""

# import libraries
import os
import re
import datetime
import csv

def process_s1(filepath):
    """
    Process Sentinel-1 filenames to handle different file endings:
    - Orb_NR_Cal_TC.tif
    - Orb_NR_Cal_TC_crop.tif
    - _crop.tif
    """
    filename = os.path.basename(filepath)
    s1_type_match = re.search(r'(S1[AB])_EW_GRDM_1S[DS]H', filename)
    s1_type = s1_type_match.group(1) if s1_type_match else "S1A"
    
    date_match = re.search(r'_(\d{8})T\d{6}_', filename)
    date_str = date_match.group(1) if date_match else "unknown_date"

    if date_match:
        date_str = date_match.group(1) # YYYYMMDD
    else:
        date_str = "unknown_date"

    #polarisation. TODO: check with celia for consistency here
    if '1SDH' in filename:
        polarisation = 'HHHV'
    elif '1SSH' in filename:
        polarisation = 'HH'
    else:
        polarisation = 'unknown_polarisation'

    code_match = re.search(r'_(\w{4})_Orb_NR_Cal_TC', filename)
    # If that fails, try to get the code before _crop.tif
    if code_match:
        code = code_match.group(1)
    else:
        code_match = re.search(r'_(\w{4})_crop\.tif$', filename)
        if code_match:
            code = code_match.group(1)
        else:
            # If that also fails, try to get the code just before any underscore
            all_codes = re.findall(r'_([A-Z0-9]{4})_', filename)
            if all_codes:
                # Take the last one which is likely to be the product code
                code = all_codes[-1]
            else:
                code = "XXXX"
        
    crop_number_match = re.search(r'_crop-(\d+)\.tif$', filename)
    crop_suffix = ""
    if crop_number_match:
        crop_suffix = f"_{crop_number_match.group(1)}"

    new_filename = f"{s1_type}_{date_str}_{polarisation}_EW_GRDM_1SDH_{code}{crop_suffix}"
    extension = os.path.splitext(filename)[1]
    return new_filename + extension

def process_ERS(filepath):
    filename = os.path.join(filepath)
    location_prefix_match = re.match(r'^([A-Z]+\d*)_', filename)
    if location_prefix_match:
        location = location_prefix_match.group(1)
    else:
        location_in_path = None
        for loc in ['CG', 'PIG', 'TG', 'CIS', 'PIG02', 'VIS', 'LDIS','FIS','RIS','CKIS','WIS','RG','GVIS']:
            if loc in filepath:
                location_in_path = loc
                break
        location = location_in_path if location_in_path else "UNK"

    # Extract the location and ERS code from the filename
    ers_date_match = re.search(r'1PNESA(\d{8})_', filename)
    if ers_date_match:
        date_str = ers_date_match.group(1)
        ers_code = f"1PNESA{date_str}"
    else:
        date_str = "unknown_date"
        ers_code = "unknown_code"

    polarisation = "VV"
    new_filename = f"ERS_{date_str}_{polarisation}_{location}_{ers_code}"
    extension = os.path.splitext(filename)[1]
    return new_filename + extension

def process_Envisat(filepath):
    filename = os.path.basename(filepath)
    date_match = re.search(r'prepro_ASA_IMP_(1PNESA\d{8})_', filename)
    if date_match:
        env_code = date_match.group(1)

        date_str_match = re.search(r'1PNESA(\d{8})_', filename)
        if date_str_match:
            date_str = date_str_match.group(1) # YYYYMMDD

        else:
            date_str = "unknown_date"

    else: 
        env_code = "UNKNOWN"
        date_str = "UNKNOWN"
    
    polarisation = "VV"
    new_filename = f"ENV_{date_str}_{polarisation}_{env_code}"
    extension = os.path.splitext(filename)[1]
    return new_filename + extension

def rename_files(directory, dry_run=True, csv_log_path="renamed_testfiles_log.csv"):
    """
    Rename all satellite data files in the given directory structure according to the new naming convention.
    Set dry_run=False to actually perform the renaming.
    """
    rename_log = []
    for root, dirs, files in os.walk(directory):
        
        # Only process 'test_' directories, comment out for train/val
        
        if "test_" not in os.path.basename(root) and not any("test_" in d for d in root.split(os.sep)):
            continue  # Skip directories that are not test directories
        
        for file in files:
            if file.lower().startswith("readme") or file.lower() == "readme.txt":
                continue
            filepath = os.path.join(root, file)
            new_filename = None
            if 'Sentinel-1' in root or 'test_s1' in root:
                new_filename = process_s1(filepath)
            elif 'ERS' in root or 'test_ERS' in root:
                if any(file.endswith(ext) for ext in ['.shp', '.prj', '.dbf', '.shx', '.qmd', '.cpg']):
                    continue
                new_filename = process_ERS(filepath)
            elif 'Envisat' in root or 'test_envisat' in root:
                if any(file.endswith(ext) for ext in ['.shp', '.prj', '.dbf', '.shx', '.qmd', '.cpg']):
                    continue
                new_filename = process_Envisat(filepath)
            
            if new_filename:
                new_filepath = os.path.join(os.path.dirname(filepath), new_filename)
                if dry_run:
                    print(f"Would rename: {filepath} to {new_filepath}")
                else:
                    os.rename(filepath, new_filepath)
                    print(f"Renamed: {filepath} to {new_filepath}")
                rename_log.append((filepath, new_filepath))
    if rename_log:
        with open(csv_log_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['original_filepath', 'new_filepath'])
            writer.writerows(rename_log)
        print(f"Rename log saved to {csv_log_path}")

if __name__ == "__main__":
    # Example usage
    # Change this to your actual directory path
    base_directory = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB/ICE-BENCH"
    
    # # Do a dry run first to see what would be renamed
    print("Dry run - no files will be renamed:")
    rename_files(base_directory, dry_run=True,csv_log_path="dry_run_renamed_files_log.csv")
    
    # Uncomment these lines to actually perform the renaming
    # print("\nPerforming actual renaming:")
    # rename_files(base_directory, dry_run=True)