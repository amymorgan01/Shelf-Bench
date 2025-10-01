import rasterio
from pathlib import Path

def check_size_mismatches(base_path, satellites=['Sentinel-1', 'ERS', 'Envisat']):

    mismatches = []
    
    for satellite in satellites:
        satellite_dir = Path(base_path) / satellite
        scenes_dir = satellite_dir / 'scenes'
        masks_dir = satellite_dir / 'masks'
        
        scene_files = list(scenes_dir.glob('*.tif'))
        mask_files = list(masks_dir.glob('*.tif'))
        
        mask_dict = {mask_file.stem: mask_file for mask_file in mask_files}
        
        for scene_file in scene_files:
            if scene_file.stem in mask_dict:
                mask_file = mask_dict[scene_file.stem]
                
                try:
                  
                    with rasterio.open(scene_file) as scene_src:
                        scene_shape = (scene_src.width, scene_src.height)
                    
                    with rasterio.open(mask_file) as mask_src:
                        mask_shape = (mask_src.width, mask_src.height)
                    
                    if scene_shape != mask_shape:
                        mismatches.append({
                            'satellite': satellite,
                            'scene_file': scene_file.name,
                            'mask_file': mask_file.name,
                            'scene_size': scene_shape,
                            'mask_size': mask_shape
                        })
                        
                except Exception as e:
                    mismatches.append({
                        'satellite': satellite,
                        'scene_file': scene_file.name,
                        'mask_file': mask_file.name,
                        'error': str(e)
                    })
    
    return mismatches

base_path = "/gws/nopw/j04/iecdt/amorgan/benchmark_data_CB"
mismatches = check_size_mismatches(base_path)

if mismatches:
    print(f"Found {len(mismatches)} mismatched files:")
    for mismatch in mismatches:
        if 'error' in mismatch:
            print(f"  {mismatch['satellite']}: {mismatch['scene_file']} - ERROR: {mismatch['error']}")
        else:
            print(f"  {mismatch['satellite']}: {mismatch['scene_file']} ({mismatch['scene_size']}) vs {mismatch['mask_file']} ({mismatch['mask_size']})")
else:
    print("All scenes and masks have matching dimensions!")