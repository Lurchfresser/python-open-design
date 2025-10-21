import os
import cv2
import random
import numpy as np
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = "tile_sources"
OUTPUT_IMAGE_PATH = "consistent_random_mosaic.png"
TILE_SIZE = (100, 100)      # The size (width, height) of each tile
OUTPUT_DIMS = (900, 1600) # The dimensions (width, height) of the final image
HORIZON_RATIO = 0.6       # Where to place the horizon (0.0=top, 1.0=bottom)

def create_consistent_random_mosaic():
    """
    Creates a mosaic with a consistent horizon by randomly sampling tiles
    from corresponding regions (sky/field) of the source images.
    """
    tile_width, tile_height = TILE_SIZE
    output_width, output_height = OUTPUT_DIMS

    # --- Validate Inputs & Pre-load Sources ---
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    source_paths = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
    if not source_paths:
        print(f"Error: No images found in '{SOURCE_DIR}'.")
        return
    
    source_images = [cv2.imread(p) for p in source_paths]
    source_images = [img for img in source_images if img is not None and img.shape[0] >= tile_height and img.shape[1] >= tile_width]
    print(f"Loaded {len(source_images)} valid source images.")

    # --- Create Blank Output Image ---
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    output_horizon_y = int(output_height * HORIZON_RATIO)
    print(f"Creating a {output_width}x{output_height} output image with horizon at y={output_horizon_y}...")

    # --- Iterate and Place Tiles with Progress Bar ---
    total_tiles = (output_height // tile_height) * (output_width // tile_width)
    with tqdm(total=total_tiles, desc="Building Mosaic") as pbar:
        for y in range(0, output_height, tile_height):
            for x in range(0, output_width, tile_width):
                # 1. Get a random source image
                source_img = random.choice(source_images)
                source_h, source_w, _ = source_img.shape
                source_horizon_y = int(source_h * HORIZON_RATIO)

                # 2. Determine the correct sampling region (sky or field)
                if y + tile_height < output_horizon_y: # Sky
                    sample_min_y, sample_max_y = 0, max(0, source_horizon_y - tile_height)
                else: # Field (includes horizon line)
                    sample_min_y, sample_max_y = min(source_horizon_y, source_h - tile_height), source_h - tile_height
                
                if sample_min_y >= sample_max_y: continue # Skip if region is invalid

                # 3. Get a random crop from that region
                random_x = random.randint(0, source_w - tile_width)
                random_y = random.randint(sample_min_y, sample_max_y)
                tile = source_img[random_y:random_y + tile_height, random_x:random_x + tile_width]
                
                # 4. Place the tile in the output image
                end_y, end_x = y + tile_height, x + tile_width
                if end_y <= output_height and end_x <= output_width:
                    output_image[y:end_y, x:end_x] = tile
                
                pbar.update(1)

    # --- Save the Final Image ---
    cv2.imwrite(OUTPUT_IMAGE_PATH, output_image)
    print(f"✅ Mosaic complete! Image saved as '{OUTPUT_IMAGE_PATH}'.")


if __name__ == "__main__":
    create_consistent_random_mosaic()