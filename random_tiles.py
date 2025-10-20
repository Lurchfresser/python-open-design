import os
import cv2
import random
import numpy as np

# --- Configuration ---
SOURCE_DIR = "tile_sources"
OUTPUT_IMAGE_PATH = "consistent_mosaic.png"
TILE_SIZE = (100, 100)      # The size (width, height) of each tile
OUTPUT_DIMS = (900, 1600) # The dimensions (width, height) of the final image
HORIZON_RATIO = 0.6       # Where to place the horizon (0.0=top, 1.0=bottom)

def create_consistent_mosaic():
    """
    Creates a single output image composed of tiles cropped from corresponding
    regions (sky/field) of the source images to maintain consistency.
    """
    tile_width, tile_height = TILE_SIZE
    output_width, output_height = OUTPUT_DIMS

    # --- Validate Inputs ---
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    source_images = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
    if not source_images:
        print(f"Error: No images found in '{SOURCE_DIR}'.")
        return

    print(f"Found {len(source_images)} source images.")

    # --- Create Blank Output Image ---
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    output_horizon_y = int(output_height * HORIZON_RATIO)
    print(f"Creating a {output_width}x{output_height} output image with horizon at y={output_horizon_y}...")

    # --- Iterate and Place Tiles ---
    for y in range(0, output_height, tile_height):
        for x in range(0, output_width, tile_width):
            # 1. Choose a random source image
            random_image_name = random.choice(source_images)
            source_path = os.path.join(SOURCE_DIR, random_image_name)
            source_img = cv2.imread(source_path)

            if source_img is None: continue

            source_h, source_w, _ = source_img.shape
            if source_h < tile_height or source_w < tile_width: continue

            # 2. Determine the sampling region based on the tile's position relative to the horizon
            source_horizon_y = int(source_h * HORIZON_RATIO)
            
            # Determine if the current tile position is in the sky, on the horizon, or in the field
            if y + tile_height < output_horizon_y: # Purely sky tile
                sample_min_y, sample_max_y = 0, source_horizon_y - tile_height
            elif y > output_horizon_y: # Purely field tile
                sample_min_y, sample_max_y = source_horizon_y, source_h - tile_height
            else: # A tile that crosses the horizon
                # We want the crop to also cross the horizon in the source image
                # Calculate where the horizon line falls *within* the current tile
                horizon_in_tile_ratio = (output_horizon_y - y) / tile_height
                # Position the crop so the horizon falls at the same relative spot
                sample_min_y = source_horizon_y - int(tile_height * horizon_in_tile_ratio)
                sample_max_y = sample_min_y

            # Ensure sampling region is valid
            if sample_min_y >= sample_max_y or sample_min_y < 0 or sample_max_y < 0:
                # Fallback for edge cases or images that don't fit the model
                sample_min_y, sample_max_y = 0, source_h - tile_height

            # 3. Find a random location in the determined region to crop from
            random_x = random.randint(0, source_w - tile_width)
            random_y = random.randint(sample_min_y, sample_max_y)

            # 4. Crop the tile
            cropped_tile = source_img[random_y:random_y + tile_height, random_x:random_x + tile_width]

            # 5. Place the tile in the output image
            end_y, end_x = y + tile_height, x + tile_width
            if end_y <= output_height and end_x <= output_width:
                output_image[y:end_y, x:end_x] = cropped_tile

    # --- Save the Final Image ---
    cv2.imwrite(OUTPUT_IMAGE_PATH, output_image)
    print(f"✅ Mosaic complete! Image saved as '{OUTPUT_IMAGE_PATH}'.")


if __name__ == "__main__":
    create_consistent_mosaic()