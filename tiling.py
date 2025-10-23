import os
import cv2

# --- Configuration ---
SOURCE_DIR = "büro"               # The directory with images to be segmented
OUTPUT_DIR = "büro_tiles"
TILE_SIZE = (200, 200)                      # The size (width, height) of each tile to crop

def segment_images_in_directory():
    """
    Segments all images in a source directory into multiple tiles of a fixed size
    without resizing, and saves them in the output directory.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: '{OUTPUT_DIR}'")

    # Check if the source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    print(f"Segmenting all images in '{SOURCE_DIR}' into {TILE_SIZE[0]}x{TILE_SIZE[1]} tiles...")
    
    tile_count = 0
    # Iterate over all files in the source directory
    for source_filename in sorted(os.listdir(SOURCE_DIR)):
        source_path = os.path.join(SOURCE_DIR, source_filename)

        # Check if it's a file and not a directory
        if not os.path.isfile(source_path):
            continue

        img = cv2.imread(source_path)
        if img is None:
            print(f"Warning: Could not read file '{source_filename}'. Skipping.")
            continue

        print(f"  - Processing '{source_filename}'...")
        img_height, img_width, _ = img.shape
        tile_width, tile_height = TILE_SIZE
        
        # Iterate over the image in steps of the tile size
        for y in range(0, img_height, tile_height):
            for x in range(0, img_width, tile_width):
                # Define the bounding box for the crop
                y1, y2 = y, y + tile_height
                x1, x2 = x, x + tile_width

                # Ensure the crop dimensions are not out of bounds
                if y2 > img_height or x2 > img_width:
                    continue

                # Crop the tile from the image
                tile = img[y1:y2, x1:x2]
                
                # Construct a unique filename for the tile
                filename = f"tile_{tile_count}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                
                # Save the new tile
                cv2.imwrite(output_path, tile)
                tile_count += 1

    print(f"✅ Tiling complete. {tile_count} total tiles saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    segment_images_in_directory()