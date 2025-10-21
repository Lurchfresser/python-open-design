import os
import cv2
import random
import subprocess
import numpy as np
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR_A = "wiese_night"  # Start with tiles from this folder
SOURCE_DIR_B = "wiese_day"  # End with tiles from this folder
OUTPUT_VIDEO_PATH = "animated_mosaic.mp4"
TILE_SIZE = (50, 50)      # The size (width, height) of each tile
OUTPUT_DIMS = (900, 1600)  # The dimensions (width, height) of the final video
HORIZON_RATIO = 0.6       # Where to place the horizon (0.0=top, 1.0=bottom)

# --- Video Animation Configuration ---
VIDEO_DURATION_SECONDS = 180  # 3 minutes
FPS = 30
# Rate of tiles to replace per frame (starts low, ends high)
START_REPLACEMENT_RATE = 0.001
END_REPLACEMENT_RATE = 2


def load_source_images(directory, tile_height, tile_width):
    """Loads and validates images from a given directory."""
    if not os.path.exists(directory):
        print(f"Error: Source directory '{directory}' not found.")
        return []
    
    source_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not source_paths:
        print(f"Error: No images found in '{directory}'.")
        return []

    source_images = [cv2.imread(p) for p in source_paths]
    valid_images = [img for img in source_images if img is not None and img.shape[0] >= tile_height and img.shape[1] >= tile_width]
    print(f"Loaded {len(valid_images)} valid source images from '{directory}'.")
    return valid_images


def get_random_tile(source_images, is_sky_tile):
    """Gets a random tile crop from the appropriate region of a random source image."""
    tile_width, tile_height = TILE_SIZE
    if not source_images:
        # Return a black tile if the source list is empty
        return np.zeros((tile_height, tile_width, 3), dtype=np.uint8)

    source_img = random.choice(source_images)
    source_h, source_w, _ = source_img.shape
    source_horizon_y = int(source_h * HORIZON_RATIO)

    if is_sky_tile:
        sample_min_y, sample_max_y = 0, max(0, source_horizon_y - tile_height)
    else:  # Field
        sample_min_y, sample_max_y = min(
            source_horizon_y, source_h - tile_height), source_h - tile_height

    if sample_min_y >= sample_max_y:
        # Fallback in case a region is too small, try again
        return get_random_tile(source_images, is_sky_tile)

    random_x = random.randint(0, source_w - tile_width)
    random_y = random.randint(sample_min_y, sample_max_y)
    return source_img[random_y:random_y + tile_height, random_x:random_x + tile_width]


def create_animated_mosaic():
    """
    Creates a video of a mosaic where tiles are replaced at an accelerating rate,
    transitioning from one source folder to another.
    """
    tile_width, tile_height = TILE_SIZE
    output_width, output_height = OUTPUT_DIMS

    # --- Validate Inputs & Pre-load Sources ---
    source_images_A = load_source_images(SOURCE_DIR_A, tile_height, tile_width)
    source_images_B = load_source_images(SOURCE_DIR_B, tile_height, tile_width)

    if not source_images_A or not source_images_B:
        print("Error: Could not load images from one or both source directories. Exiting.")
        return

    # --- Create Initial Mosaic & Store Tile Positions ---
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    output_horizon_y = int(output_height * HORIZON_RATIO)
    tile_positions = []

    print("Building initial mosaic frame from source A...")
    for y in range(0, output_height, tile_height):
        for x in range(0, output_width, tile_width):
            is_sky = (y + tile_height) < output_horizon_y
            # Initial frame uses only source A
            tile = get_random_tile(source_images_A, is_sky)

            end_y, end_x = y + tile_height, x + tile_width
            if end_y <= output_height and end_x <= output_width:
                output_image[y:end_y, x:end_x] = tile
                tile_positions.append({'x': x, 'y': y, 'is_sky': is_sky})

    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, FPS, (output_width, output_height))

    # --- Generate Video Frames ---
    total_frames = VIDEO_DURATION_SECONDS * FPS
    with tqdm(total=total_frames, desc="Generating Video") as pbar:
        for frame_num in range(total_frames):
            # Calculate progress for lerping and replacement rate
            progress = frame_num / total_frames
            
            # Replacement rate accelerates over time
            current_rate = START_REPLACEMENT_RATE + \
                (END_REPLACEMENT_RATE - START_REPLACEMENT_RATE) * progress
            
            # Probabilistically determine the number of tiles to replace
            guaranteed_replacements = int(current_rate)
            fractional_chance = current_rate - guaranteed_replacements
            additional_replacement = 1 if random.random() < fractional_chance else 0
            tiles_to_replace = guaranteed_replacements + additional_replacement

            # Randomly select and replace tiles
            for _ in range(tiles_to_replace):
                if not tile_positions:
                    continue

                # Pick a random tile position to update
                pos_info = random.choice(tile_positions)
                x, y, is_sky = pos_info['x'], pos_info['y'], pos_info['is_sky']

                # --- LERP between source folders ---
                # The chance of picking from folder B increases with progress
                if random.random() < progress:
                    selected_sources = source_images_B
                else:
                    selected_sources = source_images_A

                # Get a new random tile and place it
                new_tile = get_random_tile(selected_sources, is_sky)
                output_image[y:y+tile_height, x:x+tile_width] = new_tile

            # Write the updated frame to the video
            video_writer.write(output_image)
            pbar.update(1)

    # --- Finalize ---
    video_writer.release()
    print(f"✅ Animation complete! Video saved as '{OUTPUT_VIDEO_PATH}'.")

    # Konvertiere zu H.264 mit hoher Kompatibilität
    print("Converting video to a more compatible format with ffmpeg...")
    temp_output_path = "output_final.mp4"
    subprocess.run([
        'ffmpeg', '-i', OUTPUT_VIDEO_PATH,
        '-c:v', 'libx264',           # H.264 Codec
        '-preset', 'medium',         # Geschwindigkeit/Qualität Balance
        '-crf', '23',                # Qualität (18-28, niedriger = besser)
        '-pix_fmt', 'yuv420p',       # Kompatibles Pixelformat
        '-movflags', '+faststart',   # Optimierung für Web-Streaming
        '-y',                        # Überschreibe Ausgabedatei
        temp_output_path
    ], check=True, capture_output=True, text=True) # Use capture_output to hide ffmpeg logs

    # Replace the original with the converted one
    os.replace(temp_output_path, OUTPUT_VIDEO_PATH)
    
    print(f"✅ Video processing complete. Final output: '{OUTPUT_VIDEO_PATH}'")


if __name__ == "__main__":
    create_animated_mosaic()
