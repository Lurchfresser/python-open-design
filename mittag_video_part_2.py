import os
import cv2
import random
import subprocess
import numpy as np
from tqdm import tqdm
import librosa

# --- Configuration ---
SOURCE_DIR = "büro"      # The folder with images to use for tiles
AUDIO_INPUT_PATH = "audio_input/beat_full.mpeg" # Path to the beat audio file
OUTPUT_VIDEO_PATH = "mittag_2.mp4"
TILE_SIZE = (50, 50)        # The size (width, height) of each tile
OUTPUT_DIMS = (900, 1600)     # The dimensions (width, height) of the final video

# --- Video Animation Configuration ---
VIDEO_DURATION_SECONDS = 90
FPS = 30
# Rate of tiles to replace per beat (starts low, ends high)
START_REPLACEMENT_RATE = 10.0 # At least one tile per beat
END_REPLACEMENT_RATE = 10.0 # Increased for more dynamic effect


def detect_beats(audio_path, fps):
    """
    Detects beat onsets in an audio file and returns the corresponding frame numbers.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'. No beat detection will occur.")
        return set()

    try:
        print(f"Loading audio from '{audio_path}' to detect beats...")
        y, sr = librosa.load(audio_path)
        # Detect onsets and get their times in seconds.
        # The 'units' parameter makes it return timestamps directly.
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        
        # Convert beat times (in seconds) to frame numbers
        beat_frames = {int(time * fps) for time in onset_times}
        print(f"Detected {len(beat_frames)} beats.")
        return beat_frames
    except Exception as e:
        print(f"Could not process audio file: {e}")
        return set()


def load_and_resize_images(directory, size):
    """Loads images from a directory and resizes them to the target tile size."""
    if not os.path.exists(directory):
        print(f"Error: Source directory '{directory}' not found.")
        return []
    
    source_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not source_paths:
        print(f"Error: No images found in '{directory}'.")
        return []

    resized_images = []
    print(f"Loading and resizing images from '{directory}'...")
    for path in tqdm(source_paths, desc=f"Processing {os.path.basename(directory)}"):
        img = cv2.imread(path)
        if img is not None:
            # Resize the image to fit the tile size
            resized_img = cv2.resize(img, size)
            resized_images.append(resized_img)
            
    print(f"Loaded {len(resized_images)} valid source images from '{directory}'.")
    return resized_images


def create_tiled_transition_video():
    """
    Creates a video of a mosaic where a random number of tiles are replaced
    on each beat, using images from a single source folder.
    """
    tile_width, tile_height = TILE_SIZE
    output_width, output_height = OUTPUT_DIMS

    # --- Pre-load and Resize Source Images ---
    source_images = load_and_resize_images(SOURCE_DIR, TILE_SIZE)

    if not source_images:
        print("Error: Could not load images from the source directory. Exiting.")
        return

    # --- Detect Beats from Audio ---
    beat_frames = detect_beats(AUDIO_INPUT_PATH, FPS)

    # --- Create Initial Grid & Store Tile Positions ---
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    tile_positions = []

    print("Building initial random grid...")
    for y in range(0, output_height, tile_height):
        for x in range(0, output_width, tile_width):
            end_y, end_x = y + tile_height, x + tile_width
            if end_y <= output_height and end_x <= output_width:
                # Add position to the list for later replacement
                tile_positions.append({'x': x, 'y': y})
                # Place a random tile to create the initial full grid
                tile = random.choice(source_images)
                output_image[y:end_y, x:end_x] = tile

    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, FPS, (output_width, output_height))

    # --- Generate Video Frames ---
    total_frames = int(VIDEO_DURATION_SECONDS * FPS)
    with tqdm(total=total_frames, desc="Generating Video") as pbar:
        for frame_num in range(total_frames):
            # Only perform replacements on beat frames
            if frame_num in beat_frames:
                # Calculate progress for transition and replacement rate
                progress = frame_num / total_frames
                
                # Replacement rate accelerates over time
                current_rate = START_REPLACEMENT_RATE + \
                    (END_REPLACEMENT_RATE - START_REPLACEMENT_RATE) * progress
                
                # Probabilistically determine the number of tiles to replace on this beat
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
                    x, y = pos_info['x'], pos_info['y']

                    # Always pick a new tile from the source directory
                    new_tile = random.choice(source_images)

                    # Place the new tile on the canvas
                    output_image[y:y+tile_height, x:x+tile_width] = new_tile

            # Write the (potentially updated) frame to the video
            video_writer.write(output_image)
            pbar.update(1)

    # --- Finalize ---
    video_writer.release()
    print(f"✅ Animation complete! Video saved as '{OUTPUT_VIDEO_PATH}'.")

    # Convert to a more compatible H.264 format using ffmpeg
    print("Converting video to a more compatible format with ffmpeg...")
    temp_output_path = "output_final_tiled.mp4"
    # Add the audio to the final video during the ffmpeg conversion
    final_video_with_audio_path = "mittag_with_audio.mp4"
    try:
        ffmpeg_command = [
            'ffmpeg',
            '-i', OUTPUT_VIDEO_PATH,      # Input video
            '-i', AUDIO_INPUT_PATH,       # Input audio
            '-c:v', 'libx264',            # Video codec
            '-c:a', 'aac',                # Audio codec
            '-b:a', '192k',               # Audio bitrate
            '-shortest',                  # Finish encoding when the shortest input stream ends
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-y',
            final_video_with_audio_path
        ]
        
        # If audio file doesn't exist, create video without it
        if not os.path.exists(AUDIO_INPUT_PATH):
            print("Audio file not found, creating video without audio.")
            ffmpeg_command = [
                'ffmpeg', '-i', OUTPUT_VIDEO_PATH, '-c:v', 'libx264', '-preset', 'medium',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y',
                final_video_with_audio_path
            ]

        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

        os.remove(OUTPUT_VIDEO_PATH) # remove the silent intermediate file
        os.rename(final_video_with_audio_path, OUTPUT_VIDEO_PATH) # rename final video
        print(f"✅ Video processing complete. Final output with audio: '{OUTPUT_VIDEO_PATH}'")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"--- FFMPEG failed or not found: {e}")
        print(f"The unconverted file is still available at '{OUTPUT_VIDEO_PATH}'")


if __name__ == "__main__":
    create_tiled_transition_video()