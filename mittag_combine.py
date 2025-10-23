import subprocess
import os

# --- Configuration ---
VIDEO_INPUT_1 = "mittag.mp4"
VIDEO_INPUT_2 = "mittag_2.mp4"
AUDIO_INPUT = "audio_input/full.mpeg"
OUTPUT_VIDEO = "final_combined_video.mp4"

# --- Transition Configuration ---
WHITE_GAP_SECONDS = 4  # Duration of the white screen between videos
VIDEO_1_SCRIPT = "mittag_video.py"
VIDEO_2_SCRIPT = "mittag_video_part_2.py"
OUTPUT_DIMS = (900, 1600) # Must match the source videos
FPS = 30                  # Must match the source videos


def run_generator_script(script_name):
    """Executes a given python script and waits for it to complete."""
    try:
        print(f"--- Running script: {script_name} ---")
        subprocess.run(['python3', script_name], check=True)
        print(f"--- Finished script: {script_name} ---")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running {script_name}: {e}")
        return False


def combine_videos_with_gap():
    """
    Generates source videos, then combines them with a white gap and new audio.
    """
    # --- Step 1: Generate the source videos ---
    if not run_generator_script(VIDEO_1_SCRIPT): return
    if not run_generator_script(VIDEO_2_SCRIPT): return

    # --- Step 2: Validate Inputs ---
    if not all(os.path.exists(p) for p in [VIDEO_INPUT_1, VIDEO_INPUT_2, AUDIO_INPUT]):
        print("Error: One or more input files are missing after generation. Please check the paths.")
        return

    print("\nCombining videos with ffmpeg...")
    print(f"Video 1: {VIDEO_INPUT_1}")
    print(f"Gap:     {WHITE_GAP_SECONDS}s of white screen")
    print(f"Video 2: {VIDEO_INPUT_2}")
    print(f"Audio:   {AUDIO_INPUT}")

    # --- Step 3: Construct and Run FFmpeg Command ---
    try:
        ffmpeg_command = [
            'ffmpeg',
            # Inputs
            '-i', VIDEO_INPUT_1,
            '-i', VIDEO_INPUT_2,
            '-i', AUDIO_INPUT,
            
            # Filter Complex to create white gap and concatenate
            '-filter_complex',
            (
                # Create a 4-second white canvas
                f"color=white:s={OUTPUT_DIMS[0]}x{OUTPUT_DIMS[1]}:d={WHITE_GAP_SECONDS}:r={FPS}[white];"
                # Concatenate Video 1, the white canvas, and Video 2
                "[0:v:0][white][1:v:0]concat=n=3:v=1:a=0[final_v]"
            ),
            
            # Mapping
            '-map', '[final_v]',  # Use the concatenated video stream
            '-map', '2:a',        # Use the audio from the third input
            
            # Encoding options
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',          # Finish when the shortest stream (the combined video) ends
            '-y',                 # Overwrite output file if it exists
            OUTPUT_VIDEO
        ]

        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        
        print(f"✅ Success! Combined video saved as '{OUTPUT_VIDEO}'.")

    except subprocess.CalledProcessError as e:
        print("--- FFMPEG Error ---")
        print(f"ffmpeg failed to execute. Return code: {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print("Error: 'ffmpeg' command not found. Please make sure it is installed and accessible in your system's PATH.")

if __name__ == "__main__":
    combine_videos_with_gap()