import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import subprocess

# --- Einstellungen ---
tile_folder = "tiles"        # Ordner mit kleinen Bildern
video_path = "input2.mp4"     # Dein Video
output_path = "output.mp4"   # Ergebnis (Dateiendung zu .mp4 geändert)
tile_size = (8, 8)         # Größe jedes "Pixelbildes"
BLACK_THRESHOLD = 30         # Brightness value below which a pixel is considered 'black'
BLACK_PERCENT_LIMIT = 0.1    # If a tile is more than this % black, it's skipped

# --- Lade alle kleinen Bilder ---
tiles = []
for filename in os.listdir(tile_folder):
    img = cv2.imread(os.path.join(tile_folder, filename))
    if img is not None:
        # Ensure tile is resized to the correct dimension for placing
        img = cv2.resize(img, tile_size)
        tiles.append(img)
tiles = np.array(tiles)

# --- Video einlesen ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))

# --- Calculate new dimensions that are multiples of tile_size ---
new_w = (w // tile_size[0]) * tile_size[0]
new_h = (h // tile_size[1]) * tile_size[1]

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Use the H.264 codec ('avc1') for MP4 output
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (new_w, new_h))

print(f"🎬 Verarbeitung gestartet... (Frames werden auf {new_w}x{new_h} skaliert)")

for _ in tqdm(range(total_frames), desc="Verarbeite Frames"):
    ret, frame = cap.read()
    if not ret:
        break
    
    # --- Resize the frame to the clean dimensions ---
    frame = cv2.resize(frame, (new_w, new_h))
    
    new_frame = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    for y in range(0, new_h, tile_size[1]):
        for x in range(0, new_w, tile_size[0]):
            # Define the bounding box for the current tile area
            y1, y2 = y, y + tile_size[1]
            x1, x2 = x, x + tile_size[0]

            # The boundary check is no longer strictly necessary but is good practice
            if y2 > new_h or x2 > new_w:
                continue

            # Extract the current tile area from the original frame
            current_area = frame[y1:y2, x1:x2]
            
            # Calculate the percentage of black pixels in the area
            gray_area = cv2.cvtColor(current_area, cv2.COLOR_BGR2GRAY)
            black_pixels = np.sum(gray_area < BLACK_THRESHOLD)
            total_pixels = tile_size[0] * tile_size[1]
            black_percentage = black_pixels / total_pixels

            # If the area is not mostly black, place a random tile
            if black_percentage > BLACK_PERCENT_LIMIT:
                random_tile = random.choice(tiles)
                new_frame[y1:y2, x1:x2] = random_tile
            # If it is mostly black, new_frame remains black (as it was initialized with zeros)

    out.write(new_frame)

cap.release()
out.release()
print("✅ Konvertierung mit ffmpeg...")

# Konvertiere zu H.264 mit hoher Kompatibilität
subprocess.run([
    'ffmpeg', '-i', output_path, 
    '-c:v', 'libx264',           # H.264 Codec
    '-preset', 'medium',         # Geschwindigkeit/Qualität Balance
    '-crf', '23',                # Qualität (18-28, niedriger = besser)
    '-pix_fmt', 'yuv420p',       # Kompatibles Pixelformat
    '-movflags', '+faststart',   # Optimierung für Web-Streaming
    '-y',                        # Überschreibe Ausgabedatei
    'output_final.mp4'
], check=True)

subprocess.run([
    "mv", "output_final.mp4", output_path
])

print("✅ Fertig! Kompatibles Video: ", output_path)