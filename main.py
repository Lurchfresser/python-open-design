import cv2
import numpy as np
import os
from tqdm import tqdm

# --- Einstellungen ---
tile_folder = "tiles"        # Ordner mit kleinen Bildern
video_path = "input.mp4"     # Dein Video
output_path = "output.mp4"   # Ergebnis
tile_size = (16, 16)         # Größe jedes "Pixelbildes"

# --- Lade alle kleinen Bilder ---
tiles = []
for filename in os.listdir(tile_folder):
    img = cv2.imread(os.path.join(tile_folder, filename))
    if img is not None:
        img = cv2.resize(img, tile_size)
        tiles.append(img)
tiles = np.array(tiles)
avg_colors = np.array([np.mean(tile, axis=(0, 1)) for tile in tiles])

# --- Video einlesen ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

print("🎬 Verarbeitung gestartet...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_small = cv2.resize(frame, (w // tile_size[0], h // tile_size[1]))
    new_frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    for y in range(frame_small.shape[0]):
        for x in range(frame_small.shape[1]):
            pixel = frame_small[y, x]
            distances = np.linalg.norm(avg_colors - pixel, axis=1)
            best_match = tiles[np.argmin(distances)]
            new_frame[y*tile_size[1]:(y+1)*tile_size[1],
                      x*tile_size[0]:(x+1)*tile_size[0]] = best_match

    out.write(new_frame)

cap.release()
out.release()
print("✅ Fertig! Video gespeichert als", output_path)