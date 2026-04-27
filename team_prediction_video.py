# team_prediction_video_v2.py

import cv2
import numpy as np
import os
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv
from tqdm import tqdm
import inspect

# ============================================================
# PREDEFINED COLORS
# ============================================================
TEAM_COLORS = {
    1: {
        'player':     np.array([235, 245, 250]),
        'goalkeeper': np.array([30,  30,  30]),
    },
    2: {
        'player':     np.array([170, 250, 140]),
        'goalkeeper': np.array([206, 130,  99]),
    }
}

full_classes = ['player-ball-goalkeeper-referee-QfA4', 'ball', 'goalkeeper', 'player', 'referee']
class_names  = full_classes[1:]

# ============================================================
# MEMORI TEAM ASSIGNMENT
# Format: { track_id: team_id }
# Sekali di-assign, tidak akan berubah
# ============================================================
team_memory = {}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_center_crop(bbox, frame, crop_ratio=0.3):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) // 2
    cy = y1 + int(h * 0.35)
    half_w = int(w * crop_ratio / 2)
    half_h = int(h * crop_ratio / 2)
    crop_x1 = max(0, cx - half_w)
    crop_y1 = max(0, cy - half_h)
    crop_x2 = min(frame.shape[1], cx + half_w)
    crop_y2 = min(frame.shape[0], cy + half_h)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2)

def extract_dominant_colors(crop, n_colors=3):
    if crop is None or crop.size == 0:
        return None
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixels  = img_rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) < n_colors:
        return None
    kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42)
    kmeans.fit(pixels)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)
    return kmeans.cluster_centers_[sorted_idx]

def is_grass_color(rgb_color, threshold=40):
    r, g, b = rgb_color
    return (g > r + threshold) and (g > b + threshold)

def filter_grass_colors(colors):
    if colors is None:
        return None
    filtered = [c for c in colors if not is_grass_color(c)]
    return np.array(filtered) if filtered else None

def color_distance_lab(rgb1, rgb2):
    lab1 = rgb2lab(np.array([[rgb1 / 255.0]], dtype=np.float32))[0][0]
    lab2 = rgb2lab(np.array([[rgb2 / 255.0]], dtype=np.float32))[0][0]
    return np.linalg.norm(lab1 - lab2)

def assign_team(dominant_colors, is_goalkeeper=False, max_dist=30):
    if dominant_colors is None or len(dominant_colors) == 0:
        return -1
    role = 'goalkeeper' if is_goalkeeper else 'player'
    best_team, best_dist = -1, float('inf')
    for team_id, colors in TEAM_COLORS.items():
        target_color = colors[role]
        for dc in dominant_colors:
            dist = color_distance_lab(dc, target_color)
            if dist < best_dist:
                best_dist = dist
                best_team = team_id

    # ✅ Kalau jarak terlalu jauh, jangan assign — kembalikan -1
    if best_dist > max_dist:
        return -1

    return best_team

def find_existing_team_by_color(dominant_colors, is_goalkeeper, confidence_threshold=15):
    """
    Sebelum assign team baru, cek apakah warna ini mirip dengan
    player yang sudah pernah terdeteksi di team tertentu.
    Ini membantu re-ID pemain yang dapat track_id baru.
    """
    role = 'goalkeeper' if is_goalkeeper else 'player'
    
    best_team, best_dist = -1, float('inf')
    for team_id, colors in TEAM_COLORS.items():
        target = colors[role]
        for dc in dominant_colors:
            dist = color_distance_lab(dc, target)
            if dist < best_dist:
                best_dist = dist
                best_team = team_id
    
    # Hanya assign kalau jarak cukup dekat (confident)
    if best_dist < confidence_threshold:
        return best_team
    return -1  # tidak yakin, jangan assign

def process_frame(model, tracker, frame):
    pil_image  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = model.predict(pil_image, threshold=0.3)

    if detections is None or len(detections) == 0:
        return frame.copy()

    # Mapping class_id
    new_class_ids = []
    for raw_id in detections.class_id:
        mapped_id = int(raw_id) - 1
        mapped_id = max(0, min(mapped_id, len(class_names) - 1))
        new_class_ids.append(mapped_id)
    detections.class_id = np.array(new_class_ids)

    # ✅ Pisahkan ball sebelum ByteTrack
    ball_mask        = detections.class_id == class_names.index('ball')
    ball_detections  = detections[ball_mask]
    other_detections = detections[~ball_mask]

    # ✅ ByteTrack untuk player/goalkeeper/referee
    tracked = tracker.update_with_detections(other_detections)

    output = frame.copy()

    for i in range(len(tracked.xyxy)):
        bbox     = tracked.xyxy[i].tolist()
        cls_id   = int(tracked.class_id[i])
        cls_name = class_names[cls_id]
        track_id = int(tracked.tracker_id[i])

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Gambar referee
        if cls_name == 'referee':
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(output, f'REF {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            continue

        if cls_name not in ['player', 'goalkeeper']:
            continue

        is_gk = (cls_name == 'goalkeeper')

        # ✅ Cek memori dulu — kalau sudah pernah di-assign, pakai yang lama
        # ✅ Debug di sini
        if track_id in team_memory:
            team = team_memory[track_id]
            # print(f"  MEMORY HIT: track_id={track_id} ({cls_name}) → Team {team}")
        # Di dalam process_frame, bagian else (belum di-assign):
        else:
            crop, _ = get_center_crop(bbox, frame, crop_ratio=0.3)
            colors  = extract_dominant_colors(crop, n_colors=3)
            colors  = filter_grass_colors(colors)

            if colors is not None:
                # ✅ Coba confident assignment dulu
                team = find_existing_team_by_color(colors, is_gk, confidence_threshold=15)
                
                # Kalau tidak confident, fallback ke assign biasa dengan max_dist lebih longgar
                if team == -1:
                    team = assign_team(colors, is_goalkeeper=is_gk, max_dist=40)

            else:
                team = -1

            if team != -1:
                team_memory[track_id] = team
                print(f"[NEW] track_id={track_id} ({cls_name}) → Team {team}")

        if team == -1:
            # Belum bisa di-assign, gambar abu-abu
            cv2.rectangle(output, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(output, f'? {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            continue

        box_color = (0, 0, 255) if team == 1 else (255, 0, 0)
        label     = f"T{team} {'GK' if is_gk else ''} #{track_id}"

        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(output, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Gambar crop box hijau kecil — hanya untuk track baru
        if track_id not in team_memory:
            _, crop_coords = get_center_crop(bbox, frame, crop_ratio=0.3)
            cx1, cy1, cx2, cy2 = crop_coords
            cv2.rectangle(output, (cx1, cy1), (cx2, cy2), (0, 255, 0), 1)

    # Gambar ball
    for i in range(len(ball_detections.xyxy)):
        bbox = ball_detections.xyxy[i].tolist()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(output, 'BALL', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return output

# ============================================================
# MAIN
# ============================================================

VIDEO_PATH  = 'input_video/input_1.mp4'
OUTPUT_PATH = 'output_videos/team_prediction_v2.mp4'
os.makedirs('output_videos', exist_ok=True)

print("Loading model...")
model   = RFDETRBase(
    pretrain_weights='models/RFDETR Result Dataset With Augmentation Version 1/best_model_new_wrapped.pt',
    num_classes=4
)
model.optimize_for_inference()
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,  # default 0.25 — turunkan kalau banyak track hilang
    lost_track_buffer=150,             # ✅ naikkan ini — ingat track lebih lama (dalam frame)
    minimum_matching_threshold=0.8,   # ✅ naikkan ini — lebih ketat matching
    frame_rate=25                     # sesuaikan dengan fps video kamu
)
print(inspect.signature(sv.ByteTrack.__init__))
print("Tracker created with lost_track_buffer=150")

cap    = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"Processing {total} frames...")

for _ in tqdm(range(total), desc="Team Prediction"):
    ret, frame = cap.read()
    if not ret:
        break
    output_frame = process_frame(model, tracker, frame)
    writer.write(output_frame)

cap.release()
writer.release()

print(f"\nDone! Saved to: {OUTPUT_PATH}")
print(f"Total unique players tracked: {len(team_memory)}")
print(f"Team memory: {team_memory}")