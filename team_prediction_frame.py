import cv2
import numpy as np
import random
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv

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
# HELPER FUNCTIONS (sama seperti sebelumnya)
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

def assign_team(dominant_colors, is_goalkeeper=False):
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
    return best_team

# ============================================================
# FUNGSI UTAMA — bisa dipanggil untuk model apapun
# ============================================================

def run_team_prediction(model, frame, output_path, num_classes):
    pil_image  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = model.predict(pil_image, threshold=0.3)

    new_class_ids = []
    for raw_id in detections.class_id:
        mapped_id = int(raw_id) - 1
        mapped_id = max(0, min(mapped_id, len(class_names) - 1))
        new_class_ids.append(mapped_id)
    detections.class_id = np.array(new_class_ids)

    output = frame.copy()

    for i in range(len(detections.xyxy)):
        bbox     = detections.xyxy[i].tolist()
        cls_id   = int(detections.class_id[i])
        cls_name = class_names[cls_id]

        if cls_name not in ['player', 'goalkeeper']:
            continue

        is_gk = (cls_name == 'goalkeeper')
        crop, crop_coords = get_center_crop(bbox, frame, crop_ratio=0.3)
        colors = extract_dominant_colors(crop, n_colors=3)
        colors = filter_grass_colors(colors)
        team   = assign_team(colors, is_goalkeeper=is_gk)

        box_color = (0, 0, 255) if team == 1 else (255, 0, 0)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(output, f"T{team} {'GK' if is_gk else ''}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        cx1, cy1, cx2, cy2 = crop_coords
        cv2.rectangle(output, (cx1, cy1), (cx2, cy2), (0, 255, 0), 1)

    cv2.imwrite(output_path, output)
    print(f"Saved: {output_path}")

# ============================================================
# MAIN
# ============================================================

# Ambil frame random yang sama untuk kedua model
cap = cv2.VideoCapture('input_video/08fd33_4.mp4')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
random_frame_num = random.randint(0, total_frames - 1)
print(f"Testing frame: {random_frame_num} / {total_frames}")
cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_num)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Gagal baca frame")
    exit()

# ── Model Eli (checkpoint_best_total.pth) ──
print("\n[Model Eli] Loading...")
model_eli = RFDETRBase(
    pretrain_weights='models/RFDETR Result Dataset With Augmentation Version 1/checkpoint_best_total.pth',
    num_classes=4
)
run_team_prediction(model_eli, frame, 'output_team_test_eli.jpg', num_classes=4)

# ── Model Jemi (best_model_wrapped.pt) ──
print("\n[Model Jemi] Loading...")
model_jemi = RFDETRBase(
    pretrain_weights='models/RFDETR Result Dataset With Augmentation Version 1/best_model_new_wrapped.pt',
    num_classes=4
)
run_team_prediction(model_jemi, frame, 'output_team_test_jemi.jpg', num_classes=4)