import cv2
import os
import numpy as np
from rfdetr import RFDETRBase
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────
WEIGHTS    = r"E:\FOLDER TUGAS AKHIR 2\Football2\models\RFDETR Result Dataset With Augmentation Version 1\checkpoint_best_total.pth"
INPUT_VID  = r"E:\FOLDER TUGAS AKHIR 2\Football2\input_video\input_5.mp4"
OUTPUT_DIR = r"E:\FOLDER TUGAS AKHIR 2\Football2\runs"
OUTPUT_VID = os.path.join(OUTPUT_DIR, "output_input_5.mp4")

# Sama persis dengan notebook
full_classes = ['player-ball-goalkeeper-referee-QfA4', 'ball', 'goalkeeper', 'player', 'referee']
class_names  = full_classes[1:]   # ['ball', 'goalkeeper', 'player', 'referee']

CLASS_COLORS = {
    0: (0, 255, 255),   # ball       → kuning
    1: (255, 0, 0),     # goalkeeper → biru
    2: (0, 255, 0),     # player     → hijau
    3: (0, 0, 255),     # referee    → merah
}
THRESHOLD = 0.3
# ────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = RFDETRBase(pretrain_weights=WEIGHTS)
model.optimize_for_inference()

cap    = cv2.VideoCapture(INPUT_VID)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VID, fourcc, fps, (width, height))

print(f"Processing {total} frames...")

frame_idx   = 0
all_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pil_image  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = model.predict(pil_image, threshold=THRESHOLD)
    all_results.append(detections)

    if detections is not None and len(detections) > 0:
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            conf     = float(detections.confidence[i])
            raw_id   = int(detections.class_id[i])

            # ✅ Mapping sama persis dengan notebook
            mapped_id = (raw_id % len(full_classes)) - 1

            # Skip jika mapped_id di luar range (misal hasil -1)
            if mapped_id < 0 or mapped_id >= len(class_names):
                continue

            name  = class_names[mapped_id]
            color = CLASS_COLORS[mapped_id]
            label = f"{name}: {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    writer.write(frame)
    frame_idx += 1

    if frame_idx % 50 == 0:
        print(f"  Frame {frame_idx}/{total}")

cap.release()
writer.release()

print(f"\nSelesai! Video disimpan di: {OUTPUT_VID}")
print(f"Total frame diproses: {frame_idx}")
print("======================")

# Tampilkan hasil frame pertama
first = all_results[0] if all_results else None
if first is not None and len(first) > 0:
    print("Deteksi frame pertama:")
    for i in range(len(first.xyxy)):
        raw_id    = int(first.class_id[i])
        mapped_id = (raw_id % len(full_classes)) - 1
        if mapped_id < 0 or mapped_id >= len(class_names):
            continue
        conf = float(first.confidence[i])
        box  = first.xyxy[i]
        print(f"  [{class_names[mapped_id]}] conf={conf:.2f} box={box}")
else:
    print("Tidak ada hasil deteksi di frame pertama")

"""
## 📌 Penjelasan Mapping

raw class_id dari model  →  % 5  →  - 1  →  nama kelas
─────────────────────────────────────────────────────────
8   →  8 % 5 = 3   →  3 - 1 = 2  →  player
13  →  13 % 5 = 3  →  3 - 1 = 2  →  player
18  →  18 % 5 = 3  →  3 - 1 = 2  →  player
23  →  23 % 5 = 3  →  3 - 1 = 2  →  player
33  →  33 % 5 = 3  →  3 - 1 = 2  →  player
3   →  3 % 5 = 3   →  3 - 1 = 2  →  player
4   →  4 % 5 = 4   →  4 - 1 = 3  →  referee
1   →  1 % 5 = 1   →  1 - 1 = 0  →  ball
2   →  2 % 5 = 2   →  2 - 1 = 1  →  goalkeeper
"""