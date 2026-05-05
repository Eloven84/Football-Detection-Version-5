# inspect_kp.py
# Jalankan: python inspect_kp.py
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "models/soccana_keypoint/Model/weights/kpdet_best_final.pt"
VIDEO_PATH = "input_video/input_vid_3.mp4"
FRAME_IDX  = 500  # frame yang paling bermasalah

# KP names dari Roboflow
KP_NAMES = {
    0:"corner_TL", 1:"penbox_L_top_side", 2:"goalbox_L_top_side",
    3:"goalbox_L_bot_side", 4:"penbox_L_bot_side", 5:"corner_BL",
    6:"goalbox_L_top_inner", 7:"goalbox_L_bot_inner", 8:"penalty_spot_L",
    9:"penbox_L_top_inner", 10:"penarc_L_top_inner", 11:"penarc_L_bot_inner",
    12:"penbox_L_bot_inner", 13:"center_top", 14:"cc_top", 15:"cc_bot",
    16:"center_bot", 17:"penbox_R_top_inner", 18:"penarc_R_top_inner",
    19:"penarc_R_bot_inner", 20:"penbox_R_bot_inner", 21:"penalty_spot_R",
    22:"goalbox_R_top_inner", 23:"goalbox_R_bot_inner", 24:"corner_TR",
    25:"penbox_R_top_side", 26:"goalbox_R_top_side", 27:"goalbox_R_bot_side",
    28:"penbox_R_bot_side", 29:"corner_BR", 30:"cc_left", 31:"cc_right",
}

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ret, frame = cap.read()
cap.release()

results = model(frame, verbose=False)[0]
kps = results.keypoints.data[0]  # (32, 3)

print(f"\n=== RAW KP OUTPUT — Frame {FRAME_IDX} (NO FILTER) ===")
print(f"{'ID':>3} {'Name':25} {'x':>6} {'y':>6} {'conf':>6}  Visible?")
print("-" * 60)
for idx in range(32):
    x, y, conf = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
    visible = "YES" if conf >= 0.5 else "low"
    print(f"{idx:>3} {KP_NAMES[idx]:25} {x:>6.0f} {y:>6.0f} {conf:>6.3f}  {visible}")