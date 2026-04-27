import cv2
import numpy as np
from ultralytics import YOLO

class KeypointDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        print(f"[KeypointDetector] Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.NUM_KP = 29
        print(f"[KeypointDetector] Ready — {self.NUM_KP} keypoints.")

    def detect_keypoints(self, frame):
        orig_h, orig_w = frame.shape[:2]
        results = self.model(frame, verbose=False)[0]

        keypoints = {}
        if results.keypoints is None or len(results.keypoints.data) == 0:
            return keypoints

        # Ambil deteksi lapangan pertama
        kps = results.keypoints.data[0]  # shape (29, 3): x, y, conf

        for idx in range(len(kps)):
            x, y, conf = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
            if conf < self.confidence_threshold:
                continue
            xi, yi = int(x), int(y)
            if 0 <= xi < orig_w and 0 <= yi < orig_h:
                keypoints[idx] = (xi, yi)

        return keypoints

    def detect_keypoints_batch(self, frames):
        all_keypoints = []
        total = len(frames)
        for i, frame in enumerate(frames):
            if i % 50 == 0:
                print(f"  [KeypointDetector] {i}/{total} frames...")
            all_keypoints.append(self.detect_keypoints(frame))
        print(f"  [KeypointDetector] Done — {total} frames processed.")
        return all_keypoints