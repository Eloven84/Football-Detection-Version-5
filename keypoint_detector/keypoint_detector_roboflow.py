# keypoint_detector.py (versi baru — Roboflow API)
import cv2
import base64
import requests

class KeypointDetector:
    def __init__(self, api_key, confidence_threshold=0.5,
                 model_id="football-field-detection-f07vi/14"):
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self.model_id = model_id
        self.NUM_KP = 32
        print(f"[KeypointDetector] Ready — Roboflow API, {self.NUM_KP} keypoints.")

    def detect_keypoints(self, frame):
        orig_h, orig_w = frame.shape[:2]

        # Encode frame ke base64
        _, buffer = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        # Kirim ke Roboflow API
        response = requests.post(
            f"https://detect.roboflow.com/{self.model_id}",
            params={"api_key": self.api_key},
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        result = response.json()

        keypoints = {}
        predictions = result.get("predictions", [])
        if not predictions:
            return keypoints

        # # Ambil keypoints dari pitch pertama
        # for kp in predictions[0].get("keypoints", []):
        #     class_id = kp["class_id"]
        #     x = kp["x"]
        #     y = kp["y"]
        #     conf = kp["confidence"]

        #     if conf < self.confidence_threshold:
        #         continue
        #     if x <= 0:  # x=0 artinya tidak terdeteksi
        #         continue

        #     xi, yi = int(x), int(y)
        #     if 0 <= xi < orig_w and 0 <= yi < orig_h:
        #         keypoints[class_id] = (xi, yi)

        # Di detect_keypoints(), ganti bagian filter confidence:

        HIGHER_THRESHOLD_KPS = {27, 30}  # titik yang sering false-positive di tepi frame

        for kp in predictions[0].get("keypoints", []):
            class_id = kp["class_id"]
            x, y, conf = kp["x"], kp["y"], kp["confidence"]

            # kp 28 & 29 selalu tidak valid untuk video ini
            if class_id in (28, 29):
                continue

            # kp 27 & 30 butuh threshold lebih tinggi
            threshold = 0.4 if class_id in HIGHER_THRESHOLD_KPS else self.confidence_threshold

            if conf < threshold:
                continue
            if x <= 0 or x >= 1919:   # buang titik yang nempel di tepi frame
                continue

            xi, yi = int(x), int(y)
            if 0 <= xi < orig_w and 0 <= yi < orig_h:
                keypoints[class_id] = (xi, yi)

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