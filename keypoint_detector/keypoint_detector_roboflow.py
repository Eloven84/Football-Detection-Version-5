import torch
from ultralytics import YOLO

class KeypointDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        # ① Tentukan device otomatis
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        print(f"[KeypointDetector] Using device: {self.device}")
        
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.NUM_KP = 32
        print(f"[KeypointDetector] Loaded model: {model_path}")

    def detect_keypoints(self, frame):
        orig_h, orig_w = frame.shape[:2]
        
        # ② Tambahkan device= di sini
        results = self.model(frame, verbose=False, device=self.device)[0]
        
        keypoints = {}
        
        # ③ Guard untuk frame kosong (tidak ada deteksi)
        if results.keypoints is None or len(results.keypoints.data) == 0:
            return keypoints

        SKIP_IDS = {29}
        HIGHER_THRESHOLD_KPS = {27, 30}

        kps = results.keypoints.data[0]
        for class_id, (x, y, conf) in enumerate(kps):
            if class_id in SKIP_IDS:
                continue
            threshold = 0.4 if class_id in HIGHER_THRESHOLD_KPS else self.confidence_threshold
            if conf < threshold:
                continue
            xi, yi = int(x), int(y)
            if 0 < xi < orig_w - 1 and 0 <= yi < orig_h:
                keypoints[class_id] = (xi, yi)
        
        return keypoints

    def detect_keypoints_batch(self, frames, verbose_every=100):
        all_keypoints = []
        for i, frame in enumerate(frames):
            if i % verbose_every == 0:
                print(f"[KeypointDetector] frame {i}/{len(frames)}")
            all_keypoints.append(self.detect_keypoints(frame))
        return all_keypoints