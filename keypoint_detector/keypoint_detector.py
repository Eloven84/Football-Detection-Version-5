import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase

class KeypointDetector:
    def __init__(self, model_path, confidence_threshold=0.4):
        """
        Model ini adalah model TERPISAH yang ditraining khusus
        untuk deteksi keypoints lapangan.
        """
        self.model = RFDETRBase(
            pretrain_weights=model_path,
            num_classes=29   # 29 kelas = 29 jenis keypoint lapangan
        )
        self.model.optimize_for_inference()
        self.confidence_threshold = confidence_threshold

    def detect_keypoints(self, frame):
        """
        Deteksi keypoints dari satu frame.
        Return: dict { keypoint_id: (x, y) }
        """
        pil_image  = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        detections = self.model.predict(pil_image, threshold=self.confidence_threshold)

        if detections is None or len(detections) == 0:
            return {}

        keypoints = {}
        for i in range(len(detections.xyxy)):
            bbox     = detections.xyxy[i]
            cls_id   = int(detections.class_id[i])
            # Ambil center point dari bbox sebagai koordinat keypoint
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            keypoints[cls_id] = (cx, cy)

        return keypoints

    def detect_keypoints_batch(self, video_frames):
        """Deteksi keypoints untuk semua frame sekaligus."""
        all_keypoints = []
        for frame in video_frames:
            kp = self.detect_keypoints(frame)
            all_keypoints.append(kp)
        return all_keypoints