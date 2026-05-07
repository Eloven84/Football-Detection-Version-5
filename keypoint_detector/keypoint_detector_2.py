import numpy as np
import tensorflow as tf
import cv2


class KeypointDetector:
    def __init__(self, model_path='model_pitch', confidence_threshold=0.05):
        print(f"[KeypointDetector] Loading model from: {model_path}")
        self.model  = tf.saved_model.load(model_path)
        self.infer  = self.model.signatures['serving_default']
        self.confidence_threshold = confidence_threshold

        # Nama input & output sudah diketahui dari debug
        self.input_name  = 'input_4'      # shape (None, 240, 240, 3)
        self.output_name = 'reshape_1'    # shape (1, 34, 3)
        self.MODEL_H     = 240
        self.MODEL_W     = 240
        self.NUM_KP      = 34

        print(f"[KeypointDetector] Ready — {self.NUM_KP} keypoints, "
              f"input {self.MODEL_W}x{self.MODEL_H}.")

    # ------------------------------------------------------------------ #

    def preprocess(self, frame):
        """BGR frame → normalized float32 tensor (1, 240, 240, 3)."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.MODEL_W, self.MODEL_H))
        img = img.astype(np.float32) / 255.0
        return tf.constant(img[np.newaxis])   # (1, 240, 240, 3)

    # ------------------------------------------------------------------ #

    def detect_keypoints(self, frame):
        """
        Deteksi keypoint pada satu frame.

        Return: list of (x, y) dalam koordinat piksel frame asli.
                Keypoint dengan confidence < threshold dibuang.
        """
        orig_h, orig_w = frame.shape[:2]
        output = self.infer(**{self.input_name: self.preprocess(frame)})
        raw    = output[self.output_name].numpy()[0]  # (34, 3)

        # DEBUG: print confidence semua keypoint di frame pertama
        if not hasattr(self, '_debug_done'):
            self._debug_done = True
            all_confs = [(i, float(raw[i][2])) for i in range(len(raw))]
            all_confs.sort(key=lambda x: -x[1])
            print("\n[KP CONFIDENCE DEBUG — semua 34 keypoints]")
            for idx, conf in all_confs:
                x_n, y_n = float(raw[idx][0]), float(raw[idx][1])
                print(f"  kp_{idx:02d}: conf={conf:.4f}, "
                    f"x_norm={x_n:.3f}, y_norm={y_n:.3f}, "
                    f"in_frame={'YES' if 0<=x_n<=1 and 0<=y_n<=1 else 'NO'}")
            print()

        keypoints = {}
        for idx, (x_norm, y_norm, conf) in enumerate(raw):
            if conf < self.confidence_threshold:
                continue
            if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
                continue
            x = int(x_norm * orig_w)
            y = int(y_norm * orig_h)
            if not (0 <= x < orig_w and 0 <= y < orig_h):
                continue
            keypoints[idx] = (x, y)

        return keypoints

    # ------------------------------------------------------------------ #

    def detect_keypoints_batch(self, frames):
        """
        Deteksi keypoint untuk semua frame sekaligus.

        Return: list of list of (x, y), panjang = len(frames).
        """
        all_keypoints = []
        total = len(frames)

        for i, frame in enumerate(frames):
            if i % 50 == 0:
                print(f"  [KeypointDetector] {i}/{total} frames...")
            all_keypoints.append(self.detect_keypoints(frame))

        print(f"  [KeypointDetector] Done — {total} frames processed.")
        return all_keypoints