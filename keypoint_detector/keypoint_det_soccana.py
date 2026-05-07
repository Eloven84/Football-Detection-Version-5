"""
keypoint_detector.py
Deteksi 32 keypoint lapangan sepakbola menggunakan model Soccana fine-tuned.
Termasuk 3 layer filter:
  1. filter_conflicting_kps   — KP center circle vs sisi konflik
  2. filter_false_center_kps  — false positive KP center saat kamera lateral
  3. filter_outlier_cc_kps    — cc_left / cc_right harus simetris terhadap center
"""

from __future__ import annotations
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Indeks kelompok KP (sesuai definisi Roboflow football-field-detection-f07vi)
# ─────────────────────────────────────────────────────────────────────────────
# KP 0–7   : outer corners & mid-edge (sisi kiri)
# KP 8–15  : garis tengah & center circle area
# KP 16–23 : outer corners & mid-edge (sisi kanan)
# KP 24–28 : kotak penalty
# KP 29    : corner kanan bawah
# KP 30    : cc_left  (center circle kiri)
# KP 31    : cc_right (center circle kanan)

LEFT_SIDE_KPS   = set(range(0, 8))        # KP sisi kiri
RIGHT_SIDE_KPS  = set(range(16, 24))      # KP sisi kanan
CENTER_KPS      = set(range(8, 16))       # KP garis tengah & center
CC_KPS          = {30, 31}                # center-circle kiri/kanan
CENTER_FIELD_KP = 15                      # KP titik tengah lapangan


class KeypointDetector:
    """
    Wrapper YOLO pose model untuk deteksi keypoint lapangan.

    Parameters
    ----------
    model_path : str
        Path ke file .pt model Soccana fine-tuned (32 KP).
    confidence_threshold : float
        Threshold confidence minimum; KP di bawah ini dibuang.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_thr = confidence_threshold

        # Ambil jumlah KP secara dinamis dari model (bukan hardcode)
        self.NUM_KP: int = self.model.model.kpt_shape[0]
        print(f"[KeypointDetector] Loaded model with {self.NUM_KP} keypoints")

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
        """
        Deteksi keypoint dari satu frame.

        Returns
        -------
        keypoints : dict {kp_idx: (x, y)}   — hanya KP dengan conf >= threshold
        raw_conf  : dict {kp_idx: conf}
        """
        results = self.model(frame, verbose=False)
        keypoints: dict[int, tuple[float, float]] = {}
        raw_conf:  dict[int, float]               = {}

        if not results or results[0].keypoints is None:
            return keypoints, raw_conf

        kp_data = results[0].keypoints
        if kp_data.xy is None or len(kp_data.xy) == 0:
            return keypoints, raw_conf

        xy   = kp_data.xy[0].cpu().numpy()    # (NUM_KP, 2)
        conf = kp_data.conf[0].cpu().numpy()  # (NUM_KP,)

        for idx in range(self.NUM_KP):
            c = float(conf[idx])
            if c >= self.conf_thr:
                x, y = float(xy[idx][0]), float(xy[idx][1])
                keypoints[idx] = (x, y)
                raw_conf[idx]  = c

        # Terapkan 3 layer filter
        keypoints = self._filter_conflicting_kps(keypoints)
        keypoints = self._filter_false_center_kps(keypoints)
        keypoints = self._filter_outlier_cc_kps(keypoints, raw_conf)

        return keypoints, raw_conf

    def detect_batch(
        self,
        frames: list[np.ndarray],
        batch_size: int = 8,
    ) -> list[dict[int, tuple[float, float]]]:
        """
        Deteksi keypoint untuk banyak frame sekaligus.

        Returns
        -------
        list of keypoints dict (sudah difilter), satu per frame.
        """
        all_kp: list[dict[int, tuple[float, float]]] = []

        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            results = self.model(batch, verbose=False)

            for res in results:
                kp: dict[int, tuple[float, float]] = {}
                rc: dict[int, float]               = {}

                if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
                    all_kp.append(kp)
                    continue

                xy   = res.keypoints.xy[0].cpu().numpy()
                conf = res.keypoints.conf[0].cpu().numpy()

                for idx in range(self.NUM_KP):
                    c = float(conf[idx])
                    if c >= self.conf_thr:
                        x, y   = float(xy[idx][0]), float(xy[idx][1])
                        kp[idx] = (x, y)
                        rc[idx] = c

                kp = self._filter_conflicting_kps(kp)
                kp = self._filter_false_center_kps(kp)
                kp = self._filter_outlier_cc_kps(kp, rc)
                all_kp.append(kp)

            print(
                f"[KeypointDetector] Processed {min(start + batch_size, len(frames))}"
                f" / {len(frames)} frames"
            )

        return all_kp

    # Alias agar kompatibel dengan pemanggilan lama di main.py
    def detect_keypoints_batch(self, frames: list[np.ndarray]) -> list[dict]:
        return self.detect_batch(frames)

    # ─────────────────────────────────────────────────────────────────────
    # Filter Layer 1 — Konflik KP center circle vs sisi
    # ─────────────────────────────────────────────────────────────────────

    def _filter_conflicting_kps(
        self,
        keypoints: dict[int, tuple[float, float]],
    ) -> dict[int, tuple[float, float]]:
        """
        Jika KP center circle (30/31) terdeteksi bersamaan dengan KP sisi kiri
        DAN kanan, posisi cc bisa ambigu. Hapus cc jika konflik posisi parah.

        Aturan sederhana: cc_left (KP30) harus ada di sebelah kiri cc_right (KP31).
        """
        if 30 in keypoints and 31 in keypoints:
            x30 = keypoints[30][0]
            x31 = keypoints[31][0]
            if x30 >= x31:
                # Tertukar / keduanya palsu — buang keduanya
                keypoints = {k: v for k, v in keypoints.items() if k not in CC_KPS}

        return keypoints

    # ─────────────────────────────────────────────────────────────────────
    # Filter Layer 2 — False positive KP center saat kamera lateral
    # ─────────────────────────────────────────────────────────────────────

    def _filter_false_center_kps(
        self,
        keypoints: dict[int, tuple[float, float]],
    ) -> dict[int, tuple[float, float]]:
        """
        Saat kamera di sisi kanan lapangan:
          - KP sisi kanan terdeteksi, KP sisi kiri TIDAK terdeteksi
          - Model sering confuse garis penalty box kanan dengan center line
          - Solusi: buang CENTER_KPS jika x-nya lebih ke kanan dari KP sisi kanan

        Simetris: hal yang sama berlaku untuk kamera di sisi kiri.
        """
        present_right = [k for k in RIGHT_SIDE_KPS if k in keypoints]
        present_left  = [k for k in LEFT_SIDE_KPS  if k in keypoints]
        present_center = [k for k in CENTER_KPS    if k in keypoints]

        if not present_center:
            return keypoints

        # ── Kamera di sisi kanan: right KP ada, left KP tidak ada ──
        if len(present_right) >= 2 and len(present_left) == 0:
            right_min_x = min(keypoints[k][0] for k in present_right)
            center_xs   = {k: keypoints[k][0] for k in present_center}

            bad_center = [k for k, cx in center_xs.items() if cx > right_min_x * 0.85]
            if bad_center:
                keypoints = {k: v for k, v in keypoints.items()
                             if k not in bad_center and k not in CC_KPS}

        # ── Kamera di sisi kiri: left KP ada, right KP tidak ada ──
        elif len(present_left) >= 2 and len(present_right) == 0:
            left_max_x = max(keypoints[k][0] for k in present_left)
            center_xs  = {k: keypoints[k][0] for k in present_center}

            bad_center = [k for k, cx in center_xs.items() if cx < left_max_x * 1.15]
            if bad_center:
                keypoints = {k: v for k, v in keypoints.items()
                             if k not in bad_center and k not in CC_KPS}

        return keypoints

    # ─────────────────────────────────────────────────────────────────────
    # Filter Layer 3 — Outlier cc_left / cc_right
    # ─────────────────────────────────────────────────────────────────────

    def _filter_outlier_cc_kps(
        self,
        keypoints: dict[int, tuple[float, float]],
        raw_conf:  dict[int, float],
    ) -> dict[int, tuple[float, float]]:
        """
        KP30 (cc_left)  harus di sebelah KIRI  KP15 (field_center).
        KP31 (cc_right) harus di sebelah KANAN KP15 (field_center).
        Jika tidak → buang KP yang salah.
        """
        if CENTER_FIELD_KP not in keypoints:
            return keypoints

        cx15 = keypoints[CENTER_FIELD_KP][0]

        if 30 in keypoints and keypoints[30][0] > cx15:
            del keypoints[30]
            raw_conf.pop(30, None)

        if 31 in keypoints and keypoints[31][0] < cx15:
            del keypoints[31]
            raw_conf.pop(31, None)

        return keypoints