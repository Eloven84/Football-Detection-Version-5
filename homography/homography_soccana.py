"""
homography.py
Menghitung homography dari keypoint frame → koordinat pitch (cm).

Menggunakan:
  - SoccerPitchConfiguration dari roboflow/sports untuk ground truth 32 KP
  - ViewTransformer (cv2.findHomography wrapper) sebagai solver
  - HomographyTracker untuk fallback antar frame dengan threshold 500 cm
"""

from __future__ import annotations
import cv2
import numpy as np

try:
    from sports.configs.soccer import SoccerPitchConfiguration
    _CONFIG   = SoccerPitchConfiguration()
    VERTICES  = np.array(_CONFIG.vertices, dtype=np.float32)  # (32, 2), unit cm
except ImportError:
    raise ImportError(
        "Package 'sports' tidak ditemukan. "
        "Install dengan: pip install roboflow-sports"
    )

# Dimensi pitch dalam cm (dari SoccerPitchConfiguration)
PITCH_LENGTH_CM = int(VERTICES[:, 0].max())   # sumbu X  — biasanya 12000
PITCH_WIDTH_CM  = int(VERTICES[:, 1].max())   # sumbu Y  — biasanya 7000


# ─────────────────────────────────────────────────────────────────────────────
# ViewTransformer — thin wrapper cv2.findHomography
# ─────────────────────────────────────────────────────────────────────────────

class ViewTransformer:
    """
    Hitung homography dari source points (px frame) ke target points (cm pitch).

    Parameters
    ----------
    source : np.ndarray  shape (N, 2)   koordinat di frame (pixel)
    target : np.ndarray  shape (N, 2)   koordinat di pitch (cm)
    """

    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform array of points (N, 2) menggunakan homography matrix.
        Returns (N, 2) float32 dalam satuan target (cm).
        """
        if self.m is None:
            return np.full_like(points, np.nan, dtype=np.float32)

        pts = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.m)
        return transformed.reshape(-1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi reprojection error (dipakai internal & debug)
# ─────────────────────────────────────────────────────────────────────────────

def reproj_error_cm(
    keypoints_frame: dict[int, tuple[float, float]],
    H: np.ndarray,
) -> float:
    """
    Hitung reprojection error dalam satuan CM.
    Dipakai oleh HomographyTracker untuk menilai kualitas H.
    """
    if H is None:
        return float("inf")

    errors = []
    for idx, (fx, fy) in keypoints_frame.items():
        if idx >= len(VERTICES):
            continue
        src = np.array([[[fx, fy]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)[0][0]
        ref = VERTICES[idx]
        errors.append(float(np.linalg.norm(dst - ref)))

    return float(np.mean(errors)) if errors else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# HomographyTracker — pertahankan H antar frame, fallback jika jelek
# ─────────────────────────────────────────────────────────────────────────────

class HomographyTracker:
    """
    Menyimpan H terbaik sejauh ini dan hanya update jika:
      - H baru valid (tidak None)
      - reproj_error_cm <= max_reproj_error_cm
      - jumlah KP >= min_kp
    """

    def __init__(
        self,
        min_kp: int           = 4,
        max_reproj_error_cm: float = 500.0
    ):
        self.min_kp              = min_kp
        self.max_reproj_error_cm = max_reproj_error_cm
        self._best_H: np.ndarray | None = None
        self.update_count = 0

    def update(
        self,
        keypoints: dict[int, tuple[float, float]],
        H_candidate: np.ndarray | None,
    ) -> np.ndarray | None:
        """
        Coba update H dengan kandidat baru.
        Selalu kembalikan H terbaik yang tersedia (bisa None di awal).
        """
        if H_candidate is None or len(keypoints) < self.min_kp:
            return self._best_H

        err = reproj_error_cm(keypoints, H_candidate)
        if err <= self.max_reproj_error_cm:
            self._best_H = H_candidate
            self.update_count += 1

        return self._best_H

    @property
    def current_H(self) -> np.ndarray | None:
        return self._best_H


# ─────────────────────────────────────────────────────────────────────────────
# HomographyCalculator — antarmuka utama untuk main.py
# ─────────────────────────────────────────────────────────────────────────────

class HomographyCalculator:
    """
    Antarmuka utama yang dipanggil dari main.py.

    Alur per frame:
      1. Hitung H baru dari keypoints saat ini (ViewTransformer)
      2. Validasi via HomographyTracker
      3. Kembalikan H terbaik (fallback ke frame sebelumnya jika perlu)
      4. transform_point() → cm → pixel canvas

    Parameters
    ----------
    min_keypoints        : int    minimum KP untuk mencoba hitung H
    common_kp_threshold  : int    (legacy, tidak dipakai — tetap ada agar kompatibel)
    movement_threshold   : float  (legacy, tidak dipakai — tetap ada agar kompatibel)
    canvas_w, canvas_h   : int    ukuran canvas output taktis (px)
                                  Default mengikuti rasio 12000:7000 → 960×560
    """

    def __init__(
        self,
        min_keypoints:       int   = 4,
        common_kp_threshold: int   = 4,   # legacy compat
        movement_threshold:  float = 2.0, # legacy compat
        canvas_w:            int   = 960,
        canvas_h:            int   = 560,
        max_reproj_error_cm: float = 500.0,
    ):
        self.min_keypoints = min_keypoints
        self.canvas_w      = canvas_w
        self.canvas_h      = canvas_h

        self._tracker = HomographyTracker(
            min_kp=min_keypoints,
            max_reproj_error_cm=max_reproj_error_cm,
        )

    # ── property agar main.py bisa akses update_count ──
    @property
    def update_count(self) -> int:
        return self._tracker.update_count

    # ─────────────────────────────────────────────────────────────────────
    # Compute H untuk satu frame
    # ─────────────────────────────────────────────────────────────────────

    def get_homography(
        self,
        keypoints: dict[int, tuple[float, float]],
    ) -> np.ndarray | None:
        """
        Hitung + validasi H dari keypoints frame saat ini.
        Kembalikan H terbaik yang tersedia (mungkin dari frame sebelumnya).
        """
        H_new = self._compute_H(keypoints)
        H     = self._tracker.update(keypoints, H_new)
        return H

    def _compute_H(
        self,
        keypoints: dict[int, tuple[float, float]],
    ) -> np.ndarray | None:
        """
        Internal: gunakan ViewTransformer untuk hitung H.
        source = pixel frame, target = cm pitch (VERTICES).
        """
        valid_idx = [i for i in keypoints if i < len(VERTICES)]
        if len(valid_idx) < self.min_keypoints:
            return None

        frame_pts = np.array([keypoints[i]    for i in valid_idx], dtype=np.float32)
        pitch_pts = np.array([VERTICES[i]     for i in valid_idx], dtype=np.float32)

        try:
            vt = ViewTransformer(source=frame_pts, target=pitch_pts)
            return vt.m  # matrix (3,3) float64
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Transform satu titik: frame px → cm → canvas px
    # ─────────────────────────────────────────────────────────────────────

    def transform_point(
        self,
        point: tuple[float, float],
        H: np.ndarray | None,
        canvas_w: int | None = None,
        canvas_h: int | None = None,
        margin: int          = 5,
    ) -> tuple[int, int] | None:
        """
        Transform satu titik posisi (px frame) → posisi di canvas taktis (px).

        Pipeline:
          frame px  →[H]→  pitch cm  →[scale]→  canvas px
        """
        if H is None:
            return None

        canvas_w = canvas_w or self.canvas_w
        canvas_h = canvas_h or self.canvas_h

        src = np.array([[[point[0], point[1]]]], dtype=np.float32)
        try:
            dst_cm = cv2.perspectiveTransform(src, H)[0][0]  # (x_cm, y_cm)
        except Exception:
            return None

        # Scale cm → canvas pixel
        cx = dst_cm[0] / PITCH_LENGTH_CM * canvas_w
        cy = dst_cm[1] / PITCH_WIDTH_CM  * canvas_h

        # Tolak titik di luar canvas (dengan margin)
        if not (-margin <= cx <= canvas_w + margin and
                -margin <= cy <= canvas_h + margin):
            return None

        return (int(cx), int(cy))