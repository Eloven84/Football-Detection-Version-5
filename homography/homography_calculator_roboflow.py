import cv2
import numpy as np
from pitch_config_roboflow import PITCH_KEYPOINTS

class HomographyCalculator:
    def __init__(self, min_keypoints=4, common_kp_threshold=4,
                 movement_threshold=2, smooth_window=5,
                 max_reproj_error=15.0, max_reuse_frames=30):
        self.min_keypoints      = min_keypoints
        self.common_kp_threshold= common_kp_threshold
        self.movement_threshold = movement_threshold
        self.smooth_window      = smooth_window
        self.max_reproj_error   = max_reproj_error
        self.max_reuse_frames   = max_reuse_frames

        self.last_H         = None
        self.H_history      = []       # untuk smoothing
        self.update_count   = 0
        self.reuse_count    = 0        # berapa frame sudah pakai last_H

    # ── Internal ────────────────────────────────────────────────────────────

    def _compute_raw_H(self, detected_kp: dict):
        """Hitung H dari keypoints yang terdeteksi. Return (H, err) atau (None, inf)."""
        common_ids = [k for k in detected_kp if k in PITCH_KEYPOINTS]
        if len(common_ids) < self.min_keypoints:
            return None, float('inf')

        src_pts = np.array(
            [detected_kp[k] for k in common_ids], dtype=np.float32
        )
        dst_pts = np.array(
            [PITCH_KEYPOINTS[k] for k in common_ids], dtype=np.float32
        )

        # Cek collinearity — kalau semua titik hampir satu garis, H tidak reliable
        if self._is_collinear(src_pts):
            return None, float('inf')

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None, float('inf')

        # Hitung reprojection error
        src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
        proj  = (H @ src_h.T).T
        proj  = proj[:, :2] / proj[:, 2:3]
        err   = np.mean(np.linalg.norm(proj - dst_pts, axis=1))

        return H, err

    def _is_collinear(self, pts, threshold=0.98):
        """Return True kalau titik-titik hampir collinear (tidak cukup untuk homography)."""
        if len(pts) < 4:
            return True
        centered = pts - pts.mean(axis=0)
        _, s, _ = np.linalg.svd(centered)
        # Kalau singular value terkecil sangat kecil relatif terhadap terbesar → collinear
        return (s[1] / (s[0] + 1e-8)) < (1 - threshold)

    def _update_smooth_H(self, H_new):
        """Masukkan H baru ke history dan return rata-rata (smoothed)."""
        self.H_history.append(H_new)
        if len(self.H_history) > self.smooth_window:
            self.H_history.pop(0)
        return np.mean(self.H_history, axis=0)

    # ── Public ───────────────────────────────────────────────────────────────

    def get_homography(self, detected_kp: dict):
        """
        Return H matrix yang valid dan smooth.
        Kalau tidak bisa hitung H baru, pakai last_H (max max_reuse_frames frame).
        """
        H_new, err = self._compute_raw_H(detected_kp)

        if H_new is not None and err < self.max_reproj_error:
            H_smooth = self._update_smooth_H(H_new)
            self.last_H   = H_smooth
            self.update_count += 1
            self.reuse_count   = 0
            return H_smooth
        else:
            # Pakai last_H hanya jika belum terlalu lama
            self.reuse_count += 1
            if self.last_H is not None and self.reuse_count <= self.max_reuse_frames:
                return self.last_H
            else:
                # Sudah terlalu lama tanpa H valid → jangan render
                return None

    def transform_point(self, point, H):
        """Transform satu titik dengan H. Return None kalau H is None."""
        if H is None or point is None:
            return None
        pt = np.array([[point[0], point[1]]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pt, H)
        if transformed is None:
            return None
        x, y = int(transformed[0][0][0]), int(transformed[0][0][1])
        return (x, y)