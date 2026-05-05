import cv2
import numpy as np
from pitch_config_roboflow import PITCH_KEYPOINTS

class HomographyCalculator:
    def __init__(self, min_keypoints=4, common_kp_threshold=4,
                 movement_threshold=2, smooth_window=5,
                 max_reproj_error=80.0,   # FIX 2: naik dari 30 → 50
                 max_reuse_frames=60):    # sudah 60, pertahankan
        self.min_keypoints       = min_keypoints
        self.common_kp_threshold = common_kp_threshold
        self.movement_threshold  = movement_threshold
        self.smooth_window       = smooth_window
        self.max_reproj_error    = max_reproj_error
        self.max_reuse_frames    = max_reuse_frames

        self.last_H       = None
        self.H_history    = []
        self.update_count = 0
        self.reuse_count  = 0

        # diagnostic counters
        self._col_count        = 0
        self._ransac_fail_count = 0
        self._reproj_fail_count = 0
        self._valid_H_fail_count = 0  # FIX 1: counter baru

    def _is_valid_homography(self, H, canvas_w=1050, canvas_h=680):
        """
        FIX 1: margin dinaikkan dari 2000 → 5000.
        Kamera lapangan hanya meliput sebagian pitch, sehingga
        pojok frame yang di luar lapangan bisa transform jauh.
        """
        test_pts = np.array([
            [0, 0], [1920, 0], [1920, 1080], [0, 1080]
        ], dtype=np.float32)

        try:
            transformed = cv2.perspectiveTransform(
                test_pts.reshape(-1, 1, 2), H
            ).reshape(-1, 2)
        except Exception:
            return False

        margin = 10000  # FIX: 2000 → 5000
        for x, y in transformed:
            if abs(x) > canvas_w + margin or abs(y) > canvas_h + margin:
                return False

        det = np.linalg.det(H[:2, :2])
        if det < 0.01:
            return False

        return True

    # def _compute_raw_H(self, detected_kp: dict):
    #     common_ids = [k for k in detected_kp if k in PITCH_KEYPOINTS]
    #     if len(common_ids) < self.min_keypoints:
    #         return None, float('inf')

    #     src_pts = np.array([detected_kp[k]     for k in common_ids], dtype=np.float32)
    #     dst_pts = np.array([PITCH_KEYPOINTS[k] for k in common_ids], dtype=np.float32)

    #     if self._is_collinear(src_pts):
    #         self._col_count += 1
    #         return None, float('inf')

    #     H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)
    #     if H is None:
    #         self._ransac_fail_count += 1
    #         return None, float('inf')

    #     if not self._is_valid_homography(H):
    #         self._valid_H_fail_count += 1
    #         return None, float('inf')

    #     src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    #     proj  = (H @ src_h.T).T
    #     proj  = proj[:, :2] / proj[:, 2:3]
    #     err   = np.mean(np.linalg.norm(proj - dst_pts, axis=1))

    #     if err >= self.max_reproj_error:
    #         self._reproj_fail_count += 1
    #         return None, float('inf')

    #     return H, err

    def _compute_raw_H(self, detected_kp: dict, kp_confidences: dict = None):
        common_ids = [k for k in detected_kp if k in PITCH_KEYPOINTS]

        # --- TAMBAH: filter by confidence ---
        if kp_confidences:
            MIN_CONF = 0.3
            common_ids = [k for k in common_ids if kp_confidences.get(k, 0) >= MIN_CONF]

        if len(common_ids) < self.min_keypoints:
            return None, float('inf')

        src_pts = np.array([detected_kp[k]     for k in common_ids], dtype=np.float32)
        dst_pts = np.array([PITCH_KEYPOINTS[k] for k in common_ids], dtype=np.float32)

        if self._is_collinear(src_pts):
            self._col_count += 1
            return None, float('inf')
        
        if not self._has_enough_spread(src_pts, dst_pts, min_spread_px=80):
            return None, float('inf')

        # --- TAMBAH: weighted RANSAC berdasarkan confidence ---
        if kp_confidences:
            weights = np.array([kp_confidences.get(k, 0.1) for k in common_ids], dtype=np.float32)
            weights = weights / weights.sum()
        else:
            weights = None

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            self._ransac_fail_count += 1
            return None, float('inf')

        if not self._is_valid_homography(H):
            self._valid_H_fail_count += 1
            return None, float('inf')

        # reproj error hanya pada inliers
        if mask is not None:
            inlier_ids = [common_ids[i] for i in range(len(common_ids)) if mask[i]]
            if len(inlier_ids) < 4:
                return None, float('inf')
            src_in = np.array([detected_kp[k] for k in inlier_ids], dtype=np.float32)
            dst_in = np.array([PITCH_KEYPOINTS[k] for k in inlier_ids], dtype=np.float32)
        else:
            src_in, dst_in = src_pts, dst_pts

        src_h = np.hstack([src_in, np.ones((len(src_in), 1))])
        proj  = (H @ src_h.T).T
        proj  = proj[:, :2] / proj[:, 2:3]
        err   = np.mean(np.linalg.norm(proj - dst_in, axis=1))

        if err >= self.max_reproj_error:
            self._reproj_fail_count += 1
            return None, float('inf')

        return H, err

    def _is_collinear(self, pts, threshold=0.95):  # naik dari 0.90 → 0.95
        if len(pts) < 4:
            return True
        centered = pts - pts.mean(axis=0)
        _, s, _ = np.linalg.svd(centered)
        ratio = s[1] / (s[0] + 1e-8)
        print(f"  [collinear check] SVD ratio: {ratio:.3f} (threshold={1-threshold:.2f})")
        return ratio < (1 - threshold)

    def _has_enough_spread(self, src_pts, dst_pts, min_spread_px=80):
        """Pastikan dst_pts (canvas) punya spread cukup di KEDUA axis."""
        xs = dst_pts[:, 0]
        ys = dst_pts[:, 1]
        spread_x = xs.max() - xs.min()
        spread_y = ys.max() - ys.min()
        print(f"  [spread check] dst spread_x={spread_x:.0f} spread_y={spread_y:.0f}")
        return spread_x >= min_spread_px and spread_y >= min_spread_px

    def _update_smooth_H(self, H_new):
        self.H_history.append(H_new)
        if len(self.H_history) > self.smooth_window:
            self.H_history.pop(0)
        n = len(self.H_history)
        weights = np.array([0.5 ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        return sum(w * H for w, H in zip(weights, self.H_history))

    def get_homography(self, detected_kp: dict):
        H_new, err = self._compute_raw_H(detected_kp)

        if H_new is not None:
            # H baru valid → update, reset reuse counter
            H_smooth = self._update_smooth_H(H_new)
            self.last_H    = H_smooth
            self.update_count += 1
            self.reuse_count   = 0          # FIX 3: reset di sini
            return H_smooth
        else:
            # FIX 3: hanya naikkan reuse_count jika frame ini memang tidak
            # menghasilkan H baru — jangan reset ke 0 (sudah benar),
            # tapi pastikan tidak naik dobel
            self.reuse_count += 1
            if self.last_H is not None and self.reuse_count <= self.max_reuse_frames:
                return self.last_H
            else:
                return None

    def transform_point(self, point, H):
        if H is None or point is None:
            return None
        pt = np.array([[point[0], point[1]]], dtype=np.float32).reshape(-1, 1, 2)
        try:
            transformed = cv2.perspectiveTransform(pt, H)
        except Exception:
            return None
        if transformed is None:
            return None
        x, y = int(transformed[0][0][0]), int(transformed[0][0][1])
        return (x, y)