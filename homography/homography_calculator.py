import cv2
import numpy as np
from pitch_config import PITCH_KEYPOINTS

class HomographyCalculator:
    def __init__(self,
                 min_keypoints=4,
                 common_kp_threshold=4,
                 movement_threshold=10.0):
        """
        Args:
            min_keypoints       : minimal keypoints untuk hitung H matrix
            common_kp_threshold : minimal keypoints yang sama dengan frame sebelumnya
                                  agar update condition dievaluasi
            movement_threshold  : rata-rata pergerakan keypoint (pixel) yang
                                  memicu update H matrix
        """
        self.min_keypoints        = min_keypoints
        self.common_kp_threshold  = common_kp_threshold
        self.movement_threshold   = movement_threshold

        self.last_H               = None          # H matrix terakhir yang valid
        self.last_keypoints       = {}            # keypoints dari frame sebelumnya
        self.update_count         = 0             # berapa kali H di-update (debug)

    def compute_homography(self, detected_keypoints):
        """
        Hitung H matrix dari detected_keypoints ke PITCH_KEYPOINTS (top-down).

        detected_keypoints: { keypoint_id: (x, y) } — koordinat di frame video
        PITCH_KEYPOINTS   : { keypoint_id: (x, y) } — koordinat di tactical map

        Return: H matrix (3x3) atau None kalau gagal
        """
        # Cari keypoints yang terdeteksi DAN ada di referensi
        common_ids = [kid for kid in detected_keypoints if kid in PITCH_KEYPOINTS]
        if len(common_ids) < self.min_keypoints:
            return None
        src_pts = np.float32([detected_keypoints[kid] for kid in common_ids])
        dst_pts = np.float32([PITCH_KEYPOINTS[kid] for kid in common_ids])
        
        # Debug sekali saja
        if not hasattr(self, '_kp_debug_done'):
            self._kp_debug_done = True
            print("[H] src_pts (frame coords):", src_pts.tolist())
            print("[H] dst_pts (canvas coords):", dst_pts.tolist())
            print("[H] keypoint IDs used:", common_ids)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def should_update(self, current_keypoints):
        """
        Evaluasi apakah H matrix perlu di-update berdasarkan:
        1. Belum ada H matrix sama sekali → wajib update
        2. Jumlah common keypoints dengan frame sebelumnya < threshold → update
        3. Rata-rata pergerakan keypoints > movement_threshold → update
        4. Kalau tidak ada kondisi di atas → pakai H matrix lama

        Return: True kalau perlu update, False kalau pakai yang lama
        """
        # Kondisi 1: belum ada H matrix
        if self.last_H is None:
            return True
        # Cari keypoints yang sama antara frame ini dan frame sebelumnya
        common_ids = [kid for kid in current_keypoints if kid in self.last_keypoints]
        # Kondisi 2: terlalu sedikit common keypoints
        if len(common_ids) < self.common_kp_threshold:
            return True
        
        # Kondisi 3: hitung rata-rata pergerakan
        # distances = []
        # for kid in common_ids:
        #     prev = np.array(self.last_keypoints[kid])
        #     curr = np.array(current_keypoints[kid])
        #     distances.append(np.linalg.norm(curr - prev))

        # avg_movement = np.mean(distances)
        # print(f"  [H] avg_movement={avg_movement:.4f}, threshold={self.movement_threshold}")
        # return avg_movement > self.movement_threshold
        # Hitung berapa keypoint yang bergerak (bukan rata-rata jarak)
        moved_count = 0
        for kid in common_ids:
            prev = np.array(self.last_keypoints[kid])
            curr = np.array(current_keypoints[kid])
            if np.linalg.norm(curr - prev) >= 1.0:  # bergerak minimal 1 pixel
                moved_count += 1

        # Update kalau lebih dari N keypoints bergerak
        should = moved_count >= self.movement_threshold
        # print(f"  [H] moved_count={moved_count}, threshold={self.movement_threshold}, update={should}")
        return should

    def get_homography(self, current_keypoints):
        """
        Main method — dipanggil tiap frame.
        Return: H matrix yang valid (bisa yang baru atau yang lama)
        """
        if self.should_update(current_keypoints):
            H_new = self.compute_homography(current_keypoints)
            if H_new is not None:
                self.last_H         = H_new
                self.last_keypoints = current_keypoints.copy()
                self.update_count  += 1

        return self.last_H

    # def get_homography(self, current_keypoints):
    #     if self.should_update(current_keypoints):
    #         H_new = self.compute_homography(current_keypoints)
            
    #         if H_new is not None:
    #             self.last_H         = H_new
    #             self.last_keypoints = current_keypoints.copy()
    #             self.update_count  += 1
    #         else:
    #             # DEBUG: cari tahu kenapa compute gagal
    #             common_ids = [kid for kid in current_keypoints if kid in PITCH_KEYPOINTS]
    #             print(f"[Homography DEBUG] should_update=True tapi H gagal dihitung. "
    #                 f"current_kp={len(current_keypoints)}, "
    #                 f"valid_in_PITCH_KP={len(common_ids)}, "
    #                 f"sample_ids={list(current_keypoints.keys())[:5]}")

    #     return self.last_H

    def transform_point(self, point, H):
        """
        Transformasi satu titik (x, y) dari frame plane ke tactical map plane.
        Return: (x', y') di tactical map
        """
        if H is None:
            return None

        pt  = np.float32([[point]]) # shape (1, 1, 2)
        dst = cv2.perspectiveTransform(pt, H)
        return (int(dst[0][0][0]), int(dst[0][0][1]))

    def transform_points_batch(self, points_dict, H):
        """
        Transformasi banyak titik sekaligus.
        points_dict: { id: (x, y) }
        Return: { id: (x', y') }
        """
        if H is None or not points_dict:
            return {}

        ids = list(points_dict.keys())
        pts = np.float32([[points_dict[i]] for i in ids])
        dst = cv2.perspectiveTransform(pts, H)

        return {ids[i]: (int(dst[i][0][0]), int(dst[i][0][1])) for i in range(len(ids))}