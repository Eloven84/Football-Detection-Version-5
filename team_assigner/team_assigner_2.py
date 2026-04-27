import numpy as np
import cv2
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

# ============================================================
# PREDEFINED TEAM COLORS  (format RGB)
# ============================================================

## video 08fd33_4
TEAM_COLORS = {
    1: {
        'player':     np.array([234, 243, 247]),  # biru/putih pucat
        'goalkeeper': np.array([206, 130,  99]),  # oranye/coklat
    },
    2: {
        'player':     np.array([175, 249, 141]),  # hijau lime
        'goalkeeper': np.array([206, 130,  99]),
    }
}

# Warna anotasi BGR — kontras terhadap lapangan hijau
ANNOTATION_COLORS_BGR = {
    1: (247, 243,  234),   # oranye terang
    2: (141, 249,  175),   # hijau lime cerah
}


class TeamAssigner:
    def __init__(self):
        self.team_colors      = ANNOTATION_COLORS_BGR.copy()
        self.player_team_dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    #  CROP & COLOR EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def get_center_crop(self, bbox, frame, crop_ratio=0.3):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        cx   = (x1 + x2) // 2
        cy   = y1 + int(h * 0.35)
        half_w = int(w * crop_ratio / 2)
        half_h = int(h * crop_ratio / 2)
        return frame[
            max(0, cy - half_h): min(frame.shape[0], cy + half_h),
            max(0, cx - half_w): min(frame.shape[1], cx + half_w)
        ]

    def extract_dominant_colors(self, crop, n_colors=3):
        if crop is None or crop.size == 0:
            return None
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pixels  = img_rgb.reshape(-1, 3).astype(np.float32)
        if len(pixels) < n_colors:
            return None
        kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42)
        kmeans.fit(pixels)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        sorted_idx     = np.argsort(-counts)
        return kmeans.cluster_centers_[sorted_idx]

    def is_grass_color(self, rgb_color, threshold=35):
        """
        Deteksi warna rumput lapangan.
        FIX #1: jersey hijau lime Team 2 berbeda dari rumput:
          - Rumput:     G dominan, tapi R & B rendah, saturasinya sedang
          - Hijau lime: G sangat tinggi (>200), R cukup (>140), terang

        Strategi: hanya buang warna yang G jauh lebih tinggi dari R DAN
        kecerahannya rendah (ciri khas rumput gelap).
        """
        r, g, b = rgb_color
        brightness = (r + g + b) / 3

        # Rumput: hijau gelap, R rendah, B rendah, brightness < 150
        is_dark_green = (g > r + threshold) and (g > b + threshold) and (brightness < 150)
        return is_dark_green

    def filter_grass_colors(self, colors):
        if colors is None:
            return None
        filtered = [c for c in colors if not self.is_grass_color(c)]
        return np.array(filtered) if filtered else None

    # ──────────────────────────────────────────────────────────────────────────
    #  COLOR DISTANCE (LAB color space)
    # ──────────────────────────────────────────────────────────────────────────

    def color_distance_lab(self, rgb1, rgb2):
        lab1 = rgb2lab(np.array([[rgb1 / 255.0]], dtype=np.float32))[0][0]
        lab2 = rgb2lab(np.array([[rgb2 / 255.0]], dtype=np.float32))[0][0]
        return np.linalg.norm(lab1 - lab2)

    def assign_team_by_color(self, dominant_colors, is_goalkeeper=False, max_dist=40):
        if dominant_colors is None or len(dominant_colors) == 0:
            return -1

        role = 'goalkeeper' if is_goalkeeper else 'player'
        best_team, best_dist = -1, float('inf')

        for team_id, colors in TEAM_COLORS.items():
            target = colors[role]  # RGB
            for dc in dominant_colors:
                dist = self.color_distance_lab(dc, target)
                if dist < best_dist:
                    best_dist = dist
                    best_team = team_id

        return best_team if best_dist <= max_dist else -1

    # ──────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def assign_team_color(self, video_frames, player_tracks, sample_frames=10):
        print("[TeamAssigner] Using predefined TEAM_COLORS (RGB).")
        for tid, cols in TEAM_COLORS.items():
            print(f"  Team {tid} player RGB     : {cols['player']}")
            print(f"  Team {tid} goalkeeper RGB : {cols['goalkeeper']}")
        print("[TeamAssigner] Annotation colors BGR:")
        for tid, bgr in ANNOTATION_COLORS_BGR.items():
            print(f"  Team {tid}: BGR {bgr}")

        total_frames = len(player_tracks)
        step         = max(1, total_frames // sample_frames)
        unassigned   = 0

        for frame_num in range(0, total_frames, step):
            for player_id, track in player_tracks[frame_num].items():
                crop   = self.get_center_crop(track['bbox'], video_frames[frame_num])
                colors = self.extract_dominant_colors(crop)
                colors = self.filter_grass_colors(colors)
                if self.assign_team_by_color(colors) == -1:
                    unassigned += 1

        self.team_colors = {
            1: (247, 243,  234), # BGR untuk team 1 (biru/putih pucat)
            2: (141, 249,  175)   # BGR untuk team 2 (Hijau Lime)
        }

        print(f"[TeamAssigner] Diagnostik: {unassigned} deteksi tidak bisa di-assign dari sample.")

    # def get_player_team(self, frame, player_bbox, player_id, is_goalkeeper=False):
    #     if player_id in self.player_team_dict:
    #         return self.player_team_dict[player_id]

    #     crop   = self.get_center_crop(player_bbox, frame)
    #     colors = self.extract_dominant_colors(crop)

    #     # TAMBAHAN: Print warna yang terdeteksi
    #     print(f"DEBUG ID {player_id}: Dominant Colors {colors}")
        
    #     filtered_colors = self.filter_grass_colors(colors)
    #     team = self.assign_team_by_color(filtered_colors, is_goalkeeper=is_goalkeeper)

    #     colors = self.filter_grass_colors(colors)
    #     team   = self.assign_team_by_color(colors, is_goalkeeper=is_goalkeeper)

    #     if team == -1:
    #         team = self.assign_team_by_color(colors, is_goalkeeper=is_goalkeeper, max_dist=80)
    #     if team == -1:
    #         team = 1  # fallback

    #     self.player_team_dict[player_id] = team
    #     print(f"[TeamAssigner] NEW player_id={player_id} → Team {team}")
    #     return team

    def get_player_team(self, frame, player_bbox, player_id, is_goalkeeper=False):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        crop   = self.get_center_crop(player_bbox, frame)
        colors = self.extract_dominant_colors(crop)
        colors = self.filter_grass_colors(colors)
        
        # Coba assign dengan jarak ketat (40)
        team = self.assign_team_by_color(colors, is_goalkeeper=is_goalkeeper, max_dist=40)
        
        # Jika gagal, coba dengan jarak lebih longgar (80)
        if team == -1:
            team = self.assign_team_by_color(colors, is_goalkeeper=is_goalkeeper, max_dist=80)
        
        # Jika tetap gagal, berikan ID 0 (Unknown) daripada memaksa ke 1
        if team == -1:
            team = 0 

        self.player_team_dict[player_id] = team
        return team