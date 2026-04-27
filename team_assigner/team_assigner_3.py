import numpy as np
import cv2
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

# ============================================================
# PREDEFINED TEAM COLORS
# PENTING: semua warna di sini dalam format RGB
# (karena crop di-convert ke RGB sebelum KMeans)
# ============================================================
## video input_1: biru tua vs biru muda
# TEAM_COLORS = {
#     1: {
#         'player':     np.array([ 51,  64, 109]),   # RGB: navy blue
#         'goalkeeper': np.array([189,  76, 128]),   # RGB: pink/magenta
#     },
#     2: {
#         'player':     np.array([118, 148, 196]),   # RGB: light blue
#         'goalkeeper': np.array([189,  76, 128]),   # RGB: sama dengan GK team 1
#     }
# }

## video 08fd33_4: hijau terang vs biru muda
TEAM_COLORS = {
    1: {
        'player':     np.array([234, 243, 247]),  # RGB: biru/putih pucat
        'goalkeeper': np.array([206, 130,  99]),  # RGB: oranye/coklat
    },
    2: {
        'player':     np.array([175, 249, 141]),  # RGB: hijau terang lime
        'goalkeeper': np.array([206, 130,  99]),  # RGB: sama (asumsi shared GK color)
    }
}

# Warna anotasi dalam BGR (untuk OpenCV draw)
ANNOTATION_COLORS_BGR = {
    1: (247,  243,  234),   # BGR navy blue
    2: (141, 249, 175),   # BGR light blue
}


class TeamAssigner:
    def __init__(self):
        # Warna anotasi BGR untuk draw_annotations
        self.team_colors = ANNOTATION_COLORS_BGR.copy()
        # Memori permanen: { player_id: team_id }
        self.player_team_dict = {}

    # ----------------------------------------------------------
    # CROP & COLOR EXTRACTION
    # ----------------------------------------------------------

    def get_center_crop(self, bbox, frame, crop_ratio=0.3):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) // 2
        cy = y1 + int(h * 0.35)
        half_w = int(w * crop_ratio / 2)
        half_h = int(h * crop_ratio / 2)
        return frame[
            max(0, cy - half_h): min(frame.shape[0], cy + half_h),
            max(0, cx - half_w): min(frame.shape[1], cx + half_w)
        ]

    def extract_dominant_colors(self, crop, n_colors=3):
        if crop is None or crop.size == 0:
            return None
        # OpenCV frame = BGR → convert ke RGB untuk KMeans
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pixels  = img_rgb.reshape(-1, 3).astype(np.float32)
        if len(pixels) < n_colors:
            return None
        kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42)
        kmeans.fit(pixels)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        sorted_idx = np.argsort(-counts)
        return kmeans.cluster_centers_[sorted_idx]  # urutan: dominan → jarang

    def is_grass_color(self, rgb_color, threshold=40):
        r, g, b = rgb_color
        return (g > r + threshold) and (g > b + threshold)

    def filter_grass_colors(self, colors):
        if colors is None:
            return None
        filtered = [c for c in colors if not self.is_grass_color(c)]
        return np.array(filtered) if filtered else None

    # ----------------------------------------------------------
    # COLOR DISTANCE (LAB color space)
    # ----------------------------------------------------------

    def color_distance_lab(self, rgb1, rgb2):
        """Hitung jarak warna dalam LAB. Input keduanya RGB [0-255]."""
        lab1 = rgb2lab(np.array([[rgb1 / 255.0]], dtype=np.float32))[0][0]
        lab2 = rgb2lab(np.array([[rgb2 / 255.0]], dtype=np.float32))[0][0]
        return np.linalg.norm(lab1 - lab2)

    def assign_team_by_color(self, dominant_colors, is_goalkeeper=False, max_dist=40):
        """
        Bandingkan dominant colors (RGB) dengan TEAM_COLORS (RGB).
        Return team_id (1 atau 2), atau -1 kalau tidak confident.
        """
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

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    def assign_team_color(self, video_frames, player_tracks, sample_frames=10):
        """Diagnostik awal — tidak mengubah TEAM_COLORS."""
        print("[TeamAssigner] Using predefined TEAM_COLORS (RGB).")
        for tid, cols in TEAM_COLORS.items():
            print(f"  Team {tid} player RGB     : {cols['player']}")
            print(f"  Team {tid} goalkeeper RGB : {cols['goalkeeper']}")

        total_frames = len(player_tracks)
        step = max(1, total_frames // sample_frames)
        unassigned = 0

        for frame_num in range(0, total_frames, step):
            for player_id, track in player_tracks[frame_num].items():
                crop   = self.get_center_crop(track['bbox'], video_frames[frame_num])
                colors = self.extract_dominant_colors(crop)
                colors = self.filter_grass_colors(colors)
                if self.assign_team_by_color(colors) == -1:
                    unassigned += 1

        print(f"[TeamAssigner] Diagnostik: {unassigned} deteksi tidak bisa di-assign dari sample.")

    def get_player_team(self, frame, player_bbox, player_id, is_goalkeeper=False):
        """Assign tim ke player. Cache hasil agar tidak re-assign."""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        crop   = self.get_center_crop(player_bbox, frame)
        colors = self.extract_dominant_colors(crop)
        colors = self.filter_grass_colors(colors)
        team   = self.assign_team_by_color(colors, is_goalkeeper=is_goalkeeper)

        if team == -1:
            team = self.assign_team_by_color(colors, is_goalkeeper=is_goalkeeper, max_dist=80)
        if team == -1:
            team = 1  # fallback

        self.player_team_dict[player_id] = team
        print(f"[TeamAssigner] NEW player_id={player_id} → Team {team}")
        return team