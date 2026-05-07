import cv2
import numpy as np
from pitch_config_Soccana import CANVAS_W, CANVAS_H, MARGIN_X, MARGIN_Y, FIELD_W, FIELD_H, m_to_px, PITCH_WIDTH_M, PITCH_HEIGHT_M


# Canvas vertikal — ukuran diambil langsung dari pitch_config
# supaya renderer dan PITCH_KEYPOINTS selalu sinkron
# CANVAS_W = CANVAS_W_VM   # 408 px
# CANVAS_H = CANVAS_H_VM   # 630 px

# ── Trajectory config ─────────────────────────────────────────────────────────
# TRAJECTORY_MAX_POINTS = 60    # berapa posisi bola yang disimpan
# TRAJECTORY_FADE_STEPS = 10    # titik paling ujung sudah sangat transparan

class TacticalMapRenderer:
    def __init__(self):
        self.canvas_w = CANVAS_W   # 1050
        self.canvas_h = CANVAS_H   # 680

        # Warna per team (BGR) — harus sama dengan ANNOTATION_COLORS_BGR di team_assigner
        self.team_colors = {
            1:  (235, 120,  30),    # oranye
            2:  ( 60, 230,  60),    # hijau lime
            -1: (128, 128, 128),    # unassigned
        }
        self.ball_color    = (255, 255, 255)
        self.referee_color = (0,   255, 255)

        # ── Trajectory state ──────────────────────────────────────────────────
        # Simpan history posisi bola (koordinat canvas sudah di-transform)
        # self._ball_trail = []   # list of (x, y) — canvas coords

        self.pitch_background = self._draw_pitch()

    # ──────────────────────────────────────────────────────────────────────────
    #  DRAW PITCH BACKGROUND
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_pitch(self):
        img = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        for row in range(self.canvas_h):
            t = row / self.canvas_h
            g = int(78 + 28 * t)
            img[row, :] = (30, g, 30)

        lc = (200, 200, 200)
        tk = 2
        m2p = m_to_px

        # ── Garis tepi ──────────────────────────────────────────────────────────
        cv2.rectangle(img, m2p(0, 0), m2p(105, 68), lc, tk)

        # ── Garis tengah (vertikal di canvas horizontal) ─────────────────────────
        cv2.line(img, m2p(52.5, 0), m2p(52.5, 68), lc, tk)

        # ── Lingkaran tengah — PERBAIKAN: pakai ellipse karena skala x≠y ─────────
        center = m2p(52.5, 34)
        r_px_x = int(9.15 / PITCH_WIDTH_M  * FIELD_W)   # radius horizontal
        r_px_y = int(9.15 / PITCH_HEIGHT_M * FIELD_H)   # radius vertikal
        cv2.ellipse(img, center, (r_px_x, r_px_y), 0, 0, 360, lc, tk)
        cv2.circle(img, center, 4, lc, -1)

        # ── Kotak penalti KIRI (x=0..16.5) ──────────────────────────────────────
        cv2.rectangle(img, m2p(0,    13.84), m2p(16.5, 54.16), lc, tk)
        cv2.rectangle(img, m2p(0,    24.84), m2p(5.5,  43.16), lc, tk)

        # ── Kotak penalti KANAN (x=88.5..105) ───────────────────────────────────
        cv2.rectangle(img, m2p(88.5, 13.84), m2p(105,  54.16), lc, tk)
        cv2.rectangle(img, m2p(99.5, 24.84), m2p(105,  43.16), lc, tk)

        # ── Titik penalti ────────────────────────────────────────────────────────
        cv2.circle(img, m2p(11,  34), 5, lc, -1)
        cv2.circle(img, m2p(94,  34), 5, lc, -1)

        # ── Busur penalti — PERBAIKAN: pakai axes yang benar ────────────────────
        axes = (r_px_x, r_px_y)
        # Busur kiri: menghadap ke kanan (dalam lapangan), sekitar 53° dari vertikal
        cv2.ellipse(img, m2p(11, 34), axes, 0, -53, 53, lc, tk)
        # Busur kanan: menghadap ke kiri (dalam lapangan)
        cv2.ellipse(img, m2p(94, 34), axes, 0, 127, 233, lc, tk)

        # ── Gawang — PERBAIKAN: gambar di dalam batas canvas ───────────────────
        # Gawang kiri: garis tebal di x=0, lebar gawang 7.32m (y=30.34..37.66)
        pt1_l = m2p(0, 30.34)
        pt2_l = m2p(0, 37.66)
        cv2.line(img, pt1_l, pt2_l, (255, 255, 255), 5)

        # Gawang kanan: garis tebal di x=105
        pt1_r = m2p(105, 30.34)
        pt2_r = m2p(105, 37.66)
        cv2.line(img, pt1_r, pt2_r, (255, 255, 255), 5)

        return img
    
    # ──────────────────────────────────────────────────────────────────────────
    #  TRAJECTORY HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    # def _update_trail(self, ball_pos):
    #     """
    #     Tambah posisi bola ke trail. Jika None, tidak ditambahkan
    #     (trail tidak diputus — bola mungkin sementara tidak terdeteksi).
    #     """
    #     if ball_pos is None:
    #         return
    #     bx, by = int(ball_pos[0]), int(ball_pos[1])
    #     # Hanya simpan jika masih dalam canvas
    #     if 0 <= bx < self.canvas_w and 0 <= by < self.canvas_h:
    #         self._ball_trail.append((bx, by))
    #         # Batasi panjang trail
    #         if len(self._ball_trail) > TRAJECTORY_MAX_POINTS:
    #             self._ball_trail.pop(0)

    # def _draw_trail(self, frame, trail):
    #     """Gambar trail kumulatif dengan fade — titik lama pudar."""
    #     # Filter None
    #     valid = [(i, p) for i, p in enumerate(trail) if p is not None]
    #     if len(valid) < 2:
    #         return

    #     overlay = frame.copy()
    #     n = len(valid)

    #     for j in range(1, n):
    #         i_prev, pt1 = valid[j - 1]
    #         i_curr, pt2 = valid[j]

    #         # age: 0.0 = titik terlama, 1.0 = titik terbaru
    #         age = j / max(n - 1, 1)

    #         # Fade: titik lama sangat transparan
    #         # Hanya gambar jika age cukup tinggi (buang titik sangat lama)
    #         min_age = max(0.0, 1.0 - 120 / max(n, 1))  # tampilkan 120 titik terakhir
    #         if age < min_age:
    #             continue

    #         # Normalisasi alpha dalam window yang ditampilkan
    #         alpha = (age - min_age) / max(1.0 - min_age, 1e-6)

    #         # Warna: merah gelap → oranye → kuning (makin baru makin terang)
    #         r = 255
    #         g = int(200 * alpha)
    #         b = 0
    #         color = (b, g, r)  # BGR

    #         thickness = max(1, int(3 * alpha))
    #         cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

    #         # Titik kecil di posisi terbaru
    #         if j == n - 1:
    #             cv2.circle(overlay, pt2, 4, (0, 220, 255), -1, cv2.LINE_AA)

    #     cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # def _draw_trail(self, frame, trail):
    #     valid = [(i, p) for i, p in enumerate(trail) if p is not None]
    #     if len(valid) < 2:
    #         return

    #     overlay = frame.copy()
    #     n = len(valid)

    #     # Tampilkan SEMUA trail — tapi titik lama lebih tipis & pudar
    #     SHOW_LAST = 150   # berapa titik terakhir yang "terang penuh"
    #     FADE_OVER = 9999  # sisanya tetap digambar tapi sangat pudar

    #     for j in range(1, n):
    #         _, pt1 = valid[j - 1]
    #         _, pt2 = valid[j]

    #         # Hitung dari belakang: 0 = titik paling baru
    #         age_from_end = n - 1 - j

    #         if age_from_end < SHOW_LAST:
    #             # Window terang: gradient penuh
    #             alpha = 1.0 - (age_from_end / SHOW_LAST) * 0.7   # 1.0 → 0.3
    #         else:
    #             # Titik lama: sangat pudar tapi tetap ada
    #             alpha = 0.08

    #         r = 255
    #         g = int(200 * min(alpha * 1.5, 1.0))
    #         b = 0
    #         color = (b, g, r)  # BGR

    #         thickness = max(1, int(3 * alpha))
    #         cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

    #     # Titik bola terbaru — highlight cyan
    #     if valid:
    #         _, last_pt = valid[-1]
    #         cv2.circle(overlay, last_pt, 4, (0, 220, 255), -1, cv2.LINE_AA)

    #     cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    def _draw_trail(self, frame, trail, max_segment_px=80):
        """Gambar trail dengan filter lompatan ekstrem antar titik."""
        valid = [(i, p) for i, p in enumerate(trail) if p is not None]
        if len(valid) < 2:
            return

        overlay = frame.copy()
        n = len(valid)

        SHOW_LAST = 150

        for j in range(1, n):
            _, pt1 = valid[j - 1]
            _, pt2 = valid[j]

            # ── Filter lompatan ekstrem ─────────────────────────────
            dist = ((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2) ** 0.5
            if dist > max_segment_px:
                continue  # ← jangan gambar garis kalau lompatan terlalu jauh

            age_from_end = n - 1 - j

            if age_from_end < SHOW_LAST:
                alpha = 1.0 - (age_from_end / SHOW_LAST) * 0.7
            else:
                alpha = 0.08

            r = 255
            g = int(200 * min(alpha * 1.5, 1.0))
            b = 0
            color = (b, g, r)

            thickness = max(1, int(3 * alpha))
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

        if valid:
            _, last_pt = valid[-1]
            cv2.circle(overlay, last_pt, 4, (0, 220, 255), -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # ──────────────────────────────────────────────────────────────────────────
    #  RENDER SATU FRAME
    # ──────────────────────────────────────────────────────────────────────────

    def render(self, player_positions, ball_position=None, keypoint_positions=None, ball_trail=None):
        """
        Render satu frame tactical map.

        player_positions  : { player_id: {'map_pos': (x,y), 'team': 1|2, 'role': str} }
        ball_position     : (x, y) di canvas atau None
        keypoint_positions: dict { kp_id: (x,y) } untuk debug, atau None untuk hide
        """
        frame = self.pitch_background.copy()

        # ── Update & draw trajectory ───────────────────────────────────────
        # self._update_trail(ball_position)   # ← TAMBAHKAN
        # self._draw_trail(frame)             # ← TAMBAHKAN

        # ── Draw trail kumulatif ───────────────────────────────────────────
        if ball_trail:
            self._draw_trail(frame, ball_trail)

        # ── Debug keypoints (merah kecil) ──────────────────────────────────
        if keypoint_positions:
            for kp_id, pos in keypoint_positions.items():
                if pos is None:
                    continue
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < self.canvas_w and 0 <= y < self.canvas_h:
                    cv2.circle(frame, (x, y), 3, (0, 0, 200), -1)

        # ── Draw players & goalkeepers ─────────────────────────────────────
        for player_id, info in player_positions.items():
            pos  = info.get('map_pos')
            team = info.get('team', -1)
            role = info.get('role', 'player')
            if pos is None:
                continue
            x, y = int(pos[0]), int(pos[1])
            if not (0 <= x < self.canvas_w and 0 <= y < self.canvas_h):
                continue

            color  = self.team_colors.get(team, (128, 128, 128))
            radius = 9 if role == 'goalkeeper' else 7

            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)

            if role == 'goalkeeper':
                cv2.circle(frame, (x, y), radius + 3, (255, 255, 255), 1)

        # ── Draw ball ──────────────────────────────────────────────────────
        if ball_position is not None:
            bx, by = int(ball_position[0]), int(ball_position[1])
            if 0 <= bx < self.canvas_w and 0 <= by < self.canvas_h:
                pts = np.array([
                    [bx, by - 9],
                    [bx - 6, by + 4],
                    [bx + 6, by + 4]
                ], np.int32)
                cv2.fillPoly(frame, [pts], self.ball_color)
                cv2.polylines(frame, [pts], True, (50, 50, 50), 1)

        return frame
    
    # ──────────────────────────────────────────────────────────────────────────
    #  RESET TRAIL (opsional — panggil jika memproses video baru)
    # ──────────────────────────────────────────────────────────────────────────

    # def reset_trail(self):
    #     """Kosongkan history trajectory bola."""
    #     self._ball_trail = []