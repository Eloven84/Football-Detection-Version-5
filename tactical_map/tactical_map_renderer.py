import cv2
import numpy as np
from pitch_config import TACTICAL_MAP_WIDTH, TACTICAL_MAP_HEIGHT, PITCH_WIDTH_M, PITCH_HEIGHT_M

# ── Ukuran tactical map VERTIKAL ──────────────────────────────────────────────
_SCALE    = 6
CANVAS_W  = int(PITCH_HEIGHT_M * _SCALE)   # 408 px
CANVAS_H  = int(PITCH_WIDTH_M  * _SCALE)   # 630 px

_MARGIN_X = 24
_MARGIN_Y = 32
_FIELD_W  = CANVAS_W - 2 * _MARGIN_X       # 360 px
_FIELD_H  = CANVAS_H - 2 * _MARGIN_Y       # 566 px


def _m_to_px_vertical(x_m, y_m):
    px = _MARGIN_X + int(y_m / PITCH_HEIGHT_M * _FIELD_W)
    py = _MARGIN_Y + int(x_m / PITCH_WIDTH_M  * _FIELD_H)
    return (px, py)


# ── Trajectory config ─────────────────────────────────────────────────────────
TRAJECTORY_MAX_POINTS = 60    # berapa posisi bola yang disimpan
TRAJECTORY_FADE_STEPS = 10    # titik paling ujung sudah sangat transparan


class TacticalMapRenderer:
    def __init__(self):
        self.canvas_w = CANVAS_W
        self.canvas_h = CANVAS_H

        self.team_colors = {
            1:  (235, 120,  30),
            2:  ( 60, 230,  60),
            -1: (128, 128, 128),
        }
        self.ball_color    = (255, 255, 255)
        self.referee_color = (0,   255, 255)

        # ── Trajectory state ──────────────────────────────────────────────────
        # Simpan history posisi bola (koordinat canvas sudah di-transform)
        self._ball_trail = []   # list of (x, y) — canvas coords

        self.pitch_background = self._draw_pitch()

    # ──────────────────────────────────────────────────────────────────────────
    #  DRAW PITCH BACKGROUND
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_pitch(self):
        img = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        for row in range(self.canvas_h):
            t = row / self.canvas_h
            g = int(80 + 30 * t)
            img[row, :] = (35, g, 35)

        lc = (200, 200, 200)
        t  = 2

        tl = _m_to_px_vertical(0,   0)
        br = _m_to_px_vertical(105, 68)
        cv2.rectangle(img, tl, br, lc, t)

        mid_l = _m_to_px_vertical(52.5, 0)
        mid_r = _m_to_px_vertical(52.5, 68)
        cv2.line(img, mid_l, mid_r, lc, t)

        center = _m_to_px_vertical(52.5, 34)
        r_px   = int(9.15 / PITCH_WIDTH_M * _FIELD_H)
        cv2.circle(img, center, r_px, lc, t)
        cv2.circle(img, center, 4, lc, -1)

        cv2.rectangle(img,
                      _m_to_px_vertical(0,    13.84),
                      _m_to_px_vertical(16.5, 54.16), lc, t)
        cv2.rectangle(img,
                      _m_to_px_vertical(0,   24.84),
                      _m_to_px_vertical(5.5, 43.16), lc, t)

        cv2.rectangle(img,
                      _m_to_px_vertical(88.5, 13.84),
                      _m_to_px_vertical(105,  54.16), lc, t)
        cv2.rectangle(img,
                      _m_to_px_vertical(99.5, 24.84),
                      _m_to_px_vertical(105,  43.16), lc, t)

        cv2.circle(img, _m_to_px_vertical(11,  34), 5, lc, -1)
        cv2.circle(img, _m_to_px_vertical(94,  34), 5, lc, -1)

        r_arc = int(9.15 / PITCH_WIDTH_M * _FIELD_H)
        cv2.ellipse(img, _m_to_px_vertical(11, 34), (r_arc, r_arc), 0,  53,  127, lc, t)
        cv2.ellipse(img, _m_to_px_vertical(94, 34), (r_arc, r_arc), 0, 233,  307, lc, t)

        gw_left  = 34 - 3.66
        gw_right = 34 + 3.66
        cv2.rectangle(img,
                      _m_to_px_vertical(-2.0, gw_left),
                      _m_to_px_vertical(0,    gw_right),
                      (255, 255, 255), 2)
        cv2.rectangle(img,
                      _m_to_px_vertical(105,   gw_left),
                      _m_to_px_vertical(107.0, gw_right),
                      (255, 255, 255), 2)

        return img

    # ──────────────────────────────────────────────────────────────────────────
    #  TRAJECTORY HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _update_trail(self, ball_pos):
        """
        Tambah posisi bola ke trail. Jika None, tidak ditambahkan
        (trail tidak diputus — bola mungkin sementara tidak terdeteksi).
        """
        if ball_pos is None:
            return
        bx, by = int(ball_pos[0]), int(ball_pos[1])
        # Hanya simpan jika masih dalam canvas
        if 0 <= bx < self.canvas_w and 0 <= by < self.canvas_h:
            self._ball_trail.append((bx, by))
            # Batasi panjang trail
            if len(self._ball_trail) > TRAJECTORY_MAX_POINTS:
                self._ball_trail.pop(0)

    def _draw_trail(self, frame):
        """
        Gambar trail bola dengan efek fade (makin tua = makin transparan & kecil).

        Strategi: gambar di overlay terpisah lalu blend ke frame
        supaya alpha per-segmen benar-benar halus.
        """
        n = len(self._ball_trail)
        if n < 2:
            return

        overlay = frame.copy()

        for i in range(1, n):
            # age: 0 = titik terbaru, 1 = tertua
            age  = (n - 1 - i) / max(n - 1, 1)
            alpha = 1.0 - age                        # 1.0 di ujung baru, ~0 di ujung lama

            # Warna: putih kekuningan → oranye → merah gelap sesuai usia
            r = 255
            g = int(255 * alpha)
            b = int(100 * alpha)
            color = (b, g, r)                        # BGR

            # Ketebalan & radius makin tipis untuk titik lama
            thickness = max(1, int(3 * alpha))
            radius    = max(1, int(4 * alpha))

            pt1 = self._ball_trail[i - 1]
            pt2 = self._ball_trail[i]

            # Garis penghubung antar titik
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

            # Titik kecil di setiap posisi (membuat trail terlihat lebih halus)
            cv2.circle(overlay, pt2, radius, color, -1, cv2.LINE_AA)

        # Blend overlay ke frame — trail agak transparan agar tidak menutupi pemain
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ──────────────────────────────────────────────────────────────────────────
    #  RENDER SATU FRAME
    # ──────────────────────────────────────────────────────────────────────────

    def render(self, player_positions, ball_position=None, keypoint_positions=None):
        """
        Render satu frame tactical map.

        player_positions  : { player_id: {'map_pos': (x,y), 'team': 1|2, 'role': str} }
        ball_position     : (x, y) di tactical map canvas, atau None
        keypoint_positions: None = sembunyikan debug dots;
                            dict { kp_id: (x,y) } = tampilkan untuk debug

        Return: frame BGR shape (CANVAS_H, CANVAS_W, 3)
        """
        frame = self.pitch_background.copy()

        # ── Update & gambar trajectory bola ──────────────────────────────────
        self._update_trail(ball_position)
        self._draw_trail(frame)

        # ── Debug keypoints ───────────────────────────────────────────────────
        if keypoint_positions:
            for kp_id, pos in keypoint_positions.items():
                if pos is None:
                    continue
                x, y = pos
                if 0 <= x < self.canvas_w and 0 <= y < self.canvas_h:
                    cv2.circle(frame, (x, y), 3, (0, 0, 200), -1)

        # ── Draw players & goalkeepers ────────────────────────────────────────
        for player_id, info in player_positions.items():
            pos  = info.get('map_pos')
            team = info.get('team', -1)
            role = info.get('role', 'player')
            if pos is None:
                continue

            x, y = int(pos[0]), int(pos[1])

            # ── FIX: validasi bounds sebelum menggambar ───────────────────────
            # Tambah margin kecil agar pemain di tepi lapangan tetap terlihat
            margin = 15
            if not (-margin <= x < self.canvas_w + margin and
                    -margin <= y < self.canvas_h + margin):
                continue
            # Clamp ke canvas supaya circle tidak crash OpenCV di luar bounds
            x = max(0, min(x, self.canvas_w - 1))
            y = max(0, min(y, self.canvas_h - 1))

            color  = self.team_colors.get(team, (128, 128, 128))
            radius = 9 if role == 'goalkeeper' else 7

            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)

            if role == 'goalkeeper':
                cv2.circle(frame, (x, y), radius + 3, (255, 255, 255), 1)

        # ── Draw ball ─────────────────────────────────────────────────────────
        if ball_position is not None:
            bx, by = int(ball_position[0]), int(ball_position[1])
            if 0 <= bx < self.canvas_w and 0 <= by < self.canvas_h:
                pts = np.array([
                    [bx,     by - 9],
                    [bx - 6, by + 4],
                    [bx + 6, by + 4]
                ], np.int32)
                cv2.fillPoly(frame, [pts], self.ball_color)
                cv2.polylines(frame, [pts], True, (50, 50, 50), 1)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  RESET TRAIL (opsional — panggil jika memproses video baru)
    # ──────────────────────────────────────────────────────────────────────────

    def reset_trail(self):
        """Kosongkan history trajectory bola."""
        self._ball_trail = []