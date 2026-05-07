"""
tactical_map.py
Render tactical map canvas dengan unit konsisten (cm dari SoccerPitchConfiguration).

Semua dimensi garis lapangan (penalty box, center circle, dsb.)
dihitung dari PITCH_LENGTH_CM / PITCH_WIDTH_CM, BUKAN hardcode meter.
"""

from __future__ import annotations
import cv2
import numpy as np

try:
    from sports.configs.soccer import SoccerPitchConfiguration
    _CONFIG          = SoccerPitchConfiguration()
    PITCH_LENGTH_CM  = int(np.array(_CONFIG.vertices)[:, 0].max())  # 12000
    PITCH_WIDTH_CM   = int(np.array(_CONFIG.vertices)[:, 1].max())  # 7000
except ImportError:
    # Fallback jika sports tidak tersedia
    PITCH_LENGTH_CM = 12000
    PITCH_WIDTH_CM  = 7000

# Dimensi lapangan standar dalam cm (FIFA)
PENALTY_BOX_LENGTH_CM = 1650   # 16.5 m
PENALTY_BOX_WIDTH_CM  = 4032   # 40.32 m  (2 × 20.16)
GOAL_BOX_LENGTH_CM    = 550    # 5.5 m
GOAL_BOX_WIDTH_CM     = 1832   # 18.32 m
CENTER_CIRCLE_R_CM    = 915    # 9.15 m
PENALTY_SPOT_CM       = 1100   # 11 m dari garis gawang


class TacticalMapRenderer:
    """
    Render overlay taktis (lapangan + pemain + bola + trail).

    Parameters
    ----------
    canvas_w, canvas_h : int
        Ukuran canvas output dalam pixel.
        Default 960 × 560 (rasio ≈ 12:7, sesuai rasio pitch FIFA).
    """

    def __init__(self, canvas_w: int = 960, canvas_h: int = 560):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        # Warna (BGR)
        self.COLOR_GRASS_DARK  = (34,  139, 34)
        self.COLOR_GRASS_LIGHT = (50,  168, 50)
        self.COLOR_LINE        = (255, 255, 255)
        self.COLOR_BALL        = (0,   220, 255)
        self.COLOR_TRAIL       = (0,   180, 220)
        self.COLOR_TEAM1       = (220, 50,  50)   # merah
        self.COLOR_TEAM2       = (50,  50,  220)  # biru
        self.COLOR_GK          = (255, 165, 0)    # oranye
        self.COLOR_UNKNOWN     = (180, 180, 180)

    # ─────────────────────────────────────────────────────────────────────
    # Helper: cm → pixel canvas
    # ─────────────────────────────────────────────────────────────────────

    def _cx(self, x_cm: float) -> int:
        """Konversi koordinat X (cm) → pixel canvas."""
        return int(x_cm / PITCH_LENGTH_CM * self.canvas_w)

    def _cy(self, y_cm: float) -> int:
        """Konversi koordinat Y (cm) → pixel canvas."""
        return int(y_cm / PITCH_WIDTH_CM * self.canvas_h)

    def _cr(self, r_cm: float) -> int:
        """Konversi radius (cm) → pixel (pakai sumbu panjang sebagai referensi)."""
        return int(r_cm / PITCH_LENGTH_CM * self.canvas_w)

    # ─────────────────────────────────────────────────────────────────────
    # Draw pitch background + lines
    # ─────────────────────────────────────────────────────────────────────

    def _draw_pitch(self, canvas: np.ndarray) -> None:
        W, H = self.canvas_w, self.canvas_h

        # Background rumput (strip vertikal selang-seling)
        n_stripes = 10
        for i in range(n_stripes):
            x1 = int(i / n_stripes * W)
            x2 = int((i + 1) / n_stripes * W)
            color = self.COLOR_GRASS_DARK if i % 2 == 0 else self.COLOR_GRASS_LIGHT
            cv2.rectangle(canvas, (x1, 0), (x2, H), color, -1)

        lw = max(1, int(W * 0.003))  # lebar garis proporsional

        # ── Border lapangan ──
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), self.COLOR_LINE, lw)

        # ── Garis tengah ──
        mid_x = W // 2
        cv2.line(canvas, (mid_x, 0), (mid_x, H), self.COLOR_LINE, lw)

        # ── Center circle ──
        r_cc = self._cr(CENTER_CIRCLE_R_CM)
        cv2.circle(canvas, (mid_x, H // 2), r_cc, self.COLOR_LINE, lw)
        cv2.circle(canvas, (mid_x, H // 2), max(2, lw), self.COLOR_LINE, -1)

        # ── Penalty box kiri ──
        pb_w = self._cy(PENALTY_BOX_WIDTH_CM)
        pb_l = self._cx(PENALTY_BOX_LENGTH_CM)
        pb_top_y    = (H - pb_w) // 2
        pb_bottom_y = pb_top_y + pb_w
        cv2.rectangle(canvas, (0, pb_top_y), (pb_l, pb_bottom_y), self.COLOR_LINE, lw)

        # ── Penalty box kanan ──
        cv2.rectangle(canvas, (W - pb_l, pb_top_y), (W, pb_bottom_y), self.COLOR_LINE, lw)

        # ── Goal box kiri ──
        gb_w = self._cy(GOAL_BOX_WIDTH_CM)
        gb_l = self._cx(GOAL_BOX_LENGTH_CM)
        gb_top_y    = (H - gb_w) // 2
        gb_bottom_y = gb_top_y + gb_w
        cv2.rectangle(canvas, (0, gb_top_y), (gb_l, gb_bottom_y), self.COLOR_LINE, lw)

        # ── Goal box kanan ──
        cv2.rectangle(canvas, (W - gb_l, gb_top_y), (W, gb_bottom_y), self.COLOR_LINE, lw)

        # ── Penalty spot kiri & kanan ──
        ps = self._cx(PENALTY_SPOT_CM)
        spot_r = max(2, lw + 1)
        cv2.circle(canvas, (ps, H // 2), spot_r, self.COLOR_LINE, -1)
        cv2.circle(canvas, (W - ps, H // 2), spot_r, self.COLOR_LINE, -1)

    # ─────────────────────────────────────────────────────────────────────
    # Draw pemain
    # ─────────────────────────────────────────────────────────────────────

    def _draw_players(
        self,
        canvas: np.ndarray,
        player_positions: dict,
    ) -> None:
        """
        player_positions : dict
            key   = player_id (str atau int)
            value = {
                'map_pos' : (cx, cy) dalam pixel canvas | None,
                'team'    : int (1 atau 2, -1 = unknown),
                'role'    : 'player' | 'goalkeeper',
            }
        """
        r_player = max(4, int(self.canvas_w * 0.009))
        r_gk     = max(5, int(self.canvas_w * 0.011))

        for pid, info in player_positions.items():
            pos = info.get("map_pos")
            if pos is None:
                continue

            cx, cy = int(pos[0]), int(pos[1])
            team   = info.get("team", -1)
            role   = info.get("role", "player")

            if role == "goalkeeper":
                color = self.COLOR_GK
                r     = r_gk
            elif team == 1:
                color = self.COLOR_TEAM1
                r     = r_player
            elif team == 2:
                color = self.COLOR_TEAM2
                r     = r_player
            else:
                color = self.COLOR_UNKNOWN
                r     = r_player

            cv2.circle(canvas, (cx, cy), r,     color,          -1)
            cv2.circle(canvas, (cx, cy), r + 1, (255, 255, 255), 1)  # outline putih

    # ─────────────────────────────────────────────────────────────────────
    # Draw bola + trail
    # ─────────────────────────────────────────────────────────────────────

    def _draw_ball(
        self,
        canvas: np.ndarray,
        ball_position:  tuple[int, int] | None,
        ball_trail:     list[tuple[int, int] | None] | None = None,
        max_trail_len:  int = 30,
    ) -> None:
        # Trail
        if ball_trail:
            trail = [p for p in ball_trail[-max_trail_len:] if p is not None]
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                thick = max(1, int(alpha * 3))
                color = tuple(int(c * alpha) for c in self.COLOR_TRAIL)
                cv2.line(canvas, trail[i - 1], trail[i], color, thick)

        # Bola saat ini
        if ball_position is not None:
            bx, by = int(ball_position[0]), int(ball_position[1])
            r_ball = max(4, int(self.canvas_w * 0.008))
            cv2.circle(canvas, (bx, by), r_ball,     self.COLOR_BALL,      -1)
            cv2.circle(canvas, (bx, by), r_ball + 1, (255, 255, 255),        1)

    # ─────────────────────────────────────────────────────────────────────
    # Public: render satu frame canvas taktis
    # ─────────────────────────────────────────────────────────────────────

    def render(
        self,
        player_positions: dict,
        ball_position:    tuple[int, int] | None           = None,
        ball_trail:       list[tuple[int, int] | None] | None = None,
    ) -> np.ndarray:
        """
        Render canvas taktis untuk satu frame.

        Parameters
        ----------
        player_positions : dict  (lihat _draw_players untuk format)
        ball_position    : (cx, cy) pixel canvas | None
        ball_trail       : list posisi bola historis (pixel canvas) | None

        Returns
        -------
        canvas : np.ndarray  BGR, shape (canvas_h, canvas_w, 3)
        """
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self._draw_pitch(canvas)
        self._draw_ball(canvas, ball_position, ball_trail)
        self._draw_players(canvas, player_positions)
        return canvas