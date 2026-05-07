import numpy as np

# ── Ukuran tactical map lama (TIDAK dipakai oleh renderer baru) ───────────────
# Dibiarkan untuk backward-compat kalau ada import lain
TACTICAL_MAP_WIDTH  = 800
TACTICAL_MAP_HEIGHT = 520

# ── Dimensi lapangan FIFA standar ─────────────────────────────────────────────
PITCH_WIDTH_M  = 105.0   # sumbu panjang lapangan (x_m)
PITCH_HEIGHT_M = 68.0    # sumbu lebar lapangan  (y_m)

# ── Skala lama (horizontal) — biarkan untuk backward-compat ──────────────────
SCALE_X = TACTICAL_MAP_WIDTH  / PITCH_WIDTH_M
SCALE_Y = TACTICAL_MAP_HEIGHT / PITCH_HEIGHT_M

def meter_to_pixel(x_m, y_m):
    """Horizontal (lama) — masih dipakai oleh kode lain."""
    return (int(x_m * SCALE_X), int(y_m * SCALE_Y))


# ── Konstanta canvas VERTIKAL (dipakai oleh TacticalMapRenderer baru) ─────────
# Lapangan di-rotasikan 90°:
#   x_m [0..105] → sumbu vertikal canvas  (py)
#   y_m [0..68]  → sumbu horizontal canvas (px)
#
# Scale = 6 px/m (sama dengan yang dipakai di tactical_map_renderer.py)
_VM_SCALE   = 6
CANVAS_W_VM = int(PITCH_HEIGHT_M * _VM_SCALE)   # 68  * 6 = 408 px
CANVAS_H_VM = int(PITCH_WIDTH_M  * _VM_SCALE)   # 105 * 6 = 630 px

_MARGIN_X_VM = 24
_MARGIN_Y_VM = 32
_FIELD_W_VM  = CANVAS_W_VM - 2 * _MARGIN_X_VM   # 360 px
_FIELD_H_VM  = CANVAS_H_VM - 2 * _MARGIN_Y_VM   # 566 px


def meter_to_pixel_vertical(x_m, y_m):
    """
    Konversi koordinat lapangan (meter) → pixel canvas VERTIKAL.

    Konvensi:
      x_m ∈ [0, 105]  (panjang lapangan) → py (sumbu vertikal canvas)
      y_m ∈ [0, 68]   (lebar lapangan)   → px (sumbu horizontal canvas)

    Fungsi ini dipakai untuk:
      1. Menggambar garis lapangan di TacticalMapRenderer._draw_pitch()
      2. Mendefinisikan PITCH_KEYPOINTS dalam koordinat canvas vertikal
         sehingga HomographyCalculator menghasilkan transform yang benar.
    """
    px = _MARGIN_X_VM + int(y_m / PITCH_HEIGHT_M * _FIELD_W_VM)
    py = _MARGIN_Y_VM + int(x_m / PITCH_WIDTH_M  * _FIELD_H_VM)
    return (px, py)


# ── PITCH_KEYPOINTS dalam koordinat canvas VERTIKAL ───────────────────────────
# PENTING: HomographyCalculator memakai dict ini sebagai dst_pts.
# Koordinat harus sesuai dengan apa yang digambar TacticalMapRenderer._draw_pitch().
#
# Konvensi keypoint SoccerNet (x=panjang, y=lebar lapangan dalam meter):
PITCH_W_M  = 105
PITCH_H_M  = 68

PITCH_KEYPOINTS = {
    # ── Sudut lapangan ─────────────────────────────────────────────────────
    0:  meter_to_pixel_vertical(0,    0),       # pojok kiri atas   (x=0, y=0)
    1:  meter_to_pixel_vertical(0,    68),      # pojok kanan atas  (x=0, y=68)
    2:  meter_to_pixel_vertical(105,  0),       # pojok kiri bawah  (x=105, y=0)
    3:  meter_to_pixel_vertical(105,  68),      # pojok kanan bawah (x=105, y=68)

    # ── Garis tengah ───────────────────────────────────────────────────────
    4:  meter_to_pixel_vertical(52.5, 0),       # tengah garis kiri
    5:  meter_to_pixel_vertical(52.5, 68),      # tengah garis kanan
    6:  meter_to_pixel_vertical(52.5, 34),      # titik tengah lapangan

    # ── Kotak penalti atas (x=0..16.5) ────────────────────────────────────
    7:  meter_to_pixel_vertical(0,    13.84),
    8:  meter_to_pixel_vertical(0,    54.16),
    9:  meter_to_pixel_vertical(16.5, 13.84),
    10: meter_to_pixel_vertical(16.5, 54.16),

    # ── Kotak penalti bawah (x=88.5..105) ────────────────────────────────
    11: meter_to_pixel_vertical(105,  13.84),
    12: meter_to_pixel_vertical(105,  54.16),
    13: meter_to_pixel_vertical(88.5, 13.84),
    14: meter_to_pixel_vertical(88.5, 54.16),

    # ── Kotak gawang atas (x=0..5.5) ─────────────────────────────────────
    15: meter_to_pixel_vertical(0,    24.84),
    16: meter_to_pixel_vertical(0,    43.16),
    17: meter_to_pixel_vertical(5.5,  24.84),
    18: meter_to_pixel_vertical(5.5,  43.16),

    # ── Kotak gawang bawah (x=99.5..105) ──────────────────────────────────
    19: meter_to_pixel_vertical(105,  24.84),
    20: meter_to_pixel_vertical(105,  43.16),
    21: meter_to_pixel_vertical(99.5, 24.84),
    22: meter_to_pixel_vertical(99.5, 43.16),

    # ── Titik penalti ──────────────────────────────────────────────────────
    23: meter_to_pixel_vertical(11,   34),
    24: meter_to_pixel_vertical(94,   34),

    # ── Arc penalti (titik ujung busur) ───────────────────────────────────
    25: meter_to_pixel_vertical(16.5, 34 - 9.0),
    26: meter_to_pixel_vertical(16.5, 34 + 9.0),
    27: meter_to_pixel_vertical(88.5, 34 - 9.0),
    28: meter_to_pixel_vertical(88.5, 34 + 9.0),

    # ── Quarter-line markers ───────────────────────────────────────────────
    29: meter_to_pixel_vertical(26.25, 0),
    30: meter_to_pixel_vertical(78.75, 0),
    31: meter_to_pixel_vertical(26.25, 68),
    32: meter_to_pixel_vertical(78.75, 68),
    33: meter_to_pixel_vertical(52.5,  17),
}

# ── Legacy constants (dipakai beberapa modul lama) ────────────────────────────
PITCH_W_PX = PITCH_W_M * 10   # 1050
PITCH_H_PX = PITCH_H_M * 10   # 680
SCALE      = 10

MODEL_INPUT_H = 240
MODEL_INPUT_W = 240
N_KEYPOINTS   = 34