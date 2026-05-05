# pitch_config_roboflow.py — GANTI PITCH_KEYPOINTS dengan ini

# from sports.configs.soccer import SoccerPitchConfiguration

# config = SoccerPitchConfiguration()

PITCH_WIDTH_M  = 105.0
PITCH_HEIGHT_M = 68.0
CANVAS_W = 1050
CANVAS_H = 680
MARGIN_X = 50
MARGIN_Y = 50
FIELD_W  = CANVAS_W - 2 * MARGIN_X   # 950
FIELD_H  = CANVAS_H - 2 * MARGIN_Y   # 580

def m_to_px(x_m, y_m):
    px = MARGIN_X + int(x_m / PITCH_WIDTH_M  * FIELD_W)
    py = MARGIN_Y + int(y_m / PITCH_HEIGHT_M * FIELD_H)
    return (px, py)

# Tidak perlu import sports sama sekali
# Ini adalah isi vertices yang sudah kita baca dari source code Roboflow

_VERTICES_CM = [
    (0, 0), (0, 1450), (0, 2584), (0, 4416), (0, 5550), (0, 7000),
    (550, 2584), (550, 4416),
    (1100, 3500),
    (2015, 1450), (2015, 2584), (2015, 4416), (2015, 5550),
    (6000, 0), (6000, 2585), (6000, 4415), (6000, 7000),
    (9985, 1450), (9985, 2584), (9985, 4416), (9985, 5550),
    (10900, 3500),
    (11450, 2584), (11450, 4416),
    (12000, 0), (12000, 1450), (12000, 2584), (12000, 4416), (12000, 5550), (12000, 7000),
    (5085, 3500), (6915, 3500),
]

PITCH_KEYPOINTS = {}
for idx, (x_cm, y_cm) in enumerate(_VERTICES_CM):
    x_m = (x_cm / 12000.0) * 105.0
    y_m = (y_cm / 7000.0)  * 68.0
    PITCH_KEYPOINTS[idx] = m_to_px(x_m, y_m)

# vertices dalam cm, konversi ke meter lalu ke pixel
# Roboflow: x = length (0..12000cm), y = width (0..7000cm)
# Canvas kita: x_m = 0..105 (horizontal), y_m = 0..68 (vertikal)
# PITCH_KEYPOINTS = {}
# for idx, (x_cm, y_cm) in enumerate(config.vertices):
#     x_m = x_cm / 100.0   # cm → meter (0..120 → tapi pakai 105 karena config pakai 12000cm)
#     y_m = y_cm / 100.0   # cm → meter (0..70 → pakai 68)
#     # Roboflow config: length=12000cm=120m, width=7000cm=70m
#     # Lapangan standar: 105m x 68m → perlu rescale
#     x_scaled = x_m * (105.0 / 120.0)
#     y_scaled = y_m * (68.0  / 70.0)
#     PITCH_KEYPOINTS[idx] = m_to_px(x_scaled, y_scaled)

# # Verifikasi
# if __name__ == "__main__":
#     for k in sorted(PITCH_KEYPOINTS.keys()):
#         print(f"  ID {k:2d}: {PITCH_KEYPOINTS[k]}")

# # Roboflow vertices (0-indexed), sudah dikonversi ke canvas 1050x680 dengan margin 50
# PITCH_KEYPOINTS = {
#     0:  m_to_px(  0.00,  0.00),  # pojok kiri atas
#     1:  m_to_px(  0.00, 13.84),  # kiri, awal kotak penalti
#     2:  m_to_px(  0.00, 24.84),  # kiri, awal kotak gawang  — FIX: ini bukan (50,168)
#     3:  m_to_px(  0.00, 43.16),  # kiri, akhir kotak gawang
#     4:  m_to_px(  0.00, 54.16),  # kiri, akhir kotak penalti
#     5:  m_to_px(  0.00, 68.00),  # pojok kiri bawah
#     6:  m_to_px(  5.50, 24.84),  # kotak gawang kiri atas
#     7:  m_to_px(  5.50, 43.16),  # kotak gawang kiri bawah
#     8:  m_to_px( 11.00, 34.00),  # titik penalti kiri
#     9:  m_to_px( 16.50, 13.84),  # kotak penalti kiri atas
#     10: m_to_px( 16.50, 24.84),  # kotak penalti kiri-dalam atas
#     11: m_to_px( 16.50, 43.16),  # kotak penalti kiri-dalam bawah
#     12: m_to_px( 16.50, 54.16),  # kotak penalti kiri bawah
#     13: m_to_px( 52.50,  0.00),  # tengah atas
#     14: m_to_px( 52.50, 25.85),  # lingkaran tengah atas
#     15: m_to_px( 52.50, 42.15),  # lingkaran tengah bawah
#     16: m_to_px( 52.50, 68.00),  # tengah bawah
#     17: m_to_px( 88.50, 13.84),  # kotak penalti kanan atas
#     18: m_to_px( 88.50, 24.84),  # kotak penalti kanan-dalam atas
#     19: m_to_px( 88.50, 43.16),  # kotak penalti kanan-dalam bawah
#     20: m_to_px( 88.50, 54.16),  # kotak penalti kanan bawah
#     21: m_to_px( 94.00, 34.00),  # titik penalti kanan
#     22: m_to_px( 99.50, 24.84),  # kotak gawang kanan atas
#     23: m_to_px( 99.50, 43.16),  # kotak gawang kanan bawah
#     24: m_to_px(105.00,  0.00),  # pojok kanan atas
#     25: m_to_px(105.00, 13.84),  # kanan, awal kotak penalti
#     26: m_to_px(105.00, 24.84),  # kanan, awal kotak gawang
#     27: m_to_px(105.00, 43.16),  # kanan, akhir kotak gawang
#     28: m_to_px(105.00, 54.16),  # kanan, akhir kotak penalti
#     29: m_to_px(105.00, 68.00),  # pojok kanan bawah
#     30: m_to_px( 43.35, 34.00),  # lingkaran tengah kiri
#     31: m_to_px( 61.65, 34.00),  # lingkaran tengah kanan
# }
