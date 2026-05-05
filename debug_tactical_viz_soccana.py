"""
debug_tactical_viz_soccana.py
─────────────────────────────
Script debug pipeline tactical map menggunakan model Soccana (32 KP).

Koordinat referensi pitch menggunakan SoccerPitchConfiguration dari
roboflow/sports (unit: cm). Homography dihitung dengan ViewTransformer.

Output per frame:
  - Kiri  : frame video asli + keypoints overlay (warna per conf)
  - Kanan : tactical map (canvas lapangan) + posisi KP yang terdeteksi

Metrics terminal:
  - Jumlah KP terdeteksi per frame
  - Reprojection error dalam pixel canvas
  - Status H matrix (valid / degenerate / fallback / None)

Usage:
  python debug_tactical_viz_soccana.py \
      --video input_video/input_vid_3.mp4 \
      --kp_model models/soccana_keypoint/Model/weights/kpdet_best_final.pt \
      --conf 0.5 \
      --min_kp 4 \
      --frames 0,100,200,300,400,500,600,700 \
      --output debug_soccana_output.mp4 \
      --verbose
"""

import argparse
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer


# ══════════════════════════════════════════════════════════════════════════════
# PITCH CONFIG
# Unit koordinat: cm (sesuai SoccerPitchConfiguration roboflow/sports)
# ══════════════════════════════════════════════════════════════════════════════

PITCH_LENGTH_CM = 12000   # sumbu X (panjang lapangan)
PITCH_WIDTH_CM  = 7000    # sumbu Y (lebar lapangan)

CANVAS_W = 1050
CANVAS_H = 680
MARGIN_X = 50
MARGIN_Y = 50
FIELD_W  = CANVAS_W - 2 * MARGIN_X   # 950 px
FIELD_H  = CANVAS_H - 2 * MARGIN_Y   # 580 px

# Ambil 32 vertices dari SoccerPitchConfiguration — sumber kebenaran tunggal
CONFIG   = SoccerPitchConfiguration()
VERTICES = np.array(CONFIG.vertices, dtype=np.float32)  # shape (32, 2), unit cm

def cm_to_px(x_cm, y_cm):
    """Konversi koordinat cm lapangan → pixel canvas."""
    px = MARGIN_X + int(x_cm / PITCH_LENGTH_CM * FIELD_W)
    py = MARGIN_Y + int(y_cm / PITCH_WIDTH_CM  * FIELD_H)
    return (px, py)

# Lookup pixel canvas per KP index — untuk visualisasi dan reproj error display
PITCH_KEYPOINTS_PX = {
    i: cm_to_px(float(v[0]), float(v[1]))
    for i, v in enumerate(CONFIG.vertices)
}

KP_NAMES = {
    0:  "corner_TL",
    1:  "penbox_L_top_side",
    2:  "goalbox_L_top_side",
    3:  "goalbox_L_bot_side",
    4:  "penbox_L_bot_side",
    5:  "corner_BL",
    6:  "goalbox_L_top_inner",
    7:  "goalbox_L_bot_inner",
    8:  "penalty_spot_L",
    9:  "penbox_L_top_inner",
    10: "penarc_L_top_inner",
    11: "penarc_L_bot_inner",
    12: "penbox_L_bot_inner",
    13: "center_top",
    14: "cc_top",
    15: "cc_bot",
    16: "center_bot",
    17: "penbox_R_top_inner",
    18: "penarc_R_top_inner",
    19: "penarc_R_bot_inner",
    20: "penbox_R_bot_inner",
    21: "penalty_spot_R",
    22: "goalbox_R_top_inner",
    23: "goalbox_R_bot_inner",
    24: "corner_TR",
    25: "penbox_R_top_side",
    26: "goalbox_R_top_side",
    27: "goalbox_R_bot_side",
    28: "penbox_R_bot_side",
    29: "corner_BR",
    30: "cc_left",
    31: "cc_right",
}

# Kelompok KP untuk filter logika
CENTER_KPS   = {13, 14, 15, 16, 30, 31}
LEFT_SIDE_KPS  = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
RIGHT_SIDE_KPS = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
SIDE_KPS     = LEFT_SIDE_KPS | RIGHT_SIDE_KPS


# ══════════════════════════════════════════════════════════════════════════════
# KEYPOINT DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class KeypointDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        print(f"[KeypointDetector] Loading: {model_path}")
        self.model      = YOLO(model_path)
        self.conf_thresh = confidence_threshold
        self.NUM_KP     = self.model.model.kpt_shape[0]
        print(f"[KeypointDetector] Ready — {self.NUM_KP} KP slots.")

    def detect(self, frame: np.ndarray):
        """
        Deteksi keypoints di frame.

        Returns:
            keypoints : dict {idx: (x_px, y_px)}  — hanya KP di atas conf threshold
            raw_conf  : dict {idx: conf}            — semua KP untuk visualisasi warna
        """
        orig_h, orig_w = frame.shape[:2]
        results = self.model(frame, verbose=False)[0]

        keypoints: dict = {}
        raw_conf:  dict = {}

        if results.keypoints is None or len(results.keypoints.data) == 0:
            return keypoints, raw_conf

        kps = results.keypoints.data[0]  # (32, 3): x, y, conf
        for idx in range(len(kps)):
            x, y, conf = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
            raw_conf[idx] = conf
            if conf < self.conf_thresh:
                continue
            xi, yi = int(x), int(y)
            if 0 <= xi < orig_w and 0 <= yi < orig_h:
                keypoints[idx] = (xi, yi)

        return keypoints, raw_conf


# ══════════════════════════════════════════════════════════════════════════════
# KEYPOINT FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def filter_conflicting_kps(keypoints: dict) -> dict:
    """
    Kalau center KP terdeteksi banyak (>=5) tapi side KP sedikit (<=2),
    buang side KP yang kemungkinan besar noise.
    """
    cc_detected   = [k for k in keypoints if k in CENTER_KPS]
    side_detected = [k for k in keypoints if k in SIDE_KPS]

    if len(cc_detected) >= 5 and len(side_detected) <= 2:
        filtered = {k: v for k, v in keypoints.items() if k not in SIDE_KPS}
        print(f"       [ConflictFilter] Removed side KPs {side_detected} "
              f"(cc={cc_detected})")
        return filtered

    return keypoints


def filter_false_center_kps(keypoints: dict) -> dict:
    """
    Buang KP center yang posisi X-nya tidak masuk akal.

    Kamera di sisi kanan: KP center tidak boleh lebih ke kanan dari
    rata-rata KP sisi kanan yang terdeteksi (dan sebaliknya untuk kiri).

    Ini menangani kasus model salah mendeteksi garis penalty box kanan
    sebagai center line ketika kamera sudah jauh ke sisi kanan.
    """
    filtered = dict(keypoints)

    right_kps  = {k for k in RIGHT_SIDE_KPS if k in keypoints}
    left_kps   = {k for k in LEFT_SIDE_KPS  if k in keypoints}
    center_kps = {k for k in CENTER_KPS      if k in keypoints}

    if not center_kps:
        return filtered

    center_x = np.mean([keypoints[k][0] for k in center_kps])

    # Kamera di sisi kanan: ada >= 2 KP kanan tapi tidak ada KP kiri
    if len(right_kps) >= 2 and len(left_kps) == 0:
        right_min_x = min(keypoints[k][0] for k in right_kps)
        if center_x > right_min_x * 0.85:
            for k in list(center_kps):
                filtered.pop(k, None)
            print(f"       [FalseCenterFilter] Removed center KPs {sorted(center_kps)} "
                  f"(center_x={center_x:.0f} >= right_min_x={right_min_x:.0f})")

    # Kamera di sisi kiri: ada >= 2 KP kiri tapi tidak ada KP kanan
    elif len(left_kps) >= 2 and len(right_kps) == 0:
        left_max_x = max(keypoints[k][0] for k in left_kps)
        if center_x < left_max_x * 1.15:
            for k in list(center_kps):
                filtered.pop(k, None)
            print(f"       [FalseCenterFilter] Removed center KPs {sorted(center_kps)} "
                  f"(center_x={center_x:.0f} <= left_max_x={left_max_x:.0f})")

    return filtered


def filter_outlier_cc_kps(keypoints, raw_conf):
    """
    KP30/31 (cc_left/cc_right) harus simetris terhadap KP15 (field_center).
    Kalau salah satu jauh lebih kiri/kanan dari yang diharapkan, buang.
    """
    kp15 = keypoints.get(15)  # field center
    kp30 = keypoints.get(30)  # cc_left  — harusnya di KIRI center
    kp31 = keypoints.get(31)  # cc_right — harusnya di KANAN center

    filtered = dict(keypoints)

    if kp15 and kp30 and kp31:
        # cc_left harus lebih kiri dari center, cc_right harus lebih kanan
        if kp30[0] > kp15[0]:  # cc_left ada di kanan center → salah
            filtered.pop(30, None)
            print(f"       [CCFilter] Removed KP30(cc_left) x={kp30[0]} > center x={kp15[0]}")
        if kp31[0] < kp15[0]:  # cc_right ada di kiri center → salah
            filtered.pop(31, None)
            print(f"       [CCFilter] Removed KP31(cc_right) x={kp31[0]} < center x={kp15[0]}")

    return filtered


def is_camera_lateral(keypoints: dict, frame_width: int):
    """
    Deteksi kamera terlalu jauh ke sisi lapangan berdasarkan
    posisi cc_left (KP30) atau cc_right (KP31) di tepi frame.
    """
    kp30 = keypoints.get(30)
    kp31 = keypoints.get(31)
    if kp30 and kp30[0] < 60:
        return True, f"KP30(cc_left) x={kp30[0]} terlalu kiri"
    if kp31 and kp31[0] > frame_width - 60:
        return True, f"KP31(cc_right) x={kp31[0]} terlalu kanan"
    return False, None


# ══════════════════════════════════════════════════════════════════════════════
# HOMOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════

class HomographyTracker:
    """Buffer H matrix dari frame-frame sebelumnya sebagai fallback."""

    # Threshold dalam cm: ~500cm ≈ 5px di canvas — H dianggap bagus
    REPROJ_GOOD_CM = 500

    def __init__(self, max_age: int = 50):
        self.buffer:  list = []   # list of (H, reproj_cm, frame_num)
        self.max_age: int  = max_age

    def update(self, H, reproj_cm: float, frame_num: int):
        if H is not None and not np.isnan(reproj_cm) and reproj_cm < self.REPROJ_GOOD_CM:
            self.buffer.append((H, reproj_cm, frame_num))

    def get_fallback(self, current_frame: int):
        """Ambil H terbaik (reproj terkecil) yang tidak lebih tua dari max_age."""
        valid = [
            (H, r, fn) for H, r, fn in self.buffer
            if current_frame - fn <= self.max_age
        ]
        if not valid:
            return None, None
        best = min(valid, key=lambda x: x[1])
        return best[0], best[2]   # H, frame_num asal


def compute_homography(keypoints_frame: dict, min_kp: int = 4):
    """
    Hitung H matrix menggunakan ViewTransformer (roboflow/sports).

    Source : pixel frame  (dari detector)
    Target : cm lapangan  (dari VERTICES / SoccerPitchConfiguration)

    Returns:
        H        : np.ndarray (3,3) atau None
        n_kp     : jumlah KP yang dipakai
        status   : string deskriptif
    """
    valid_idx = sorted(k for k in keypoints_frame if k < len(VERTICES))
    if len(valid_idx) < min_kp:
        return None, len(valid_idx), "too_few_kp"

    frame_pts = np.array([keypoints_frame[i] for i in valid_idx], dtype=np.float32)
    pitch_pts = VERTICES[valid_idx]

    # Spread check — cegah degenerate homography
    if (frame_pts[:, 0].max() - frame_pts[:, 0].min() < 50 or
            frame_pts[:, 1].max() - frame_pts[:, 1].min() < 50):
        return None, len(valid_idx), "low_spread"

    try:
        vt = ViewTransformer(source=frame_pts, target=pitch_pts)
        return vt.m, len(valid_idx), f"OK(n={len(valid_idx)})"
    except Exception as e:
        return None, len(valid_idx), f"VT_failed({e})"


def reproj_error_cm(keypoints_frame: dict, H) -> float:
    """
    Reprojection error dalam cm — dipakai untuk HomographyTracker.
    (frame px → cm, bandingkan dengan VERTICES)
    """
    if H is None:
        return float('nan')
    valid_idx = sorted(k for k in keypoints_frame if k < len(VERTICES))
    if not valid_idx:
        return float('nan')

    src  = np.array([[keypoints_frame[k]] for k in valid_idx], dtype=np.float32)
    dst  = VERTICES[valid_idx]
    proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - dst, axis=1)))


def reproj_error_px(keypoints_frame: dict, H) -> float:
    """
    Reprojection error dalam pixel canvas — dipakai untuk display dan flag.
    (frame px → cm → canvas px, bandingkan dengan PITCH_KEYPOINTS_PX)
    """
    if H is None:
        return float('nan')
    valid_idx = [k for k in keypoints_frame if k in PITCH_KEYPOINTS_PX]
    if not valid_idx:
        return float('nan')

    src      = np.array([[keypoints_frame[k]] for k in valid_idx], dtype=np.float32)
    proj_cm  = cv2.perspectiveTransform(src, H).reshape(-1, 2)

    errors = []
    for i, kid in enumerate(valid_idx):
        ref_px  = PITCH_KEYPOINTS_PX[kid]
        proj_px = cm_to_canvas(proj_cm[i][0], proj_cm[i][1])
        if proj_px is not None:
            errors.append(np.sqrt(
                (proj_px[0] - ref_px[0])**2 + (proj_px[1] - ref_px[1])**2
            ))

    return float(np.mean(errors)) if errors else float('nan')


def transform_point_cm(pt, H):
    """Transform satu titik dari pixel frame → koordinat cm lapangan."""
    if H is None or pt is None:
        return None
    src = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    return (float(dst[0][0][0]), float(dst[0][0][1]))


def cm_to_canvas(x_cm: float, y_cm: float, margin: int = 10):
    """Konversi koordinat cm → pixel canvas. Return None jika di luar canvas."""
    px = MARGIN_X + int(x_cm / PITCH_LENGTH_CM * FIELD_W)
    py = MARGIN_Y + int(y_cm / PITCH_WIDTH_CM  * FIELD_H)
    if -margin <= px <= CANVAS_W + margin and -margin <= py <= CANVAS_H + margin:
        return (px, py)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# TACTICAL MAP CANVAS
# ══════════════════════════════════════════════════════════════════════════════

PITCH_GREEN = (34, 139, 34)
LINE_WHITE  = (255, 255, 255)


def draw_pitch_canvas() -> np.ndarray:
    """Gambar lapangan kosong di canvas."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), PITCH_GREEN, dtype=np.uint8)
    lw = 2
    mx, my = MARGIN_X, MARGIN_Y
    fw, fh = FIELD_W, FIELD_H

    # Outline lapangan
    cv2.rectangle(canvas, (mx, my), (mx+fw, my+fh), LINE_WHITE, lw)

    # Garis tengah
    cv2.line(canvas, (mx+fw//2, my), (mx+fw//2, my+fh), LINE_WHITE, lw)

    # Lingkaran tengah — radius 915cm
    cx, cy = mx + fw//2, my + fh//2
    r = int(915 / PITCH_LENGTH_CM * FIELD_W)
    cv2.circle(canvas, (cx, cy), r, LINE_WHITE, lw)
    cv2.circle(canvas, (cx, cy), 4, LINE_WHITE, -1)

    # Kotak penalti kiri — panjang 2015cm, lebar 4100cm
    pb_w = int(2015 / PITCH_LENGTH_CM * FIELD_W)
    pb_t = int(1450 / PITCH_WIDTH_CM  * FIELD_H)
    pb_b = int(5550 / PITCH_WIDTH_CM  * FIELD_H)
    cv2.rectangle(canvas, (mx, my+pb_t), (mx+pb_w, my+pb_b), LINE_WHITE, lw)
    cv2.rectangle(canvas, (mx+fw-pb_w, my+pb_t), (mx+fw, my+pb_b), LINE_WHITE, lw)

    # Kotak gawang kiri — panjang 550cm, lebar 1832cm
    gb_w = int(550  / PITCH_LENGTH_CM * FIELD_W)
    gb_t = int(2584 / PITCH_WIDTH_CM  * FIELD_H)
    gb_b = int(4416 / PITCH_WIDTH_CM  * FIELD_H)
    cv2.rectangle(canvas, (mx, my+gb_t), (mx+gb_w, my+gb_b), LINE_WHITE, lw)
    cv2.rectangle(canvas, (mx+fw-gb_w, my+gb_t), (mx+fw, my+gb_b), LINE_WHITE, lw)

    # Semicircle penalty kiri & kanan
    r_arc = int(915 / PITCH_LENGTH_CM * FIELD_W)
    for cx_arc, flip in [(mx+pb_w, 1), (mx+fw-pb_w, -1)]:
        cv2.ellipse(canvas, (cx_arc, my+fh//2), (r_arc, r_arc),
                    0, -60*flip, 60*flip, LINE_WHITE, lw)

    return canvas


def draw_tactical_frame(keypoints_map: dict, frame_num: int, metrics: dict,
                         player_positions: list = None, ball_pos=None) -> np.ndarray:
    """
    Gambar tactical map dengan KP overlay dan metrics.

    Args:
        keypoints_map    : dict {idx: (canvas_x, canvas_y)}
        frame_num        : nomor frame saat ini
        metrics          : dict berisi h_status, n_kp, reproj, n_player_mapped
        player_positions : list of dict {map_pos, team, role} (opsional)
        ball_pos         : (x, y) pixel canvas atau None
    """
    canvas = draw_pitch_canvas()

    # KP overlay (kuning)
    for kid, (kx, ky) in keypoints_map.items():
        cv2.circle(canvas, (kx, ky), 5, (255, 255, 0), -1)
        cv2.putText(canvas, str(kid), (kx+4, ky-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    # Player overlay
    if player_positions:
        team_colors = {1: (255, 50, 50), 2: (50, 50, 255), -1: (180, 180, 180)}
        for pinfo in player_positions:
            mp = pinfo.get('map_pos')
            if mp is None:
                continue
            col = team_colors.get(pinfo.get('team', -1), (180, 180, 180))
            r   = 10 if pinfo.get('role') == 'goalkeeper' else 8
            cv2.circle(canvas, mp, r, col, -1)
            cv2.circle(canvas, mp, r, LINE_WHITE, 1)

    # Bola
    if ball_pos:
        cv2.circle(canvas, ball_pos, 7, LINE_WHITE, -1)
        cv2.circle(canvas, ball_pos, 7, (0, 0, 0), 2)

    # Metrics overlay
    reproj = metrics.get('reproj', float('nan'))
    info_lines = [
        f"Frame: {frame_num}",
        f"KP detected: {metrics.get('n_kp', 0)}",
        f"H status: {metrics.get('h_status', 'N/A')}",
        f"Reproj err: {reproj:.1f}px" if not np.isnan(reproj) else "Reproj err: N/A",
        f"Players mapped: {metrics.get('n_player_mapped', 0)}",
    ]
    for i, line in enumerate(info_lines):
        if "Reproj" in line and not np.isnan(reproj):
            color = (0, 255, 0) if reproj < 15 else (0, 165, 255) if reproj < 50 else (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(canvas, line, (10, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO FRAME OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def draw_kp_overlay(frame: np.ndarray, keypoints: dict,
                    raw_conf: dict, conf_thresh: float) -> np.ndarray:
    """Overlay KP pada frame asli. Warna: merah (conf rendah) → hijau (conf tinggi)."""
    out = frame.copy()
    for idx in range(32):
        conf = raw_conf.get(idx, 0.0)
        if conf < 0.1:
            continue
        color = (0, int(255 * conf), int(255 * (1 - conf)))
        if idx in keypoints:
            x, y = keypoints[idx]
            cv2.circle(out, (x, y), 7, color, -1)
            cv2.putText(out, f"{idx}({conf:.2f})", (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    cv2.putText(out, f"Green=high conf, Red=low conf, threshold={conf_thresh}",
                (10, out.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def verify_kp_mapping():
    """Print tabel mapping KP index → nama → koordinat pixel canvas."""
    print("\n=== KP MAPPING VERIFICATION (32 KP) ===")
    print(f"{'ID':>4}  {'Name':25}  {'cm (x,y)':>20}  {'Canvas px':>15}")
    print("-" * 72)
    for kid, v in enumerate(CONFIG.vertices):
        name   = KP_NAMES.get(kid, f"kp{kid}")
        ref_px = PITCH_KEYPOINTS_PX[kid]
        print(f"{kid:>4}  {name:25}  ({v[0]:>6.0f},{v[1]:>6.0f})  {str(ref_px):>15}")
    print("=== END MAPPING ===\n")


def reproj_flag(reproj_px: float) -> str:
    if np.isnan(reproj_px):
        return "✗"
    if reproj_px < 15:
        return "✓"
    if reproj_px < 50:
        return "~"
    return "✗"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def process_video(args):
    detector = KeypointDetector(args.kp_model, confidence_threshold=args.conf)
    tracker  = HomographyTracker(max_age=50)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Video] {total_frames} frames @ {fps:.1f} fps | {orig_w}x{orig_h}")

    # Pilih frame yang diproses
    if args.frames:
        frame_indices = sorted(set(int(f) for f in args.frames.split(',')))
        frame_indices = [f for f in frame_indices if 0 <= f < total_frames]
        mode = 'selected'
        print(f"[Debug] Processing {len(frame_indices)} frames: {frame_indices}")
    else:
        frame_indices = list(range(total_frames))
        mode = 'all'
        print(f"[Debug] Processing ALL {total_frames} frames")

    # Output video writer
    tac_display_h = orig_h
    tac_display_w = int(tac_display_h * CANVAS_W / CANVAS_H)
    out_w = orig_w + 4 + tac_display_w
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps if mode == 'all' else 5,
        (out_w, orig_h)
    )

    # Metrics accumulator
    summary = {
        'frames_processed': 0,
        'frames_h_valid':   0,
        'reproj_errors_px': [],
        'kp_counts':        [],
    }

    verify_kp_mapping()

    frame_num = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num not in frame_indices:
            frame_num += 1
            continue

        if frame_num % 50 == 0:
            print(f"  [Progress] Frame {frame_num}/{total_frames}...")

        # ── Deteksi keypoints ────────────────────────────────────────────
        keypoints, raw_conf = detector.detect(frame)

        # Cek kamera lateral (info saja, tidak mempengaruhi pipeline)
        lateral, lateral_reason = is_camera_lateral(keypoints, orig_w)
        if lateral:
            print(f"       [WARN] Camera lateral: {lateral_reason}")

        # Deteksi ulang (reset duplikasi yang ada di code lama)
        keypoints, raw_conf = detector.detect(frame)

        # ── Filter keypoints ─────────────────────────────────────────────
        keypoints = filter_conflicting_kps(keypoints)
        keypoints = filter_false_center_kps(keypoints)
        keypoints = filter_outlier_cc_kps(keypoints, raw_conf)
        n_kp = len(keypoints)

        # ── Hitung homography ────────────────────────────────────────────
        H, n_common, h_note = compute_homography(keypoints, min_kp=args.min_kp)
        r_cm = reproj_error_cm(keypoints, H)
        r_px = reproj_error_px(keypoints, H)

        # Simpan ke tracker kalau H bagus
        tracker.update(H, r_cm, frame_num)

        # Fallback ke H dari frame sebelumnya kalau perlu
        h_source = "current"
        if H is None or np.isnan(r_px) or r_px > 50:
            H_fb, from_frame = tracker.get_fallback(frame_num)
            if H_fb is not None:
                H       = H_fb
                r_cm    = reproj_error_cm(keypoints, H)
                r_px    = reproj_error_px(keypoints, H)
                h_note  = f"FALLBACK(from={from_frame})"
                h_source = f"fallback(frame={from_frame})"

        h_status = h_note if h_note else f"no_H(n={n_common})"

        # ── Transform KP ke canvas ───────────────────────────────────────
        kp_on_map = {}
        for kid, kp_src in keypoints.items():
            pt_cm = transform_point_cm(kp_src, H)
            if pt_cm is not None:
                canvas_pt = cm_to_canvas(pt_cm[0], pt_cm[1])
                if canvas_pt is not None:
                    kp_on_map[kid] = canvas_pt

        # ── Print terminal ───────────────────────────────────────────────
        reproj_str = f"{r_px:.1f}px" if not np.isnan(r_px) else "N/A"
        flag = reproj_flag(r_px)
        print(f"  [{flag}] Frame {frame_num:4d} | KP={n_kp:2d} | "
              f"H={h_note} | Reproj={reproj_str} | src={h_source}")

        # Verbose: detail per KP
        if args.verbose or mode == 'selected':
            for kid, (kx, ky) in keypoints.items():
                name    = KP_NAMES.get(kid, f"kp{kid}")
                conf    = raw_conf.get(kid, 0.0)
                ref_px  = PITCH_KEYPOINTS_PX.get(kid)
                if H is not None and ref_px is not None:
                    pt_cm = transform_point_cm((kx, ky), H)
                    if pt_cm is not None:
                        proj_px = cm_to_canvas(pt_cm[0], pt_cm[1])
                        if proj_px is not None:
                            err = np.sqrt((proj_px[0]-ref_px[0])**2 +
                                          (proj_px[1]-ref_px[1])**2)
                            warn = " ← HIGH ERROR" if err > 30 else ""
                            print(f"       KP{kid:2d} {name:25s} "
                                  f"frame=({kx},{ky}) "
                                  f"map=({proj_px[0]},{proj_px[1]}) "
                                  f"ref={ref_px} err={err:.1f}px "
                                  f"conf={conf:.2f}{warn}")
                        else:
                            print(f"       KP{kid:2d} {name:25s} "
                                  f"frame=({kx},{ky}) → OUT_OF_CANVAS conf={conf:.2f}")
                    else:
                        print(f"       KP{kid:2d} {name:25s} "
                              f"frame=({kx},{ky}) → transform=None conf={conf:.2f}")
                else:
                    print(f"       KP{kid:2d} {name:25s} "
                          f"frame=({kx},{ky}) conf={conf:.2f} [no H]")

        # ── Render output frame ──────────────────────────────────────────
        metrics = {
            'h_status':        h_status,
            'n_kp':            n_kp,
            'reproj':          r_px,
            'n_player_mapped': 0,
        }

        frame_overlay = draw_kp_overlay(frame, keypoints, raw_conf, args.conf)
        cv2.putText(frame_overlay,
                    f"Frame {frame_num} | KP={n_kp} | {h_note} | Reproj={reproj_str}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        tac_frame   = draw_tactical_frame(kp_on_map, frame_num, metrics)
        tac_resized = cv2.resize(tac_frame, (tac_display_w, tac_display_h))
        divider     = np.full((orig_h, 4, 3), 180, dtype=np.uint8)
        combined    = np.hstack([frame_overlay, divider, tac_resized])

        writer.write(combined)

        if mode == 'selected':
            still_path = args.output.replace('.mp4', f'_frame{frame_num:04d}.jpg')
            cv2.imwrite(still_path, combined)
            print(f"  [Saved] {still_path}")

        # Update summary
        summary['frames_processed'] += 1
        summary['kp_counts'].append(n_kp)
        if H is not None:
            summary['frames_h_valid'] += 1
        if not np.isnan(r_px):
            summary['reproj_errors_px'].append(r_px)

        frame_num += 1

    cap.release()
    writer.release()

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Frames processed  : {summary['frames_processed']}")
    print(f"Frames H valid    : {summary['frames_h_valid']} / {summary['frames_processed']}")
    if summary['kp_counts']:
        kc = summary['kp_counts']
        print(f"KP per frame      : mean={np.mean(kc):.1f}  "
              f"min={min(kc)}  max={max(kc)}")
    if summary['reproj_errors_px']:
        errs = summary['reproj_errors_px']
        print(f"Reproj error (px) : mean={np.mean(errs):.1f}  "
              f"median={np.median(errs):.1f}  max={max(errs):.1f}")
        print(f"  < 15px (excellent) : {sum(1 for e in errs if e < 15)} / {len(errs)}")
        print(f"  < 30px (good)      : {sum(1 for e in errs if e < 30)} / {len(errs)}")
        print(f"  < 50px (acceptable): {sum(1 for e in errs if e < 50)} / {len(errs)}")
        print(f"  >= 50px (bad)      : {sum(1 for e in errs if e >= 50)} / {len(errs)}")
    print(f"\nOutput saved: {args.output}")
    print("="*60)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Debug tactical viz — Soccana 32-KP model"
    )
    p.add_argument('--video',    required=True,
                   help='Path ke video input')
    p.add_argument('--kp_model', required=True,
                   help='Path ke model YOLO keypoint (.pt)')
    p.add_argument('--conf',     type=float, default=0.5,
                   help='Confidence threshold keypoint (default: 0.5)')
    p.add_argument('--min_kp',   type=int, default=4,
                   help='Min KP untuk homography (default: 4)')
    p.add_argument('--frames',   default=None,
                   help='Frame indices dipisah koma, e.g. "0,100,200". '
                        'Kosongkan untuk proses semua frame.')
    p.add_argument('--output',   default='debug_soccana_output.mp4',
                   help='Path output video (default: debug_soccana_output.mp4)')
    p.add_argument('--verbose',  action='store_true',
                   help='Print detail per-KP di setiap frame')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_video(args)