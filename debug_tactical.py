"""
debug_tactical_viz.py  v3
─────────────────────────
JALUR 2: Line-based homography constraint.

Ide utama:
  Halfway line di video selalu terdeteksi Hough sebagai garis vertikal.
  Kita tahu garis itu = x=525 di canvas (halfway line pitch).
  → Sampling beberapa titik di sepanjang garis tersebut sebagai pseudo-keypoints.
  → Pseudo-keypoints ini digabung dengan keypoint model untuk findHomography.

Kenapa ini lebih baik dari sebelumnya:
  - Model keypoint hanya reliable di area halfway/center → semua di x≈525
  - Dengan line constraint, kita dapat titik-titik yang tersebar di SUMBU Y
  - Spread_y menjadi lebih baik → H matrix lebih stabil
  - Tidak butuh intersection (yang susah dapat)

CARA PAKAI:
  python debug_tactical_viz_v3.py

Output: debug_tactical_frames_v3/ folder berisi JPG per frame sample.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

# ── PITCH CONFIG ───────────────────────────────────────────────────────────────
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

# Halfway line di canvas = x=525 (tengah dari CANVAS_W=1050)
HALFWAY_X_CANVAS = CANVAS_W // 2  # = 525

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VIDEO_PATH    = "input_video/input_1.mp4"
KP_MODEL_PATH = "models/best_roboflow_keypoints_detection.pt"
CONF_THRESH   = 0.3
TEST_FRAMES   = [0, 100, 200, 300, 400, 500, 600, 700]
OUTPUT_DIR    = "debug_tactical_frames_v3_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Jumlah pseudo-keypoint yang di-sample sepanjang halfway line
N_PSEUDO_KP = 5  # akan di-sample di y = 20%, 35%, 50%, 65%, 80% dari frame height

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
print("Loading keypoint model...")
kp_model = YOLO(KP_MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# HOUGH LINE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_field_lines(frame):
    """
    Deteksi garis putih lapangan pakai Hough.
    Return: list of (x1,y1,x2,y2), mask_lines, edges
    """
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (35, 30, 40), (90, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask_green_exp = cv2.dilate(mask_green, kernel, iterations=2)

    mask_white = cv2.inRange(frame, (160, 160, 160), (255, 255, 255))
    mask_lines = cv2.bitwise_and(mask_white, mask_green_exp)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_lines = cv2.morphologyEx(mask_lines, cv2.MORPH_CLOSE, kernel_close)

    edges = cv2.Canny(mask_lines, 30, 100)
    lines_raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=40, minLineLength=60, maxLineGap=25
    )

    result = []
    if lines_raw is not None:
        for line in lines_raw:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length < 60:
                continue
            if max(y1, y2) < h * 0.25:
                continue
            result.append((x1, y1, x2, y2))

    return result, mask_lines, edges


def classify_line(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    angle = np.degrees(np.arctan2(dy, dx + 1e-8))
    if angle < 20:
        return 'H'
    elif angle > 70:
        return 'V'
    else:
        return 'D'


# ══════════════════════════════════════════════════════════════════════════════
# HALFWAY LINE DETECTION & PSEUDO-KEYPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def fit_halfway_line(lines, frame_w, frame_h):
    """
    Dari semua garis Hough yang terdeteksi, cari garis vertikal yang
    paling likely adalah halfway line.

    Strategi:
    1. Filter hanya garis 'V' (vertikal, angle > 70°)
    2. Cluster garis-garis V berdasarkan posisi x rata-rata
    3. Pilih cluster yang posisi x-nya paling dekat ke tengah frame (x ≈ frame_w/2)
    4. Fit garis tunggal dari semua segmen di cluster tersebut
    
    Return:
        line_params: (x_at_y, slope) → x = x_at_y + slope * (y - frame_h/2)
                     atau None kalau tidak ditemukan
        debug_info: dict untuk visualisasi
    """
    v_lines = [(x1, y1, x2, y2) for (x1, y1, x2, y2) in lines 
               if classify_line(x1, y1, x2, y2) == 'V']
    
    if not v_lines:
        return None, {"v_lines": [], "selected": [], "reason": "no_V_lines"}

    # Hitung x tengah tiap garis
    line_cx = [(x1 + x2) / 2 for (x1, y1, x2, y2) in v_lines]
    
    # Cluster berdasarkan x (eps=80px)
    eps = 80
    used = [False] * len(v_lines)
    clusters = []
    for i in range(len(v_lines)):
        if used[i]:
            continue
        group_idx = [i]
        used[i] = True
        for j in range(i + 1, len(v_lines)):
            if not used[j] and abs(line_cx[i] - line_cx[j]) < eps:
                group_idx.append(j)
                used[j] = True
        clusters.append(group_idx)
    
    # Pilih cluster yang x tengahnya paling dekat ke frame_w/2
    frame_cx = frame_w / 2
    best_cluster = min(clusters, key=lambda g: abs(np.mean([line_cx[i] for i in g]) - frame_cx))
    best_lines   = [v_lines[i] for i in best_cluster]

    # FIX B — Validasi: x cluster harus dalam range 35%–65% frame width
    # Kalau di luar range ini, bukan halfway line (mungkin tiang gawang / garis penalty)
    cluster_x_mean = np.mean([line_cx[i] for i in best_cluster])
    if not (frame_w * 0.35 <= cluster_x_mean <= frame_w * 0.65):
        return None, {
            "v_lines": v_lines, "selected": [], 
            "reason": f"rejected_not_center(x={cluster_x_mean:.0f}, range=[{frame_w*0.35:.0f},{frame_w*0.65:.0f}])"
        }
    
    # Fit garis tunggal dengan semua titik dari cluster terpilih
    pts = []
    for (x1, y1, x2, y2) in best_lines:
        pts.append([x1, y1])
        pts.append([x2, y2])
    pts = np.array(pts, dtype=np.float32)
    
    # Linear regression: x = a*y + b
    y_vals = pts[:, 1].reshape(-1, 1)
    x_vals = pts[:, 0]
    ones   = np.ones_like(y_vals)
    A      = np.hstack([y_vals, ones])
    result, _, _, _ = np.linalg.lstsq(A, x_vals, rcond=None)
    slope, intercept = result  # x = slope*y + intercept
    
    # x_at_mid = posisi x halfway line di tengah frame
    y_mid      = frame_h / 2
    x_at_mid   = slope * y_mid + intercept
    
    return (x_at_mid, slope, intercept), {
        "v_lines": v_lines,
        "selected": best_lines,
        "cluster_x_mean": cluster_x_mean,
        "reason": "ok"
    }


def generate_pseudo_keypoints(line_params, frame_h, frame_w, n=N_PSEUDO_KP):
    """
    Sampling N titik di sepanjang halfway line yang terdeteksi.
    
    Untuk setiap titik di frame (src), kita tahu posisi canvas-nya (dst):
        dst_x = HALFWAY_X_CANVAS (= 525, selalu)
        dst_y = di-interpolasi dari posisi y relatif di frame
    
    Mapping y frame → y canvas:
        Kita asumsikan FOV kamera mencakup SEBAGIAN lapangan secara vertikal.
        Pendekatan: gunakan keypoint model yang sudah ada untuk estimasi
        top/bottom boundary, atau gunakan asumsi sederhana bahwa:
        - y_frame_top_field ≈ frame_h * 0.15  (bawah tribun)
        - y_frame_bot_field ≈ frame_h * 0.95
        mapping linear ke canvas y [MARGIN_Y, CANVAS_H - MARGIN_Y]
    
    Return: list of (src_pt, dst_pt) → pseudo-keypoints
    """
    if line_params is None:
        return []
    
    x_at_mid, slope, intercept = line_params
    
    # Y sampling: hindari area tribun (atas) dan sangat bawah
    y_fracs = np.linspace(0.20, 0.90, n)
    
    pseudo_kps = []
    for y_frac in y_fracs:
        y_src = y_frac * frame_h
        x_src = slope * y_src + intercept
        
        # Skip kalau di luar frame
        if not (0 <= x_src <= frame_w):
            continue
        
        # Mapping ke canvas:
        # y_frac [0.20, 0.90] → canvas y [MARGIN_Y, CANVAS_H - MARGIN_Y]
        # Asumsi: frame_h*0.20 = top of pitch visible, frame_h*0.90 = bottom
        # (ini approximation — bisa di-refine dengan info dari keypoint model)
        y_canvas = MARGIN_Y + (y_frac - 0.20) / (0.90 - 0.20) * FIELD_H
        y_canvas = np.clip(y_canvas, MARGIN_Y, CANVAS_H - MARGIN_Y)
        
        pseudo_kps.append(([x_src, y_src], [HALFWAY_X_CANVAS, y_canvas]))
    
    return pseudo_kps


def refine_y_mapping(kps_xy, kps_conf, line_params, frame_h, frame_w, conf_thresh=CONF_THRESH):
    """
    Gunakan keypoint model yang reliable (ID 13, 14, 15, 16 di halfway line)
    untuk meng-kalibrasi mapping y frame → y canvas.
    
    Keypoint 13: (525, 50)   ← atas
    Keypoint 14: (525, 264)  ← tengah atas  
    Keypoint 15: (525, 415)  ← tengah bawah
    Keypoint 16: (525, 630)  ← bawah
    
    Ini memberi kita pasangan (y_frame, y_canvas) yang akurat untuk
    interpolasi/ekstrapolasi ke pseudo-keypoints di seluruh halfway line.
    
    Return: refined pseudo-keypoints list, calibration info
    """
    if line_params is None:
        return [], {}
    
    x_at_mid, slope, intercept = line_params
    
    # Halfway line keypoint IDs (x ≈ 525 di canvas)
    HALFWAY_KP_IDS = [13, 14, 15, 16]
    
    calib_pairs = []  # (y_frame, y_canvas)
    for kid in HALFWAY_KP_IDS:
        if kid >= len(kps_conf):
            continue
        c = float(kps_conf[kid])
        if c < conf_thresh:
            continue
        x_f, y_f = float(kps_xy[kid][0]), float(kps_xy[kid][1])
        if x_f <= 0 and y_f <= 0:
            continue
        y_c = PITCH_KEYPOINTS[kid][1]  # canvas y dari ground truth
        calib_pairs.append((y_f, y_c))
    
    if len(calib_pairs) < 2:
        # Tidak cukup data → fallback ke mapping default
        return generate_pseudo_keypoints(line_params, frame_h, frame_w), {"method": "default"}
    
    # Fit linear: y_canvas = a * y_frame + b
    y_frames  = np.array([p[0] for p in calib_pairs]).reshape(-1, 1)
    y_canvases = np.array([p[1] for p in calib_pairs])
    ones = np.ones_like(y_frames)
    A = np.hstack([y_frames, ones])
    result, _, _, _ = np.linalg.lstsq(A, y_canvases, rcond=None)
    a_y, b_y = result  # y_canvas = a_y * y_frame + b_y
    
    # Sampling pseudo-keypoints dengan mapping yang sudah di-kalibrasi
    y_fracs = np.linspace(0.15, 0.92, N_PSEUDO_KP)
    pseudo_kps = []
    for y_frac in y_fracs:
        y_src = y_frac * frame_h
        x_src = slope * y_src + intercept
        
        if not (0 <= x_src <= frame_w):
            continue
        
        y_canvas = a_y * y_src + b_y
        y_canvas = np.clip(y_canvas, MARGIN_Y - 20, CANVAS_H - MARGIN_Y + 20)
        
        pseudo_kps.append(([x_src, y_src], [HALFWAY_X_CANVAS, y_canvas]))
    
    return pseudo_kps, {
        "method": "calibrated",
        "n_calib": len(calib_pairs),
        "a_y": float(a_y),
        "b_y": float(b_y),
        "calib_pairs": calib_pairs
    }


# ══════════════════════════════════════════════════════════════════════════════
# HOMOGRAPHY (augmented dengan pseudo-keypoints)
# ══════════════════════════════════════════════════════════════════════════════

def compute_H_augmented(kps_xy, kps_conf, pseudo_kps, conf_thresh=CONF_THRESH):
    """
    Hitung homography dengan:
    1. Keypoint model (conf >= thresh)
    2. Pseudo-keypoints dari halfway line detection

    FIX A: reproj error dihitung DUA KALI:
      - internal (inliers saja) → untuk validasi H
      - kp_only reproj → hanya dari KP model asli, untuk monitoring akurasi sesungguhnya

    FIX C: spread check juga dilakukan di src (frame coords), bukan hanya dst (canvas)

    Return: H, src_pts_used, dst_pts_used, kp_src, kp_dst, status_str
    """
    src_pts, dst_pts, sources = [], [], []
    n_kp_pts = 0  # jumlah titik dari KP model (bukan pseudo)

    # ── Keypoint model ────────────────────────────────────────────────────────
    for kid in range(len(kps_conf)):
        c = float(kps_conf[kid])
        if c < conf_thresh or kid not in PITCH_KEYPOINTS:
            continue
        x, y = float(kps_xy[kid][0]), float(kps_xy[kid][1])
        if x <= 0 and y <= 0:
            continue
        src_pts.append([x, y])
        dst_pts.append(list(PITCH_KEYPOINTS[kid]))
        sources.append(f"kp_{kid}")
        n_kp_pts += 1

    # Simpan KP-only points untuk Fix A (reproj monitoring)
    kp_src = [s for s in src_pts]
    kp_dst = [d for d in dst_pts]

    # ── Pseudo-keypoints dari halfway line ───────────────────────────────────
    for (src_pt, dst_pt) in pseudo_kps:
        src_pts.append(src_pt)
        dst_pts.append(dst_pt)
        sources.append("pseudo")

    if len(src_pts) < 4:
        return None, src_pts, dst_pts, kp_src, kp_dst, f"too_few({len(src_pts)})"

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    # Spread check di dst (canvas)
    spread_x_dst = dst[:, 0].max() - dst[:, 0].min()
    spread_y_dst = dst[:, 1].max() - dst[:, 1].min()
    if spread_x_dst < 80 or spread_y_dst < 80:
        return None, src_pts, dst_pts, kp_src, kp_dst, \
               f"low_spread_dst(dx={spread_x_dst:.0f},dy={spread_y_dst:.0f})"

    # FIX C — Spread check juga di src (frame coords)
    spread_x_src = src[:, 0].max() - src[:, 0].min()
    spread_y_src = src[:, 1].max() - src[:, 1].min()
    if spread_x_src < 50 or spread_y_src < 50:
        return None, src_pts, dst_pts, kp_src, kp_dst, \
               f"low_spread_src(dx={spread_x_src:.0f},dy={spread_y_src:.0f})"

    # Collinear check di src
    centered = src - src.mean(axis=0)
    _, s, _ = np.linalg.svd(centered)
    ratio = s[1] / (s[0] + 1e-8)
    if ratio < 0.05:
        return None, src_pts, dst_pts, kp_src, kp_dst, f"collinear_src(ratio={ratio:.3f})"

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, src_pts, dst_pts, kp_src, kp_dst, "ransac_fail"

    # Validity check
    test_pts = np.array([[0,0],[1920,0],[1920,1080],[0,1080]], dtype=np.float32).reshape(-1,1,2)
    try:
        tc = cv2.perspectiveTransform(test_pts, H).reshape(-1, 2)
        margin = 10000
        for x, y in tc:
            if abs(x) > CANVAS_W + margin or abs(y) > CANVAS_H + margin:
                return None, src_pts, dst_pts, kp_src, kp_dst, "invalid_H(out_of_bounds)"
    except:
        return None, src_pts, dst_pts, kp_src, kp_dst, "invalid_H(transform_fail)"

    det = np.linalg.det(H[:2, :2])
    if det < 0.01:
        return None, src_pts, dst_pts, kp_src, kp_dst, f"invalid_H(det={det:.4f})"

    # Reproj error internal (inliers only) — untuk validasi threshold
    if mask is not None:
        inlier_mask = mask.ravel() == 1
        if inlier_mask.sum() < 4:
            return None, src_pts, dst_pts, kp_src, kp_dst, "too_few_inliers"
        src_in = src[inlier_mask]
        dst_in = dst[inlier_mask]
    else:
        src_in, dst_in = src, dst
        inlier_mask = np.ones(len(src_pts), dtype=bool)

    src_h = np.hstack([src_in, np.ones((len(src_in), 1))])
    proj  = (H @ src_h.T).T
    proj  = proj[:, :2] / proj[:, 2:3]
    err_internal = float(np.mean(np.linalg.norm(proj - dst_in, axis=1)))

    MAX_REPROJ = 80.0
    if err_internal >= MAX_REPROJ:
        return None, src_pts, dst_pts, kp_src, kp_dst, f"high_reproj_internal({err_internal:.1f}px)"

    n_pseudo_inliers = int(inlier_mask[n_kp_pts:].sum()) if mask is not None else len(pseudo_kps)
    status = (f"ok(internal={err_internal:.1f}px,"
              f"pseudo_inliers={n_pseudo_inliers}/{len(pseudo_kps)},"
              f"kp={n_kp_pts})")
    return H, src_pts, dst_pts, kp_src, kp_dst, status


def reproj_error_pts(H, src_pts, dst_pts):
    """Reproj error dari semua titik (KP + pseudo) — untuk info saja."""
    if H is None or len(src_pts) == 0:
        return None
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    sh  = np.hstack([src, np.ones((len(src), 1))])
    proj = (H @ sh.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return float(np.mean(np.linalg.norm(proj - dst, axis=1)))


def reproj_error_kp_only(H, kp_src, kp_dst):
    """
    FIX A — Reproj error HANYA dari keypoint model asli (bukan pseudo).
    Ini adalah metrik akurasi yang sesungguhnya: seberapa jauh H
    memetakan KP model ke posisi ground truth di canvas.
    """
    if H is None or len(kp_src) == 0:
        return None
    src = np.array(kp_src, dtype=np.float32)
    dst = np.array(kp_dst, dtype=np.float32)
    sh  = np.hstack([src, np.ones((len(src), 1))])
    proj = (H @ sh.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return float(np.mean(np.linalg.norm(proj - dst, axis=1)))


def transform_point(point, H):
    if H is None or point is None:
        return None
    pt = np.array([[point[0], point[1]]], dtype=np.float32).reshape(-1, 1, 2)
    try:
        t = cv2.perspectiveTransform(pt, H)
        return (int(t[0][0][0]), int(t[0][0][1]))
    except:
        return None


def is_valid_pos(pos, margin=80):
    if pos is None:
        return False
    x, y = pos
    return (-margin < x < CANVAS_W + margin) and (-margin < y < CANVAS_H + margin)


# ══════════════════════════════════════════════════════════════════════════════
# PITCH DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_pitch():
    img = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)
    lw = 2

    def fp(x_m, y_m): return m_to_px(x_m, y_m)

    cv2.rectangle(img, (MARGIN_X, MARGIN_Y),
                  (MARGIN_X + FIELD_W, MARGIN_Y + FIELD_H), (255, 255, 255), lw)
    cv2.line(img, fp(52.5, 0), fp(52.5, 68), (255, 255, 255), lw)
    cx, cy = fp(52.5, 34)
    cv2.circle(img, (cx, cy), int(9.15 / PITCH_WIDTH_M * FIELD_W), (255, 255, 255), lw)
    cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)
    cv2.rectangle(img, fp(0, 13.84), fp(16.5, 54.16), (255, 255, 255), lw)
    cv2.rectangle(img, fp(0, 24.84), fp(5.5,  43.16), (255, 255, 255), lw)
    cv2.rectangle(img, fp(88.5, 13.84), fp(105, 54.16), (255, 255, 255), lw)
    cv2.rectangle(img, fp(99.5, 24.84), fp(105, 43.16), (255, 255, 255), lw)

    for kid, (px, py) in PITCH_KEYPOINTS.items():
        cv2.circle(img, (px, py), 3, (100, 100, 100), -1)
        cv2.putText(img, str(kid), (px + 3, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 130, 130), 1)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total_frames} frames total\n")
print(f"{'Frame':>6} | {'KP≥0.3':>6} | {'Pseudo':>6} | {'H status':<50} | {'Reproj KP':>10} | {'Reproj all':>10} | Lines")
print("-" * 125)

for fn in TEST_FRAMES:
    if fn >= total_frames:
        print(f"Frame {fn}: out of range, skip")
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {fn}: read failed")
        continue

    frame_h, frame_w = frame.shape[:2]

    # ── 1. Keypoint model ────────────────────────────────────────────────────
    results  = kp_model(frame, verbose=False)[0]
    has_kp   = results.keypoints is not None and len(results.keypoints) > 0
    kps_xy   = results.keypoints.xy[0].cpu().numpy()   if has_kp else np.zeros((32, 2))
    kps_conf = results.keypoints.conf[0].cpu().numpy() if has_kp else np.zeros(32)
    high_conf_cnt = int((kps_conf >= CONF_THRESH).sum())

    # ── 2. Hough line detection ──────────────────────────────────────────────
    lines, mask_lines, edges = detect_field_lines(frame)
    n_h = sum(1 for l in lines if classify_line(*l) == 'H')
    n_v = sum(1 for l in lines if classify_line(*l) == 'V')
    n_d = sum(1 for l in lines if classify_line(*l) == 'D')

    # ── 3. Halfway line fit & pseudo-keypoints ───────────────────────────────
    line_params, line_debug = fit_halfway_line(lines, frame_w, frame_h)
    
    # Gunakan calibrated mapping jika ada keypoint model yang cukup
    pseudo_kps, calib_info = refine_y_mapping(
        kps_xy, kps_conf, line_params, frame_h, frame_w
    )
    
    # ── 4. Hitung H dengan pseudo-keypoints ──────────────────────────────────
    H, src_all, dst_all, kp_src, kp_dst, h_status = compute_H_augmented(kps_xy, kps_conf, pseudo_kps)
    err_all    = reproj_error_pts(H, src_all, dst_all)         # semua titik (info)
    err_kp     = reproj_error_kp_only(H, kp_src, kp_dst)       # FIX A: KP-only (metrik utama)

    print(f"{fn:>6} | {high_conf_cnt:>6} | {len(pseudo_kps):>6} | {h_status:<50} | "
          f"kp={f'{err_kp:.1f}px' if err_kp else 'N/A':<9} | all={f'{err_all:.1f}px' if err_all else 'N/A':<9} | "
          f"H={n_h} V={n_v} D={n_d}")

    # ── 5. Annotate frame ────────────────────────────────────────────────────
    frame_ann = frame.copy()

    # Garis Hough
    LINE_COLORS = {'H': (0, 200, 255), 'V': (255, 100, 0), 'D': (200, 0, 255)}
    for (x1, y1, x2, y2) in lines:
        t = classify_line(x1, y1, x2, y2)
        cv2.line(frame_ann, (x1, y1), (x2, y2), LINE_COLORS[t], 2)

    # Halfway line fitted (putih tebal)
    if line_params is not None:
        x_at_mid, slope, intercept = line_params
        y_top = int(frame_h * 0.15)
        y_bot = int(frame_h * 0.95)
        x_top = int(slope * y_top + intercept)
        x_bot = int(slope * y_bot + intercept)
        cv2.line(frame_ann, (x_top, y_top), (x_bot, y_bot), (255, 255, 255), 3)
        cv2.putText(frame_ann, "HALFWAY LINE (fitted)", (x_top + 5, y_top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Pseudo-keypoints di frame (magenta)
    for (src_pt, dst_pt) in pseudo_kps:
        sx, sy = int(src_pt[0]), int(src_pt[1])
        cv2.circle(frame_ann, (sx, sy), 10, (255, 0, 255), 2)
        cv2.circle(frame_ann, (sx, sy), 3,  (255, 0, 255), -1)
        cv2.putText(frame_ann, f"→({dst_pt[0]:.0f},{dst_pt[1]:.0f})",
                    (sx + 8, sy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

    # Keypoint model
    for kid in range(len(kps_conf)):
        c = float(kps_conf[kid])
        x, y = int(kps_xy[kid][0]), int(kps_xy[kid][1])
        if x <= 0 and y <= 0:
            continue
        color  = (0, 255, 0) if c >= CONF_THRESH else (0, 0, 200)
        radius = 8 if c >= CONF_THRESH else 4
        cv2.circle(frame_ann, (x, y), radius, color, -1)
        cv2.putText(frame_ann, f"{kid}:{c:.2f}", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)

    calib_method = calib_info.get("method", "none")
    n_calib      = calib_info.get("n_calib", 0)
    halfway_reason = line_debug.get("reason", "?")
    info = [
        f"Frame: {fn}  |  KP conf>={CONF_THRESH}: {high_conf_cnt}  |  H: {h_status[:55]}",
        f"Hough: {len(lines)} lines (H={n_h} V={n_v} D={n_d})  |  "
        f"Halfway: {halfway_reason}  |  Pseudo-KP: {len(pseudo_kps)} ({calib_method},{n_calib}pts)",
        f"Reproj KP-only: {f'{err_kp:.1f}px' if err_kp else 'N/A'}  |  "
        f"Reproj all: {f'{err_all:.1f}px' if err_all else 'N/A'}",
        "WHITE=halfway fitted | MAGENTA=pseudo-KP | GREEN=kp>=0.3 | BLUE=vert Hough",
    ]
    for i, txt in enumerate(info):
        cv2.putText(frame_ann, txt, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(frame_ann, txt, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    # Debug mask
    mw, mh = 280, 158
    mask_small = cv2.resize(cv2.cvtColor(mask_lines, cv2.COLOR_GRAY2BGR), (mw, mh))
    edge_small = cv2.resize(cv2.cvtColor(edges,      cv2.COLOR_GRAY2BGR), (mw, mh))
    panel = np.hstack([mask_small, edge_small])
    frame_ann[frame_h - mh - 5: frame_h - 5,
              frame_w - mw*2 - 5: frame_w - 5] = panel

    # ── 6. Tactical map ──────────────────────────────────────────────────────
    tac = draw_pitch()

    # Draw halfway line constraint di canvas (garis magenta vertikal)
    cv2.line(tac, (HALFWAY_X_CANVAS, MARGIN_Y),
             (HALFWAY_X_CANVAS, CANVAS_H - MARGIN_Y), (255, 0, 255), 1)
    cv2.putText(tac, "halfway constraint", (HALFWAY_X_CANVAS + 3, MARGIN_Y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

    if H is not None:
        # Keypoint model → transformed (merah)
        for kid in range(len(kps_conf)):
            c = float(kps_conf[kid])
            if c < CONF_THRESH or kid not in PITCH_KEYPOINTS:
                continue
            x, y = float(kps_xy[kid][0]), float(kps_xy[kid][1])
            if x <= 0 and y <= 0:
                continue
            mapped = transform_point((x, y), H)
            if mapped and is_valid_pos(mapped):
                gt = PITCH_KEYPOINTS[kid]
                cv2.line(tac, mapped, gt, (0, 200, 255), 1)  # error line kuning
                cv2.circle(tac, mapped, 7, (0, 0, 255), -1)
                cv2.putText(tac, str(kid), (mapped[0]+4, mapped[1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)

        # Pseudo-keypoints → transformed (magenta)
        for (src_pt, dst_pt) in pseudo_kps:
            mapped = transform_point(src_pt, H)
            if mapped and is_valid_pos(mapped):
                target = (int(dst_pt[0]), int(dst_pt[1]))
                cv2.line(tac, mapped, target, (200, 0, 200), 1)
                cv2.circle(tac, mapped, 5, (255, 0, 255), -1)

        # Camera FOV box
        corners = np.array([[0,0],[frame_w,0],[frame_w,frame_h],[0,frame_h]],
                            dtype=np.float32).reshape(-1,1,2)
        try:
            tc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            cv2.polylines(tac, [np.int32(tc)], True, (255, 165, 0), 2)
        except:
            pass

    # Legend
    legend = [
        "RED    = model KP transformed",
        "MAGENTA= pseudo-KP (halfway line)",
        "GRAY   = ground truth PITCH_KEYPOINTS",
        "ORANGE = camera FOV",
        "MAGENTA line = halfway constraint (x=525)",
        f"reproj KP-only: {f'{err_kp:.1f}px' if err_kp else 'N/A'}  (FIX A)",
        f"reproj all: {f'{err_all:.1f}px' if err_all else 'N/A'}",
        f"pseudo-KP: {len(pseudo_kps)}  calib: {calib_method}({n_calib}pts)",
    ]
    for i, t in enumerate(legend):
        cv2.putText(tac, t, (8, CANVAS_H - 130 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1)

    # ── 7. Combine & save ────────────────────────────────────────────────────
    display_w = min(frame_w, 1280)
    scale     = display_w / frame_w
    display_h = int(frame_h * scale)

    frame_disp = cv2.resize(frame_ann, (display_w, display_h))
    tac_disp_w = int(display_h * CANVAS_W / CANVAS_H)
    tac_disp   = cv2.resize(tac, (tac_disp_w, display_h))

    divider  = np.ones((display_h, 4, 3), dtype=np.uint8) * 180
    combined = np.hstack([frame_disp, divider, tac_disp])

    out_path = os.path.join(OUTPUT_DIR, f"frame_{fn:04d}.jpg")
    cv2.imwrite(out_path, combined)

cap.release()
print(f"\nDone! Output: {OUTPUT_DIR}/")
print("""
Cara baca output:
  Frame kiri:
    Putih tebal   = halfway line yang di-fit dari Hough
    Magenta circle= pseudo-keypoints (titik di sepanjang halfway line)
    Label →(x,y)  = target posisi di canvas
    Hijau besar   = KP model conf >= 0.3
    Biru garis    = Hough vertikal

  Tactical map kanan:
    Magenta vertikal = halfway line constraint (x=525)
    Merah circle     = KP model setelah transform
    Magenta circle   = pseudo-KP setelah transform (seharusnya dekat x=525)
    Garis kuning     = error KP vs ground truth
    Orange box       = area kamera (FOV)

Yang perlu diperhatikan:
  1. Apakah pseudo-KP magenta di tac map jatuh tepat di/dekat garis magenta (x=525)?
     → Kalau ya, constraint halfway line bekerja dengan benar
  2. Apakah FOV box sekarang bentuk trapezoid yang masuk akal?
  3. Apakah reproj error turun dari sebelumnya (49.7px)?
  4. Cek label "calib" di legend:
     - calibrated(Npts) = bagus, y-mapping pakai keypoint model
     - default(0pts)    = y-mapping pakai asumsi linear (kurang akurat)
""")