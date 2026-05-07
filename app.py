"""
app.py
Flask backend untuk Football Detection Dashboard.

Endpoint:
  POST /api/upload                — upload video + mulai proses analisis (background thread)
  GET  /api/videos                — list semua video (paginasi, search, filter status)
  GET  /api/video/<id>            — detail satu video + hasil analisis
  PUT  /api/video/<id>            — update nama/tim video
  DELETE /api/video/<id>          — hapus video + semua data analisis
  GET  /api/image/<analysis_id>/<image_type>  — serve gambar dari BLOB database

Database: MySQL via PyMySQL
Run: python app.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Fix KMeans memory leak on Windows MKL

import io
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import pymysql
import pymysql.cursors
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=".", static_url_path="")

# ── Folder lokal ──
UPLOAD_FOLDER   = os.path.join(os.path.dirname(__file__), "input_video")
OUTPUT_FOLDER   = os.path.join(os.path.dirname(__file__), "output_videos")
ANALYSIS_FOLDER = os.path.join(os.path.dirname(__file__), "output_analysis")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

os.makedirs(UPLOAD_FOLDER,   exist_ok=True)
os.makedirs(OUTPUT_FOLDER,   exist_ok=True)
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

# ── MySQL config ──
DB_CONFIG = {
    "host":        "localhost",
    "port":        3306,
    "user":        "root",
    "password":    "",           # ← isi password MySQL kamu
    "db":          "soccana_football_detection",
    "charset":     "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit":  True,
}

# ── Model path ──
TRACKER_MODEL = "models/RFDETR Result Dataset With Augmentation Version 2/checkpoint_best_total.pth"
BALL_MODEL    = "models/best_yolo8s_ball-detection.pt"
KP_MODEL      = "models/kpdet_best_final.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    return pymysql.connect(**DB_CONFIG)


def db_execute(sql: str, args=None, fetch: str = "none"):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, args or ())
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return cur.lastrowid
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def hex_to_bgr(hex_color: str) -> tuple:
    """Konversi warna hex (#RRGGBB) → tuple BGR untuk OpenCV."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)   # OpenCV = BGR


def get_video_info(filepath: str) -> dict:
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return {}
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dur = int(frames / fps) if fps > 0 else 0
    return {
        "fps":          round(fps, 2),
        "duration_sec": dur,
        "resolution":   f"{w}x{h}",
    }


def image_to_blob(path: str) -> bytes | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def save_image_to_db(analysis_id: int, image_type: str, image_path: str) -> None:
    blob = image_to_blob(image_path)
    if blob is None:
        print(f"[DB] WARNING: Gambar tidak ditemukan: {image_path}")
        return
    ext     = os.path.splitext(image_path)[1].lower()
    mime    = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    size_kb = len(blob) // 1024
    db_execute(
        """INSERT INTO AnalysisImages
               (analysis_id, image_type, image_data, mime_type, file_size_kb)
           VALUES (%s, %s, %s, %s, %s)
           ON DUPLICATE KEY UPDATE
               image_data=%s, mime_type=%s, file_size_kb=%s""",
        (analysis_id, image_type, blob, mime, size_kb,
         blob, mime, size_kb),
    )
    print(f"[DB] Saved image '{image_type}' ({size_kb} KB) → analysis_id={analysis_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(video_id: int, video_path: str, color_config: dict) -> None:
    """
    Jalankan full pipeline di background thread.
    color_config berisi warna yang dipilih user dari UI (hex strings).
    """
    t_start = time.time()

    try:
        db_execute(
            "UPDATE Videos SET status='processing', progress_pct=5 WHERE video_id=%s",
            (video_id,),
        )

        # ── Import modul analisis ─────────────────────────────────────────────
        from trackers                     import Tracker
        from team_assigner                import TeamAssigner
        from team_ball_control            import TeamBallControl
        from player_ball_assigner         import PlayerBallAssigner
        from camera_movement_estimator    import CameraMovementEstimator
        from speed_and_distance_estimator import SpeedAndDistance_Estimator
        # Import KeypointDetector — coba semua kemungkinan nama file
        # Prioritas: keypoint_det_soccana (versi terbaru dengan detect_batch + 3 filter)
        _kp_module = None
        for _mod_name in (
            "keypoint_det_soccana",       # versi terbaru (detect_batch, 3 filter)
            "keypoint_detector",           # nama generik
            "keypoint_detection_Soccana",  # versi lama (detect_keypoints_batch saja)
        ):
            try:
                import importlib as _il
                _kp_module = _il.import_module(_mod_name)
                print(f"[Pipeline {video_id}] KeypointDetector loaded from: {_mod_name}")
                break
            except ImportError:
                continue
        if _kp_module is None:
            raise ImportError("Tidak bisa menemukan modul KeypointDetector. "
                              "Pastikan salah satu file berikut ada: "
                              "keypoint_det_soccana.py / keypoint_detector.py / keypoint_detection_Soccana.py")
        KeypointDetector = _kp_module.KeypointDetector
        from homography                   import HomographyCalculator
        from tactical_map                 import TacticalMapRenderer
        from utils                        import read_video, save_video
        from heatmap_analyzer             import HeatmapAnalyzer
        from zone_analyzer                import ZoneAnalyzer
        import numpy as np

        # ── Konversi warna user (hex → BGR tuple) ────────────────────────────
        t1_sec_bgr = hex_to_bgr(color_config.get("team1_color_secondary",  "#E53E3E"))
        t1_gk_bgr  = hex_to_bgr(color_config.get("team1_goalkeeper_color", "#F6AD55"))
        t2_sec_bgr = hex_to_bgr(color_config.get("team2_color_secondary",  "#3182CE"))
        t2_gk_bgr  = hex_to_bgr(color_config.get("team2_goalkeeper_color", "#48BB78"))
        ref_bgr    = hex_to_bgr(color_config.get("referee_color",           "#ECC94B"))

        # ── Patch module-level dicts di team_assigner ───────────────────────
        # Dilakukan dengan hasattr agar aman meski nama atribut berbeda.
        import team_assigner as ta_module
        import numpy as _np

        def _bgr_to_rgb(bgr: tuple) -> _np.ndarray:
            return _np.array([bgr[2], bgr[1], bgr[0]])

        # ANNOTATION_COLORS_BGR: warna bounding-box yang ditampilkan di video
        if hasattr(ta_module, "ANNOTATION_COLORS_BGR"):
            ta_module.ANNOTATION_COLORS_BGR[1] = t1_sec_bgr
            ta_module.ANNOTATION_COLORS_BGR[2] = t2_sec_bgr
        else:
            # Fallback: buat dict baru jika atribut belum ada
            ta_module.ANNOTATION_COLORS_BGR = {1: t1_sec_bgr, 2: t2_sec_bgr}

        # TEAM_COLORS: warna jersey untuk klasifikasi tim (RGB numpy array)
        if hasattr(ta_module, "TEAM_COLORS"):
            ta_module.TEAM_COLORS[1]["goalkeeper"] = _bgr_to_rgb(t1_gk_bgr)
            ta_module.TEAM_COLORS[2]["goalkeeper"] = _bgr_to_rgb(t2_gk_bgr)

        # ── 1. Baca video ─────────────────────────────────────────────────────
        print(f"[Pipeline {video_id}] Reading video...")
        video_frames = read_video(video_path)
        frame_h, frame_w = video_frames[0].shape[:2]
        total_frames = len(video_frames)
        print(f"[Pipeline {video_id}] Total frames: {total_frames}")
        db_execute("UPDATE Videos SET progress_pct=10 WHERE video_id=%s", (video_id,))

        # ── 2. Tracking ───────────────────────────────────────────────────────
        print(f"[Pipeline {video_id}] Tracking...")
        tracker = Tracker(model_path=TRACKER_MODEL, ball_model_path=BALL_MODEL)
        tracks  = tracker.get_object_tracks(video_frames, read_from_stub=False)
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
        tracker.add_object_positions(tracks)
        db_execute("UPDATE Videos SET progress_pct=30 WHERE video_id=%s", (video_id,))

        # ── 3. Camera movement ────────────────────────────────────────────────
        cam_estimator = CameraMovementEstimator(video_frames[0])
        cam_movement  = cam_estimator.get_camera_movement(video_frames, read_from_stub=False)
        cam_estimator.add_adjust_positions_to_tracks(tracks, cam_movement)

        # ── 4. Team assignment ────────────────────────────────────────────────
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames, tracks["player"], sample_frames=10)

        # Override annotation colors pada instance (warna bbox video output)
        team_assigner.team_colors[1] = t1_sec_bgr
        team_assigner.team_colors[2] = t2_sec_bgr

        for frame_num, player_track in enumerate(tracks["player"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id,
                    is_goalkeeper=False,
                )
                tracks["player"][frame_num][player_id]["team"]       = team
                tracks["player"][frame_num][player_id]["team_color"] = \
                    team_assigner.team_colors.get(team, (200, 200, 200))

        for frame_num, gk_track in enumerate(tracks["goalkeeper"]):
            for gk_id, track in gk_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num], track["bbox"], gk_id,
                    is_goalkeeper=True,
                )
                # Warna goalkeeper sesuai pilihan user
                gk_color = t1_gk_bgr if team == 1 else t2_gk_bgr
                tracks["goalkeeper"][frame_num][gk_id]["team"]       = team
                tracks["goalkeeper"][frame_num][gk_id]["team_color"] = gk_color

        db_execute("UPDATE Videos SET progress_pct=45 WHERE video_id=%s", (video_id,))

        # ── 5. Speed & Distance ───────────────────────────────────────────────
        speed_est = SpeedAndDistance_Estimator()
        speed_est.add_speed_and_distance_to_tracks(tracks)

        # ── 6. Ball possession ────────────────────────────────────────────────
        player_assigner = PlayerBallAssigner()
        team_control    = TeamBallControl()
        for frame_num, player_track in enumerate(tracks["player"]):
            ball_frm = tracks["ball"][frame_num]
            if not ball_frm:
                team_control.update(-1, player_track)
                continue
            ball_bbox       = ball_frm[1]["bbox"]
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            team_control.update(assigned_player, tracks["player"][frame_num])
            if assigned_player != -1:
                tracks["player"][frame_num][assigned_player]["has_ball"] = True
        team_ball_control = np.array(team_control.get_sequence())
        db_execute("UPDATE Videos SET progress_pct=55 WHERE video_id=%s", (video_id,))

        # ── 7. Keypoint detection ─────────────────────────────────────────────
        print(f"[Pipeline {video_id}] Detecting keypoints...")
        kp_detector = KeypointDetector(model_path=KP_MODEL, confidence_threshold=0.5)

        # Patch: tambahkan detect_batch ke instance jika tidak ada
        # (untuk kompatibilitas dengan keypoint_detection_Soccana.py versi lama)
        if not hasattr(kp_detector, "detect_batch"):
            def _detect_batch_patch(frames, batch_size=8):
                """Patch detect_batch untuk versi KeypointDetector lama."""
                results = []
                for i, frame in enumerate(frames):
                    if i % 50 == 0:
                        print(f"[KeypointDetector] {i}/{len(frames)} frames...")
                    # detect_keypoints_batch ada di versi lama
                    if hasattr(kp_detector, "detect_keypoints"):
                        kp = kp_detector.detect_keypoints(frame)
                    elif hasattr(kp_detector, "detect"):
                        kp, _ = kp_detector.detect(frame)
                    else:
                        kp = {}
                    results.append(kp)
                print(f"[KeypointDetector] Done — {len(frames)} frames processed.")
                return results
            import types
            kp_detector.detect_batch = types.MethodType(
                lambda self, frames, batch_size=8: _detect_batch_patch(frames, batch_size),
                kp_detector
            )

        raw_kp = kp_detector.detect_batch(video_frames)

        # Normalisasi output: pastikan semua entry adalah dict {int: (float,float)}
        # dan buang koordinat di luar batas frame
        all_kp = []
        for kp in raw_kp:
            if kp is None:
                all_kp.append({})
                continue
            filtered = {}
            for kid, val in kp.items():
                # val bisa (x, y) atau (x, y, conf) tergantung versi
                x, y = float(val[0]), float(val[1])
                if 0 <= x < frame_w and 0 <= y < frame_h:
                    filtered[int(kid)] = (x, y)
            all_kp.append(filtered)
        db_execute("UPDATE Videos SET progress_pct=65 WHERE video_id=%s", (video_id,))

        # ── 8. Ball trajectory ────────────────────────────────────────────────
        renderer         = TacticalMapRenderer()
        heatmap_analyzer = HeatmapAnalyzer(
            canvas_w=960, canvas_h=560,
            gaussian_radius=25, output_scale=2.0,
        )
        zone_analyzer   = ZoneAnalyzer(
            canvas_w=960, canvas_h=560,
            grid_cols=6, grid_rows=4, output_scale=2.0,
        )
        homography_calc = HomographyCalculator(min_keypoints=4)

        hc_pass1     = HomographyCalculator(min_keypoints=4)
        raw_ball_pos = []
        for frame_num, kp in enumerate(all_kp):
            H        = hc_pass1.get_homography(kp)
            ball_frm = tracks["ball"][frame_num]
            pos      = None
            if ball_frm:
                bpos = ball_frm[1].get("position_adjusted") or ball_frm[1].get("position")
                if bpos:
                    candidate = hc_pass1.transform_point(
                        bpos, H,
                        canvas_w=renderer.canvas_w,
                        canvas_h=renderer.canvas_h,
                        margin=0,
                    )
                    if candidate is not None:
                        cx, cy = candidate
                        if 0 <= cx <= renderer.canvas_w and 0 <= cy <= renderer.canvas_h:
                            pos = candidate
            raw_ball_pos.append(pos)

        MAX_JUMP_PX = 120
        WINDOW      = 9
        filtered_ball = []
        prev = None
        for p in raw_ball_pos:
            if p is None:
                filtered_ball.append(None)
                continue
            if prev is not None:
                dist = ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5
                if dist > MAX_JUMP_PX:
                    filtered_ball.append(None)
                    prev = None
                    continue
            filtered_ball.append(p)
            prev = p

        xs   = [p[0] if p else None for p in filtered_ball]
        ys   = [p[1] if p else None for p in filtered_ball]
        half = WINDOW // 2
        ball_trail_smooth = []
        for i in range(len(filtered_ball)):
            wx = [xs[j] for j in range(max(0, i - half), min(len(xs), i + half + 1))
                  if xs[j] is not None]
            wy = [ys[j] for j in range(max(0, i - half), min(len(ys), i + half + 1))
                  if ys[j] is not None]
            if wx and wy:
                ball_trail_smooth.append((int(np.median(wx)), int(np.median(wy))))
            else:
                ball_trail_smooth.append(None)

        # ── 9. Build tactical frames + collect data ───────────────────────────
        print(f"[Pipeline {video_id}] Rendering tactical map...")
        tactical_frames = []
        for frame_num, kp in enumerate(all_kp):
            H = homography_calc.get_homography(kp)
            player_positions = {}

            for player_id, track in tracks["player"][frame_num].items():
                pos = track.get("position_adjusted") or track.get("position")
                if pos is None:
                    continue
                map_pos = homography_calc.transform_point(
                    pos, H,
                    canvas_w=renderer.canvas_w,
                    canvas_h=renderer.canvas_h,
                    margin=5,
                )
                player_positions[player_id] = {
                    "map_pos": map_pos,
                    "team":    track.get("team", -1),
                    "role":    "player",
                }

            for gk_id, track in tracks["goalkeeper"][frame_num].items():
                pos = track.get("position_adjusted") or track.get("position")
                if pos is None:
                    continue
                map_pos = homography_calc.transform_point(
                    pos, H,
                    canvas_w=renderer.canvas_w,
                    canvas_h=renderer.canvas_h,
                    margin=5,
                )
                player_positions[f"gk_{gk_id}"] = {
                    "map_pos": map_pos,
                    "team":    track.get("team", -1),
                    "role":    "goalkeeper",
                }

            tac_frame = renderer.render(
                player_positions=player_positions,
                ball_position=ball_trail_smooth[frame_num],
                ball_trail=ball_trail_smooth[: frame_num + 1],
            )
            heatmap_analyzer.collect(
                frame_num=frame_num,
                tracks=tracks,
                homography_calc=homography_calc,
                keypoints=kp,
                team_ball_control=team_ball_control,
            )
            zone_analyzer.collect(
                frame_num=frame_num,
                tracks=tracks,
                homography_calc=homography_calc,
                keypoints=kp,
                ball_map_pos=ball_trail_smooth[frame_num],
            )
            tactical_frames.append(tac_frame)

        db_execute("UPDATE Videos SET progress_pct=75 WHERE video_id=%s", (video_id,))

        # ── 10. Save analysis images ke folder sementara ──────────────────────
        video_name = (
            os.path.basename(video_path)
            .replace(".mp4", "").replace(".avi", "")
            .replace(".mov", "").replace(".mkv", "")
        )
        tmp_dir = os.path.join(ANALYSIS_FOLDER, f"tmp_{video_id}")
        os.makedirs(tmp_dir, exist_ok=True)

        saved_hm   = heatmap_analyzer.save_all(output_dir=tmp_dir, video_name=video_name, fmt="png")
        saved_zone = zone_analyzer.save_all(output_dir=tmp_dir, video_name=video_name, fmt="png")

        # ── 11. Hitung statistik ──────────────────────────────────────────────
        ctrl   = team_ball_control
        t1_pct = float((ctrl == 1).sum() / max(len(ctrl), 1) * 100)
        t2_pct = float((ctrl == 2).sum() / max(len(ctrl), 1) * 100)
        hm_stats = heatmap_analyzer.get_summary_stats()

        def avg_speed(team_id: int) -> float:
            speeds = []
            for frame in tracks["player"]:
                for _, data in frame.items():
                    if data.get("team") == team_id and data.get("speed") is not None:
                        speeds.append(data["speed"])
            return round(float(np.mean(speeds)), 1) if speeds else 0.0

        avg_s1 = avg_speed(1)
        avg_s2 = avg_speed(2)

        all_speeds = [
            d["speed"]
            for frame in tracks["player"]
            for _, d in frame.items()
            if d.get("speed") is not None
        ]
        player_speed_max = round(max(all_speeds), 1) if all_speeds else 0.0

        # ── 12. Annotasi video ────────────────────────────────────────────────
        print(f"[Pipeline {video_id}] Drawing annotations...")
        output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        output_frames = cam_estimator.draw_camera_movement(output_frames, cam_movement)
        speed_est.draw_speed_and_distance(output_frames, tracks)

        # Gambar ulang bbox wasit dengan warna pilihan user
        for frame_num, frame in enumerate(output_frames):
            for track_id, referee in tracks["referee"][frame_num].items():
                tracker.draw_ellipse(frame, referee["bbox"], ref_bgr, track_id)

        # Side-by-side
        tac_h   = frame_h
        tac_w   = int(tac_h * renderer.canvas_w / renderer.canvas_h)
        divider = np.full((frame_h, 4, 3), 180, dtype=np.uint8)
        combined_frames = [
            np.hstack([output_frames[i], divider,
                       cv2.resize(tactical_frames[i], (tac_w, tac_h))])
            for i in range(len(output_frames))
        ]

        # ── 13. Simpan output video ───────────────────────────────────────────
        out_filename = f"{video_name}_output_{video_id}.avi"
        out_path     = os.path.join(OUTPUT_FOLDER, out_filename)
        save_video(combined_frames, out_path)
        print(f"[Pipeline {video_id}] Saved video → {out_path}")
        db_execute("UPDATE Videos SET progress_pct=88 WHERE video_id=%s", (video_id,))

        # ── 14. Simpan ke database ────────────────────────────────────────────
        t_elapsed = round(time.time() - t_start, 1)

        analysis_id = db_execute(
            """INSERT INTO AnalysisResults
               (video_id, model_name, tracking_method, processing_time_sec,
                anotated_video_path, ball_posession_home, ball_posession_away,
                player_speed_max_kmh, avg_speed_team1_kmh, avg_speed_team2_kmh,
                total_frames, analysis_date)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                video_id, "RF-DETR Base", "ByteTrack", t_elapsed,
                out_path, round(t1_pct, 1), round(t2_pct, 1),
                player_speed_max, avg_s1, avg_s2,
                total_frames, datetime.now(),
            ),
        )

        # Zone stats JSON
        zone_stats_path = saved_zone.get("zone_stats", "")
        if zone_stats_path and os.path.exists(zone_stats_path):
            with open(zone_stats_path, encoding="utf-8") as f:
                zone_stats_json = json.load(f)
        else:
            zone_stats_json = {}

        db_execute(
            """INSERT INTO AnalysisMetadata (analysis_id, heatmap_stats, zone_stats)
               VALUES (%s, %s, %s)""",
            (analysis_id, json.dumps(hm_stats), json.dumps(zone_stats_json)),
        )

        # Simpan gambar ke BLOB
        image_map = {
            "heatmap_team1":    saved_hm.get("heatmap_team1"),
            "heatmap_team2":    saved_hm.get("heatmap_team2"),
            "heatmap_combined": saved_hm.get("heatmap_combined"),
            "zone_control":     saved_hm.get("zone_control"),
            "zone_grid":        saved_zone.get("zone_grid"),
            "zone_voronoi":     saved_zone.get("zone_voronoi"),
        }
        for img_type, img_path in image_map.items():
            if img_path and os.path.exists(img_path):
                save_image_to_db(analysis_id, img_type, img_path)

        db_execute("UPDATE Videos SET progress_pct=95 WHERE video_id=%s", (video_id,))
        db_execute(
            "UPDATE Videos SET file_path_output=%s WHERE video_id=%s",
            (out_path, video_id),
        )

        # ── Selesai ───────────────────────────────────────────────────────────
        db_execute(
            "UPDATE Videos SET status='done', progress_pct=100 WHERE video_id=%s",
            (video_id,),
        )
        print(f"[Pipeline {video_id}] Done in {t_elapsed}s")

        # Hapus file sementara (gambar sudah tersimpan di DB)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"[Pipeline {video_id}] ERROR:\n{err_msg}")
        db_execute(
            "UPDATE Videos SET status='error', error_msg=%s WHERE video_id=%s",
            (str(e)[:2000], video_id),
        )


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")


# ── Upload & process video ──────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "message": "File tidak ditemukan di request."}), 400

    f = request.files["video"]
    if not f or not allowed_file(f.filename):
        return jsonify({"success": False,
                        "message": "Format file tidak didukung. Gunakan MP4/AVI/MOV/MKV."}), 400

    video_name = request.form.get("video_name", "").strip() or secure_filename(f.filename)
    team1      = request.form.get("team1", "Tim 1").strip()
    team2      = request.form.get("team2", "Tim 2").strip()

    color_config = {
        "team1_color_secondary":  request.form.get("team1_color_secondary",  "#E53E3E"),
        "team1_goalkeeper_color": request.form.get("team1_goalkeeper_color", "#F6AD55"),
        "team2_color_secondary":  request.form.get("team2_color_secondary",  "#3182CE"),
        "team2_goalkeeper_color": request.form.get("team2_goalkeeper_color", "#48BB78"),
        "referee_color":          request.form.get("referee_color",           "#ECC94B"),
    }

    filename  = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    vinfo   = get_video_info(save_path)

    video_id = db_execute(
        """INSERT INTO Videos
           (video_name, team1_name, team1_color_secondary, team1_goalkeeper_color,
            team2_name, team2_color_secondary, team2_goalkeeper_color,
            referee_color, file_path_raw, file_size_mb,
            status, progress_pct, duration_sec, fps, resolution)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'queued',0,%s,%s,%s)""",
        (
            video_name, team1,
            color_config["team1_color_secondary"],
            color_config["team1_goalkeeper_color"],
            team2,
            color_config["team2_color_secondary"],
            color_config["team2_goalkeeper_color"],
            color_config["referee_color"],
            save_path, round(size_mb, 2),
            vinfo.get("duration_sec"), vinfo.get("fps"), vinfo.get("resolution"),
        ),
    )

    t = threading.Thread(
        target=run_pipeline,
        args=(video_id, save_path, color_config),
        daemon=True,
    )
    t.start()

    return jsonify({
        "success":  True,
        "video_id": video_id,
        "message":  "Upload berhasil. Pipeline sedang berjalan.",
    })


# ── List semua video ─────────────────────────────────────────────────────────

@app.route("/api/videos", methods=["GET"])
def list_videos():
    page   = max(1, int(request.args.get("page", 1)))
    limit  = min(50, max(1, int(request.args.get("limit", 12))))
    offset = (page - 1) * limit
    q      = request.args.get("q", "").strip()
    status = request.args.get("status", "").strip()

    where  = []
    params = []
    if q:
        where.append("video_name LIKE %s")
        params.append(f"%{q}%")
    if status:
        where.append("status = %s")
        params.append(status)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    total = db_execute(
        f"SELECT COUNT(*) as cnt FROM Videos {where_sql}", params, fetch="one"
    )
    total = total["cnt"] if total else 0

    rows = db_execute(
        f"""SELECT video_id, video_name, team1_name, team2_name,
                   status, progress_pct, duration_sec, fps, resolution,
                   created_at, updated_at
            FROM Videos {where_sql}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s""",
        params + [limit, offset],
        fetch="all",
    )

    videos = []
    for r in (rows or []):
        r["created_at"] = str(r["created_at"]) if r["created_at"] else None
        r["updated_at"] = str(r["updated_at"]) if r["updated_at"] else None
        videos.append(r)

    return jsonify({"total": total, "videos": videos, "page": page, "limit": limit})


# ── Detail satu video ────────────────────────────────────────────────────────

@app.route("/api/video/<int:video_id>", methods=["GET"])
def get_video(video_id: int):
    v = db_execute("SELECT * FROM Videos WHERE video_id=%s", (video_id,), fetch="one")
    if not v:
        return jsonify({"success": False, "message": "Video tidak ditemukan."}), 404

    for k in ("created_at", "updated_at"):
        if v.get(k):
            v[k] = str(v[k])

    ar = db_execute(
        "SELECT * FROM AnalysisResults WHERE video_id=%s ORDER BY created_at DESC LIMIT 1",
        (video_id,), fetch="one",
    )
    result = {}
    if ar:
        ar_id = ar["analysis_id"]
        for k in ("analysis_date", "created_at"):
            if ar.get(k):
                ar[k] = str(ar[k])

        meta_row = db_execute(
            "SELECT * FROM AnalysisMetadata WHERE analysis_id=%s", (ar_id,), fetch="one"
        )
        meta = {}
        if meta_row:
            for key in ("heatmap_stats", "zone_stats", "extra_data"):
                val = meta_row.get(key)
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except Exception:
                        pass
                meta[key] = val

        images = db_execute(
            "SELECT image_id, image_type, mime_type, file_size_kb "
            "FROM AnalysisImages WHERE analysis_id=%s",
            (ar_id,), fetch="all",
        ) or []

        result = {**ar, "metadata": meta, "images": images}

    color_info = {
        "team1_color_secondary":  v.pop("team1_color_secondary",  "#E53E3E"),
        "team1_goalkeeper_color": v.pop("team1_goalkeeper_color", "#F6AD55"),
        "team2_color_secondary":  v.pop("team2_color_secondary",  "#3182CE"),
        "team2_goalkeeper_color": v.pop("team2_goalkeeper_color", "#48BB78"),
        "referee_color":          v.pop("referee_color",           "#ECC94B"),
    }

    return jsonify({**v, "result": result, "color_config": color_info})


# ── Serve gambar dari BLOB ───────────────────────────────────────────────────

@app.route("/api/image/<int:video_id>/<image_type>", methods=["GET"])
def serve_image(video_id: int, image_type: str):
    ar = db_execute(
        "SELECT analysis_id FROM AnalysisResults WHERE video_id=%s "
        "ORDER BY created_at DESC LIMIT 1",
        (video_id,), fetch="one",
    )
    if not ar:
        return jsonify({"error": "Analysis tidak ditemukan."}), 404

    row = db_execute(
        "SELECT image_data, mime_type FROM AnalysisImages "
        "WHERE analysis_id=%s AND image_type=%s",
        (ar["analysis_id"], image_type), fetch="one",
    )
    if not row or not row["image_data"]:
        return jsonify({"error": f"Gambar '{image_type}' tidak ditemukan."}), 404

    return send_file(
        io.BytesIO(row["image_data"]),
        mimetype=row["mime_type"],
        as_attachment=False,
        download_name=f"{image_type}.png",
    )


# ── Serve video output dari lokal ────────────────────────────────────────────

@app.route("/api/video-stream/<int:video_id>", methods=["GET"])
def serve_video(video_id: int):
    v = db_execute(
        "SELECT file_path_output FROM Videos WHERE video_id=%s", (video_id,), fetch="one"
    )
    if not v or not v["file_path_output"]:
        return jsonify({"error": "Video output tidak ditemukan."}), 404

    path = v["file_path_output"]
    if not os.path.exists(path):
        return jsonify({"error": "File tidak ada di disk."}), 404

    return send_from_directory(os.path.dirname(path), os.path.basename(path))


# ── Update metadata video ────────────────────────────────────────────────────

@app.route("/api/video/<int:video_id>", methods=["PUT"])
def update_video(video_id: int):
    data       = request.get_json(silent=True) or {}
    video_name = data.get("video_name", "").strip()
    if not video_name:
        return jsonify({"success": False, "message": "Nama video tidak boleh kosong."}), 400

    db_execute(
        "UPDATE Videos SET video_name=%s, team1_name=%s, team2_name=%s WHERE video_id=%s",
        (video_name, data.get("team1_name", "Tim 1"),
         data.get("team2_name", "Tim 2"), video_id),
    )
    return jsonify({"success": True, "message": "Data diperbarui."})


# ── Hapus video ──────────────────────────────────────────────────────────────

@app.route("/api/video/<int:video_id>", methods=["DELETE"])
def delete_video(video_id: int):
    v = db_execute(
        "SELECT file_path_raw, file_path_output FROM Videos WHERE video_id=%s",
        (video_id,), fetch="one",
    )
    if not v:
        return jsonify({"success": False, "message": "Video tidak ditemukan."}), 404

    for key in ("file_path_raw", "file_path_output"):
        p = v.get(key)
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    db_execute("DELETE FROM Videos WHERE video_id=%s", (video_id,))
    return jsonify({"success": True, "message": "Video dan seluruh data analisis dihapus."})


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Football Detection — Flask Server")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)