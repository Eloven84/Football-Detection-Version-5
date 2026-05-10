"""
app.py
Flask backend untuk Football Detection & Analysis System.
Calvin Institute of Technology — Tugas Akhir

Endpoints:
  POST   /api/upload              — Upload video + mulai analisis (background thread)
  GET    /api/videos              — List semua video (pagination, search, filter)
  GET    /api/video/<id>          — Detail satu video + hasil analisis
  PUT    /api/video/<id>          — Update metadata video
  DELETE /api/video/<id>          — Hapus video + hasil
  GET    /outputs/<id>/<filename> — Serve file output (video, heatmap, dll)
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import mysql.connector
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
OUTPUT_DIR  = BASE_DIR / "outputs"
UPLOAD_DIR  = BASE_DIR / "uploads"
STUB_DIR    = BASE_DIR / "stubs"

for d in (OUTPUT_DIR, UPLOAD_DIR, STUB_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

# ── Import modul analisis ─────────────────────────────────────────────────────
from trackers                     import Tracker
from team_assigner                import TeamAssigner
from team_ball_control            import TeamBallControl
from player_ball_assigner         import PlayerBallAssigner
from camera_movement_estimator    import CameraMovementEstimator
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from keypoint_detector            import KeypointDetector
from homography                   import HomographyCalculator
from tactical_map                 import TacticalMapRenderer
# from utils                        import read_video, save_video
from heatmap_analyzer             import HeatmapAnalyzer  # Import HeatmapAnalyzer
from zone_analyzer                import ZoneAnalyzer     # Import ZoneAnalyzer

# ── Konstanta path model ──────────────────────────────────────────────────────
RFDETR_WEIGHTS   = str(BASE_DIR / "models" / "RFDETR Result Dataset With Augmentation Version 2" / "checkpoint_best_total.pth")
BALL_YOLO_WEIGHTS= str(BASE_DIR / "models" / "best_yolo8s_ball-detection.pt")
KP_MODEL_PATH    = str(BASE_DIR / "models" / "kpdet_best_final.pt")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

# ── MySQL config — sesuaikan dengan environment kamu ──────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "3306")),
    "user":     os.getenv("DB_USER",     "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME",     "football_detection_final"),
    "charset":  "utf8mb4",
}


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db() -> mysql.connector.MySQLConnection:
    """Buka koneksi DB per-request (simpan di Flask g)."""
    if "db" not in g:
        g.db = mysql.connector.connect(**DB_CONFIG)
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None and db.is_connected():
        db.close()


def query(sql: str, params: tuple = (), *, fetch: str = "none"):
    """
    Jalankan SQL.
    fetch = 'one' | 'all' | 'none'
    Return: row(s) atau lastrowid.
    """
    db  = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute(sql, params)
    if fetch == "one":
        row = cur.fetchone()
        cur.close()
        return row
    if fetch == "all":
        rows = cur.fetchall()
        cur.close()
        return rows
    db.commit()
    lid = cur.lastrowid
    cur.close()
    return lid


# ─────────────────────────────────────────────────────────────────────────────
# Status helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_status(video_id: int, status: str, progress: int = 0, error_msg: str = ""):
    """Update status & progress di tabel Videos."""
    # Gunakan koneksi baru karena dipanggil dari background thread
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute(
            "UPDATE Videos SET status=%s, progress_pct=%s, error_msg=%s, updated_at=NOW() "
            "WHERE video_id=%s",
            (status, progress, error_msg, video_id),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB] _set_status error: {e}")


def _save_analysis(
    video_id:             int,
    processing_time_sec:  float,
    annotated_path:       str,
    heatmap_home_path:    str,
    heatmap_away_path:    str,
    zone_grid_path:       str,
    zone_voronoi_path:    str,
    possession_home:      float,
    possession_away:      float,
    ball_speed_max:       float,
    ball_speed_avg:       float,
    player_speed_max:     float,
    total_frames:         int,
    metadata_dict:        dict,
) -> int:
    """Simpan AnalysisResults + AnalysisMetadata. Return analysis_id."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor()

        cur.execute(
            """INSERT INTO AnalysisResults
               (video_id, model_name, tracking_method, processing_time_sec,
                anotated_video_path, heatmap_home_path, heatmap_away_path,
                zone_grid_path, zone_voronoi_path,
                analysis_date,
                ball_posession_home, ball_posession_away,
                ball_speed_max_kmh, ball_speed_avg_kmh,
                player_speed_max_kmh, total_frames)
               VALUES (%s,'RF-DETR Base','ByteTrack',%s,%s,%s,%s,%s,%s,NOW(),%s,%s,%s,%s,%s,%s)""",
            (
                video_id, processing_time_sec,
                annotated_path, heatmap_home_path, heatmap_away_path,
                zone_grid_path, zone_voronoi_path,
                possession_home, possession_away,
                ball_speed_max, ball_speed_avg,
                player_speed_max, total_frames,
            ),
        )
        analysis_id = cur.lastrowid

        cur.execute(
            "INSERT INTO AnalysisMetadata (analysis_id, analysis_data) VALUES (%s, %s)",
            (analysis_id, json.dumps(metadata_dict, ensure_ascii=False)),
        )
        conn.commit()
        cur.close()
        conn.close()
        return analysis_id
    except Exception as e:
        print(f"[DB] _save_analysis error: {e}")
        return -1


# ─────────────────────────────────────────────────────────────────────────────
# Video I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_video(path: str) -> list:
    cap    = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(frames: list, out_path: str, fps: float = 24.0):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def get_video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = int(frames / fps) if fps else 0
    return {
        "fps":          round(fps, 2),
        "total_frames": frames,
        "duration_sec": duration,
        "resolution":   f"{w}x{h}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis pipeline (dijalankan di background thread)
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(video_id: int, video_path: str, out_dir: Path):
    """
    Full analysis pipeline:
      1. Read video
      2. Detect & Track (RF-DETR + ByteTrack + Ball YOLO)
      3. Interpolate ball
      4. Team assignment (color-based)
      5. Camera movement estimation
      6. Speed & distance estimation
      7. Keypoint detection
      8. Homography computation
      9. Ball possession
     10. Draw annotations + tactical map
     11. Heatmap & Zone analysis
     12. Save outputs → DB
    """
    t_start = time.time()
    _set_status(video_id, "processing", 5)

    try:
        # ── 1. Read video ────────────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Reading video…")
        frames = read_video(video_path)
        if not frames:
            raise ValueError("Tidak ada frame terbaca dari video.")

        n_frames = len(frames)
        video_meta = get_video_meta(video_path)
        fps = video_meta["fps"]

        # ── Update meta ke Videos table ─────────────────────────────────────
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cur  = conn.cursor()
            cur.execute(
                "UPDATE Videos SET fps=%s, duration_sec=%s, resolution=%s WHERE video_id=%s",
                (fps, video_meta["duration_sec"], video_meta["resolution"], video_id),
            )
            conn.commit(); cur.close(); conn.close()
        except Exception as e:
            print(f"[DB] meta update error: {e}")

        _set_status(video_id, "processing", 10)

        # ── 2. Tracker ───────────────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Detecting & Tracking…")
        tracker = Tracker(
            model_path      = RFDETR_WEIGHTS,
            ball_model_path = BALL_YOLO_WEIGHTS if os.path.exists(BALL_YOLO_WEIGHTS) else None,
        )

        stub_tracks = str(STUB_DIR / f"{video_id}_tracks.pkl")
        tracks = tracker.get_object_tracks(
            frames,
            read_from_stub = os.path.exists(stub_tracks),
            stub_path      = stub_tracks,
        )
        _set_status(video_id, "processing", 25)

        # ── 3. Interpolate ball ──────────────────────────────────────────────
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # ── 4. Team assignment ───────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Assigning teams…")
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(frames, tracks["player"])

        all_players_frames = tracks["player"]
        for frame_num, player_track in enumerate(all_players_frames):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    frames[frame_num], track["bbox"], player_id
                )
                tracks["player"][frame_num][player_id]["team"]       = team
                tracks["player"][frame_num][player_id]["team_color"]  = team_assigner.team_colors.get(team, (0, 0, 255))

        # Goalkeeper team assignment
        for frame_num, gk_track in enumerate(tracks["goalkeeper"]):
            for gk_id, track in gk_track.items():
                team = team_assigner.get_player_team(
                    frames[frame_num], track["bbox"], gk_id, is_goalkeeper=True
                )
                tracks["goalkeeper"][frame_num][gk_id]["team"]       = team
                tracks["goalkeeper"][frame_num][gk_id]["team_color"]  = team_assigner.team_colors.get(team, (0, 255, 255))

        _set_status(video_id, "processing", 35)

        # ── 5. Camera movement ───────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Estimating camera movement…")
        cam_est = CameraMovementEstimator(frames[0])
        stub_cam = str(STUB_DIR / f"{video_id}_camera.pkl")
        camera_movement = cam_est.get_camera_movement(
            frames,
            read_from_stub = os.path.exists(stub_cam),
            stub_path      = stub_cam,
        )
        _set_status(video_id, "processing", 42)

        # ── 6. Positions ─────────────────────────────────────────────────────
        tracker.add_object_positions(tracks)
        cam_est.add_adjust_positions_to_tracks(tracks, camera_movement)

        # ── 7. Speed & distance ──────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Computing speed & distance…")
        speed_est = SpeedAndDistance_Estimator()
        speed_est.frame_rate = fps
        speed_est.add_speed_and_distance_to_tracks(tracks)
        _set_status(video_id, "processing", 50)

        # ── 8. Keypoint detection ────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Detecting pitch keypoints…")
        kp_detector = KeypointDetector(KP_MODEL_PATH, confidence_threshold=0.5)
        print("Detecting pitch keypoints...")
        all_kp = kp_detector.detect_keypoints_batch(frames)
        _set_status(video_id, "processing", 62)

        # ── 9. Homography ────────────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Computing homography…")
        homography_calc = HomographyCalculator(
            min_keypoints       = 4,
            canvas_w            = 960,
            canvas_h            = 560,
            max_reproj_error_cm = 500.0,
        )

        # ── 10. Ball possession ──────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Computing ball possession…")
        pba  = PlayerBallAssigner()
        tbc  = TeamBallControl()
        import numpy as np

        for frame_num, ball_bbox_dict in enumerate(tracks["ball"]):
            ball_bbox = ball_bbox_dict.get(1, {}).get("bbox")
            if ball_bbox:
                all_players = {**tracks["player"][frame_num], **tracks["goalkeeper"][frame_num]}
                assigned = pba.assign_ball_to_player(all_players, ball_bbox)
                if assigned != -1:
                    pid  = assigned
                    pdict = all_players.get(pid, {})
                    tbc.update(assigned, pdict if "team" in pdict else
                               tracks["player"][frame_num].get(pid,
                               tracks["goalkeeper"][frame_num].get(pid, {})))
                    # Mark has_ball
                    if pid in tracks["player"][frame_num]:
                        tracks["player"][frame_num][pid]["has_ball"] = True
                    elif pid in tracks["goalkeeper"][frame_num]:
                        tracks["goalkeeper"][frame_num][pid]["has_ball"] = True
                else:
                    tbc.update(-1, {})
            else:
                tbc.update(-1, {})

        ball_control_seq = np.array(tbc.get_sequence(), dtype=float)
        # Hitung possession %
        valid = ball_control_seq[ball_control_seq > 0]
        if len(valid) > 0:
            poss_home = float((valid == 1).sum() / len(valid) * 100)
            poss_away = float((valid == 2).sum() / len(valid) * 100)
        else:
            poss_home, poss_away = 50.0, 50.0

        _set_status(video_id, "processing", 70)

        # ── 11. Speed stats ──────────────────────────────────────────────────
        all_speeds_t1, all_speeds_t2 = [], []
        all_ball_speeds = []

        for frame_num in range(n_frames):
            for pid, pinfo in tracks["player"][frame_num].items():
                sp = pinfo.get("speed")
                if sp is None:
                    continue
                team = pinfo.get("team", 0)
                if team == 1:
                    all_speeds_t1.append(sp)
                elif team == 2:
                    all_speeds_t2.append(sp)

        ball_speed_max = 0.0
        ball_speed_avg = 0.0
        player_speed_max = 0.0
        avg_speed_t1 = 0.0
        avg_speed_t2 = 0.0

        if all_speeds_t1:
            avg_speed_t1    = round(float(np.mean(all_speeds_t1)), 2)
        if all_speeds_t2:
            avg_speed_t2    = round(float(np.mean(all_speeds_t2)), 2)
        if all_speeds_t1 or all_speeds_t2:
            player_speed_max = round(float(max(
                (max(all_speeds_t1) if all_speeds_t1 else 0),
                (max(all_speeds_t2) if all_speeds_t2 else 0),
            )), 2)

        # ── 12. Draw annotations ─────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Drawing annotations…")
        output_frames = tracker.draw_annotations(frames, tracks, ball_control_seq)
        _set_status(video_id, "processing", 78)

        # ── 13. Tactical map + Heatmap + Zone ────────────────────────────────
        print(f"[Pipeline] [{video_id}] Rendering tactical maps…")
        tac_renderer = TacticalMapRenderer(canvas_w=960, canvas_h=560)
        heatmap_ana  = HeatmapAnalyzer(canvas_w=960, canvas_h=560)
        zone_ana     = ZoneAnalyzer(canvas_w=960, canvas_h=560)

        ball_trail: list = []
        tac_frames: list = []

        for frame_num in range(n_frames):
            kp = all_kp[frame_num]
            H  = homography_calc.get_homography(kp)

            # Player positions for tactical map
            player_map_pos = {}
            for pid, pinfo in tracks["player"][frame_num].items():
                pos = pinfo.get("position_adjusted") or pinfo.get("position")
                if pos is None:
                    continue
                mp = homography_calc.transform_point(pos, H, canvas_w=960, canvas_h=560)
                player_map_pos[pid] = {
                    "map_pos": mp,
                    "team":    pinfo.get("team", -1),
                    "role":    "player",
                }
            for gid, ginfo in tracks["goalkeeper"][frame_num].items():
                pos = ginfo.get("position_adjusted") or ginfo.get("position")
                if pos is None:
                    continue
                mp = homography_calc.transform_point(pos, H, canvas_w=960, canvas_h=560)
                player_map_pos[gid] = {
                    "map_pos": mp,
                    "team":    ginfo.get("team", -1),
                    "role":    "goalkeeper",
                }

            # Ball
            ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")
            ball_map  = None
            if ball_bbox:
                from utils import get_center_of_bbox
                bpos = tracks["ball"][frame_num].get(1, {}).get("position") or \
                       (get_center_of_bbox(ball_bbox) if ball_bbox else None)
                if bpos:
                    ball_map = homography_calc.transform_point(bpos, H, canvas_w=960, canvas_h=560)
            ball_trail.append(ball_map)

            # Render tactical canvas
            tac_canvas = tac_renderer.render(
                player_positions = player_map_pos,
                ball_position    = ball_map,
                ball_trail       = ball_trail,
            )

            # Overlay tactical map (bottom-right corner) pada annotation frame
            ann_frame = output_frames[frame_num].copy()
            fh, fw    = ann_frame.shape[:2]
            th, tw    = tac_canvas.shape[:2]
            # Scale tactical map to 30% of frame width
            scale = fw * 0.30 / tw
            nt, nw = int(th * scale), int(tw * scale)
            small  = cv2.resize(tac_canvas, (nw, nt))
            margin = 10
            y1, y2 = fh - nt - margin, fh - margin
            x1, x2 = fw - nw - margin, fw - margin
            # alpha blend
            roi = ann_frame[y1:y2, x1:x2]
            blended = cv2.addWeighted(small, 0.85, roi, 0.15, 0)
            ann_frame[y1:y2, x1:x2] = blended
            output_frames[frame_num] = ann_frame

            # Collect for heatmap & zone
            tbc_arr = ball_control_seq[:frame_num+1] if frame_num < len(ball_control_seq) else ball_control_seq
            heatmap_ana.collect(frame_num, tracks, homography_calc, kp, tbc_arr)
            zone_ana.collect(frame_num, tracks, homography_calc, kp, ball_map)

        _set_status(video_id, "processing", 87)

        # ── 14. Save output video ─────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Saving output video…")
        out_video_path = str(out_dir / "annotated.avi")
        save_video(output_frames, out_video_path, fps)

        # ── 15. Save heatmaps ─────────────────────────────────────────────────
        print(f"[Pipeline] [{video_id}] Saving heatmaps…")
        hm_paths  = heatmap_ana.save_all(output_dir=str(out_dir), video_name=str(video_id))
        zon_paths = zone_ana.save_all(output_dir=str(out_dir), video_name=str(video_id))

        # Rename heatmap outputs ke nama yang dipakai index.html
        heatmap_home_path = ""
        heatmap_away_path = ""
        for key, src in hm_paths.items():
            if "team1" in key and src.endswith((".png", ".jpg")):
                dst = str(out_dir / "heatmap_home.jpg")
                os.replace(src, dst)
                heatmap_home_path = dst
            elif "team2" in key and src.endswith((".png", ".jpg")):
                dst = str(out_dir / "heatmap_away.jpg")
                os.replace(src, dst)
                heatmap_away_path = dst

        zone_grid_path    = zon_paths.get("zone_grid",    "")
        zone_voronoi_path = zon_paths.get("zone_voronoi", "")

        _set_status(video_id, "processing", 93)

        # ── 16. Build metadata dict ───────────────────────────────────────────
        metadata = {
            "avg_speed_team1":  avg_speed_t1,
            "avg_speed_team2":  avg_speed_t2,
            "heatmap_home_path": heatmap_home_path,
            "heatmap_away_path": heatmap_away_path,
            "zone_grid_path":    zone_grid_path,
            "zone_voronoi_path": zone_voronoi_path,
            "heatmap_stats":     heatmap_ana.get_summary_stats(),
            "zone_stats":        zon_paths.get("zone_stats", ""),
            "ball_speed": {
                "max_kmh": ball_speed_max,
                "avg_kmh": ball_speed_avg,
            },
        }

        # ── 17. Simpan ke DB ──────────────────────────────────────────────────
        proc_time = round(time.time() - t_start, 1)
        _save_analysis(
            video_id             = video_id,
            processing_time_sec  = proc_time,
            annotated_path       = out_video_path,
            heatmap_home_path    = heatmap_home_path,
            heatmap_away_path    = heatmap_away_path,
            zone_grid_path       = zone_grid_path,
            zone_voronoi_path    = zone_voronoi_path,
            possession_home      = round(poss_home, 1),
            possession_away      = round(poss_away, 1),
            ball_speed_max       = ball_speed_max,
            ball_speed_avg       = ball_speed_avg,
            player_speed_max     = player_speed_max,
            total_frames         = n_frames,
            metadata_dict        = metadata,
        )

        _set_status(video_id, "done", 100)
        print(f"[Pipeline] [{video_id}] DONE in {proc_time}s")

    except Exception:
        err = traceback.format_exc()
        print(f"[Pipeline] [{video_id}] ERROR:\n{err}")
        _set_status(video_id, "error", 0, err[-500:])


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Upload video + simpan ke DB + jalankan analisis di background."""
    if "video" not in request.files:
        return jsonify({"success": False, "message": "Tidak ada file video."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"success": False, "message": "Nama file kosong."}), 400

    # ── Validasi ekstensi ──────────────────────────────────────────────────
    ALLOWED = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"success": False, "message": f"Format tidak didukung: {ext}"}), 400

    # ── Ambil form data ────────────────────────────────────────────────────
    video_name  = request.form.get("video_name", file.filename)
    team1       = request.form.get("team1", "Tim 1")
    team2       = request.form.get("team2", "Tim 2")
    t1_sec      = request.form.get("team1_color_secondary",  "#E53E3E")
    t1_gk       = request.form.get("team1_goalkeeper_color", "#F6AD55")
    t2_sec      = request.form.get("team2_color_secondary",  "#3182CE")
    t2_gk       = request.form.get("team2_goalkeeper_color", "#48BB78")
    ref_col     = request.form.get("referee_color",          "#ECC94B")

    # ── Insert Videos row (status=queued) ──────────────────────────────────
    video_id = query(
        """INSERT INTO Videos
           (video_name, team1_name, team1_color_secondary, team1_goalkeeper_color,
            team2_name, team2_color_secondary, team2_goalkeeper_color,
            referee_color, status, progress_pct)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'queued',0)""",
        (video_name, team1, t1_sec, t1_gk, team2, t2_sec, t2_gk, ref_col),
    )

    # ── Simpan file ────────────────────────────────────────────────────────
    vid_upload_dir = UPLOAD_DIR / str(video_id)
    vid_upload_dir.mkdir(parents=True, exist_ok=True)
    raw_path = str(vid_upload_dir / f"raw{ext}")
    file.save(raw_path)

    # ── Update file info di DB ─────────────────────────────────────────────
    size_mb = round(os.path.getsize(raw_path) / 1024 / 1024, 2)
    query(
        "UPDATE Videos SET file_path_raw=%s, file_size_mb=%s WHERE video_id=%s",
        (raw_path, size_mb, video_id),
    )

    # ── Buat output dir ────────────────────────────────────────────────────
    out_dir = OUTPUT_DIR / str(video_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Jalankan analisis di background ────────────────────────────────────
    t = threading.Thread(
        target=run_analysis,
        args=(video_id, raw_path, out_dir),
        daemon=True,
    )
    t.start()

    return jsonify({"success": True, "video_id": video_id, "message": "Upload berhasil, analisis dimulai."})


@app.route("/api/videos", methods=["GET"])
def list_videos():
    """Daftar video dengan pagination, search & filter status."""
    page   = max(1, int(request.args.get("page",  1)))
    limit  = max(1, min(50, int(request.args.get("limit", 12))))
    q      = request.args.get("q",      "").strip()
    status = request.args.get("status", "").strip()
    offset = (page - 1) * limit

    where, params = [], []
    if q:
        where.append("(video_name LIKE %s OR team1_name LIKE %s OR team2_name LIKE %s)")
        like = f"%{q}%"
        params += [like, like, like]
    if status:
        where.append("status = %s")
        params.append(status)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    total = query(f"SELECT COUNT(*) AS cnt FROM Videos {where_sql}", tuple(params), fetch="one")["cnt"]
    rows  = query(
        f"SELECT video_id, video_name, team1_name, team2_name, status, "
        f"progress_pct, duration_sec, fps, resolution, created_at "
        f"FROM Videos {where_sql} ORDER BY created_at DESC LIMIT %s OFFSET %s",
        tuple(params) + (limit, offset),
        fetch="all",
    )

    # Serialize datetime
    for r in rows:
        if r.get("created_at"):
            r["created_at"] = str(r["created_at"])

    return jsonify({"total": total, "page": page, "limit": limit, "videos": rows})


@app.route("/api/video/<int:video_id>", methods=["GET"])
def get_video(video_id: int):
    """Detail satu video + hasil analisis."""
    v = query(
        "SELECT * FROM Videos WHERE video_id=%s", (video_id,), fetch="one"
    )
    if not v:
        return jsonify({"success": False, "message": "Video tidak ditemukan."}), 404

    # Serialize
    for k in ("created_at", "updated_at"):
        if v.get(k):
            v[k] = str(v[k])

    # Ambil AnalysisResults
    res = query(
        "SELECT * FROM AnalysisResults WHERE video_id=%s ORDER BY analysis_id DESC LIMIT 1",
        (video_id,), fetch="one",
    )
    if res:
        for k in ("analysis_date", "created_at"):
            if res.get(k):
                res[k] = str(res[k])

        # Ambil metadata JSON
        meta_row = query(
            "SELECT analysis_data FROM AnalysisMetadata WHERE analysis_id=%s",
            (res["analysis_id"],), fetch="one",
        )
        if meta_row and meta_row.get("analysis_data"):
            try:
                res["metadata"] = json.loads(meta_row["analysis_data"])
            except Exception:
                res["metadata"] = {}
        else:
            res["metadata"] = {}

        v["result"] = res
    else:
        v["result"] = None

    return jsonify(v)


@app.route("/api/video/<int:video_id>", methods=["PUT"])
def update_video(video_id: int):
    """Update metadata video (nama, tim, dll)."""
    data = request.get_json(silent=True) or {}
    allowed = ["video_name", "team1_name", "team2_name"]
    sets, params = [], []
    for col in allowed:
        if col in data:
            sets.append(f"{col}=%s")
            params.append(data[col])
    if not sets:
        return jsonify({"success": False, "message": "Tidak ada field yang diupdate."}), 400

    params.append(video_id)
    query(f"UPDATE Videos SET {', '.join(sets)}, updated_at=NOW() WHERE video_id=%s", tuple(params))
    return jsonify({"success": True, "message": "Data diperbarui."})


@app.route("/api/video/<int:video_id>", methods=["DELETE"])
def delete_video(video_id: int):
    """Hapus video + file output."""
    v = query("SELECT file_path_raw FROM Videos WHERE video_id=%s", (video_id,), fetch="one")
    if not v:
        return jsonify({"success": False, "message": "Video tidak ditemukan."}), 404

    # Hapus dari DB (cascade hapus AnalysisResults & Metadata)
    query("DELETE FROM Videos WHERE video_id=%s", (video_id,))

    # Hapus file
    import shutil
    for d in (UPLOAD_DIR / str(video_id), OUTPUT_DIR / str(video_id)):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    return jsonify({"success": True, "message": "Video dihapus."})


# ─────────────────────────────────────────────────────────────────────────────
# Serve output files (video, heatmap, zone)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/outputs/<int:video_id>/<path:filename>")
def serve_output(video_id: int, filename: str):
    """Serve file hasil analisis dari folder outputs/<video_id>/."""
    out_dir = OUTPUT_DIR / str(video_id)
    return send_from_directory(str(out_dir), filename)


# ─────────────────────────────────────────────────────────────────────────────
# Serve index.html (SPA)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/<path:path>")
def serve_spa(path=""):
    index = BASE_DIR / "index.html"
    if index.exists():
        return index.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}
    return "index.html not found", 404


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Football Detection — Flask Backend")
    print("  Calvin Institute of Technology")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)