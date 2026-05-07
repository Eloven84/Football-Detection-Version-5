from __future__ import annotations
"""
main.py
Pipeline utama: Video → Tracking → Homography → Tactical Map → Output Video

Modul yang dipakai:
  trackers                  — deteksi & tracking pemain/bola (RFDETR + YOLOv8)
  team_assigner             — klasifikasi warna tim
  team_ball_control         — hitung penguasaan bola per tim
  player_ball_assigner      — tentukan pemain mana yang pegang bola
  camera_movement_estimator — kompensasi pergerakan kamera
  speed_and_distance_estimator
  keypoint_detector         — deteksi 32 KP lapangan (Soccana fine-tuned)
  homography                — hitung H frame→pitch (unit cm), HomographyTracker
  tactical_map              — render canvas taktis + smooth ball trail
  utils                     — read_video, save_video
"""

"""
main.py
Pipeline utama: Video → Tracking → Homography → Tactical Map → Output Video
"""

# ═══════════════════════════════════════════════════════════════════
# INTEGRASI HeatmapAnalyzer ke main.py
# Tambahkan bagian-bagian ini ke main.py yang sudah ada
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# INTEGRASI ZoneAnalyzer ke main.py
# ═══════════════════════════════════════════════════════════════════

import os
import cv2
import json
import numpy as np

from trackers                     import Tracker
from team_assigner                import TeamAssigner
from team_ball_control            import TeamBallControl
from player_ball_assigner         import PlayerBallAssigner
from camera_movement_estimator    import CameraMovementEstimator
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from keypoint_detector            import KeypointDetector
from homography                   import HomographyCalculator
from tactical_map                 import TacticalMapRenderer
from utils                        import read_video, save_video
from heatmap_analyzer             import HeatmapAnalyzer  # Import HeatmapAnalyzer
from zone_analyzer                import ZoneAnalyzer     # Import ZoneAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi path
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_PATH     = "input_video/input_1.mp4"
TRACKER_MODEL  = "models/RFDETR Result Dataset With Augmentation Version 2/checkpoint_best_total.pth"
BALL_MODEL     = "models/best_yolo8s_ball-detection.pt"
KP_MODEL       = "models/kpdet_best_final.pt"
STUB_DIR       = "stubs"
OUTPUT_DIR     = "output_videos"


def _stub(video_path: str, suffix: str) -> str:
    base = os.path.basename(video_path).replace(".mp4", "")
    return os.path.join(STUB_DIR, f"{base}_{suffix}.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(STUB_DIR,   exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Baca video ────────────────────────────────────────────────────────
    video_frames = read_video(VIDEO_PATH)
    frame_h, frame_w = video_frames[0].shape[:2]
    print(f"[main] Total frames: {len(video_frames)}")

    # ── 2. Object tracking ──────────────────────────────────────────────────
    tracker = Tracker(model_path=TRACKER_MODEL, ball_model_path=BALL_MODEL)
    tracks  = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path=_stub(VIDEO_PATH, "track"),
    )
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    tracker.add_object_positions(tracks)

    # ── 3. Camera movement ───────────────────────────────────────────────────
    cam_estimator = CameraMovementEstimator(video_frames[0])
    cam_movement  = cam_estimator.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path=_stub(VIDEO_PATH, "cammove"),
    )
    cam_estimator.add_adjust_positions_to_tracks(tracks, cam_movement)

    # ── 4. Team assignment ───────────────────────────────────────────────────
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks["player"], sample_frames=10)

    for frame_num, player_track in enumerate(tracks["player"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id, is_goalkeeper=False
            )
            tracks["player"][frame_num][player_id]["team"]       = team
            tracks["player"][frame_num][player_id]["team_color"] = \
                team_assigner.team_colors.get(team, (200, 200, 200))

    for frame_num, gk_track in enumerate(tracks["goalkeeper"]):
        for gk_id, track in gk_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], gk_id, is_goalkeeper=True
            )
            tracks["goalkeeper"][frame_num][gk_id]["team"]       = team
            tracks["goalkeeper"][frame_num][gk_id]["team_color"] = \
                team_assigner.team_colors.get(team, (200, 200, 200))

    print(f"[TeamAssigner] Unique players assigned: {len(team_assigner.player_team_dict)}")

    # ── 5. Speed & Distance ──────────────────────────────────────────────────
    speed_est = SpeedAndDistance_Estimator()
    speed_est.add_speed_and_distance_to_tracks(tracks)

    # ── 6. Ball possession ───────────────────────────────────────────────────
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

    # ── 7. Keypoint detection ────────────────────────────────────────────────
    print("[main] Detecting pitch keypoints...")

    kp_detector = KeypointDetector(model_path=KP_MODEL, confidence_threshold=0.5)
    raw_kp      = kp_detector.detect_batch(video_frames)

    # Filter: buang KP yang koordinatnya di luar batas frame
    all_kp = [
        {kid: (x, y) for kid, (x, y) in kp.items()
         if 0 <= x < frame_w and 0 <= y < frame_h}
        for kp in raw_kp
    ]

    # ── 7.5. Ball trajectory — pre-pass + smooth ─────────────────────────────
    print("[main] Computing ball trajectory...")

    renderer= TacticalMapRenderer()

    # ── Inisialisasi (setelah TacticalMapRenderer dibuat) ────────────
    heatmap_analyzer = HeatmapAnalyzer(
        canvas_w=960,        # sama dengan TacticalMapRenderer
        canvas_h=560,
        gaussian_radius=25,  # naikkan untuk heatmap lebih smooth
        output_scale=2.0,    # output 2× lebih besar dari canvas (1920×1120)
    )

    # ── Inisialisasi — letakkan setelah renderer = TacticalMapRenderer() ──
    zone_analyzer = ZoneAnalyzer(
        canvas_w=960,
        canvas_h=560,
        grid_cols=6,      # lapangan dibagi 6×4 = 24 zona
        grid_rows=4,
        output_scale=2.0,
        player_weight=1.0,
        ball_weight=2.0,  # bola diberi bobot 2× lebih tinggi dari pemain
    )

    # Kalau kamu mau analisis per babak, definisikan batas frame-nya di sini.
    # Sesuaikan dengan video kamu (misal: 750 frame total, babak 1 = 0–374).
    # BABAK_1_END   = 374   # frame terakhir babak 1
    # BABAK_2_START = 375   # frame pertama babak 2

    homography_calc = HomographyCalculator(min_keypoints=4)

    # Pass 1: kumpulkan posisi bola mentah di canvas
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
                # Terima hanya jika benar-benar di dalam canvas
                if candidate is not None:
                    cx, cy = candidate
                    if 0 <= cx <= renderer.canvas_w and 0 <= cy <= renderer.canvas_h:
                        pos = candidate

        raw_ball_pos.append(pos)

    # Smooth: filter lompatan ekstrem lalu median filter
    MAX_JUMP_PX = 120 # Maksimal pergerakan bola antar frame (dalam pixel canvas) yang dianggap valid. Naik 50 -> 120
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
        wx = [xs[j] for j in range(max(0, i - half), min(len(xs), i + half + 1)) if xs[j] is not None]
        wy = [ys[j] for j in range(max(0, i - half), min(len(ys), i + half + 1)) if ys[j] is not None]
        if wx and wy:
            ball_trail_smooth.append((int(np.median(wx)), int(np.median(wy))))
        else:
            ball_trail_smooth.append(None)

    valid_ball = sum(1 for p in ball_trail_smooth if p is not None)
    print(f"[main] Ball trajectory: {valid_ball} / {len(ball_trail_smooth)} valid frames")

    # ── 8. Build tactical frames ─────────────────────────────────────────────
    print("[main] Rendering tactical map...")
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
        # ── Collect per frame (di dalam loop build tactical frames) ───────\
        heatmap_analyzer.collect(
            frame_num=frame_num,
            tracks=tracks,
            homography_calc=homography_calc,
            keypoints=kp,
            team_ball_control=team_ball_control,
        )
        # ── Collect — letakkan di dalam loop tactical frames (step 8) ─
        zone_analyzer.collect(
            frame_num=frame_num,
            tracks=tracks,
            homography_calc=homography_calc,
            keypoints=kp,
            ball_map_pos=ball_trail_smooth[frame_num],   # posisi bola smooth
        )
        tactical_frames.append(tac_frame)

    print(f"[HomographyCalculator] H updates: {homography_calc.update_count}")

    # ── Save semua output setelah loop selesai ────────────────────────
    video_name = os.path.basename(VIDEO_PATH).replace(".mp4", "")
    
    saved_files = heatmap_analyzer.save_all(
        output_dir="output_analysis",
        video_name=video_name,
        fmt="png",       # ganti ke "jpg" kalau mau file lebih kecil
    )

    # ── Save — letakkan setelah loop tactical frames selesai ──────
    video_name = os.path.basename(VIDEO_PATH).replace(".mp4", "")
    
    # Keseluruhan video
    zone_files = zone_analyzer.save_all(
        output_dir="output_analysis",
        video_name=video_name,
        fmt="png",
    )
    
    # # Per babak (opsional — hapus kalau tidak butuh)
    # zone_analyzer.save_segment(
    #     output_dir="output_analysis",
    #     segment_name=f"{video_name}_babak1",
    #     frame_start=0,
    #     frame_end=BABAK_1_END,
    #     fmt="png",
    # )
    # zone_analyzer.save_segment(
    #     output_dir="output_analysis",
    #     segment_name=f"{video_name}_babak2",
    #     frame_start=BABAK_2_START,
    #     frame_end=None,   # sampai frame terakhir
    #     fmt="png",
    # )

    # ── 9. Annotasi video utama ──────────────────────────────────────────────
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_frames = cam_estimator.draw_camera_movement(output_frames, cam_movement)
    speed_est.draw_speed_and_distance(output_frames, tracks)

    # ── 10. Side-by-side: video utama | tactical map ─────────────────────────
    tac_h   = frame_h
    tac_w   = int(tac_h * renderer.canvas_w / renderer.canvas_h)
    divider = np.full((frame_h, 4, 3), 180, dtype=np.uint8)

    combined_frames = [
        np.hstack([
            output_frames[i],
            divider,
            cv2.resize(tactical_frames[i], (tac_w, tac_h)),
        ])
        for i in range(len(output_frames))
    ]

    # ── 11. Simpan output ────────────────────────────────────────────────────
    output_name = os.path.basename(VIDEO_PATH).replace(".mp4", "_test_output.mp4")
    output_path = os.path.join(OUTPUT_DIR, output_name)
    save_video(combined_frames, output_path)
    print(f"[main] Saved → {output_path}")

    # Print summary stats ke console
    stats = heatmap_analyzer.get_summary_stats()
    print("\n" + "="*50)
    print("SUMMARY ANALISIS")
    print("="*50)
    print(f"Total frame dianalisis : {stats['total_frames_analyzed']}")
    print(f"Ball Possession        : Tim 1 {stats['ball_possession']['team1_pct']}%"
        f" | Tim 2 {stats['ball_possession']['team2_pct']}%")
    print(f"Zone Coverage          : Tim 1 {stats['zone_coverage']['team1_pct']}%"
        f" | Tim 2 {stats['zone_coverage']['team2_pct']}%")
    print(f"Data points posisi     : Tim 1 {stats['position_datapoints']['team1']}"
        f" | Tim 2 {stats['position_datapoints']['team2']}")
    print("="*50)
    print(f"Output tersimpan di  : output_analysis/{video_name}_*.png")

    # ── 5. Print stats zona ke console ───────────────────────────────
    with open(zone_files["zone_stats"]) as f:
        zstats = json.load(f)
    
    print("\n" + "="*50)
    print("ZONE CONTROL SUMMARY")
    print("="*50)
    print(f"Pixel dominance  : Tim 1 {zstats['pixel_dominance']['team1_pct']}%"
        f" | Tim 2 {zstats['pixel_dominance']['team2_pct']}%")
    print(f"Grid zona dikuasai: Tim 1 {zstats['grid_zone_dominance']['team1_zones']}"
        f" zona ({zstats['grid_zone_dominance']['team1_pct']}%)"
        f" | Tim 2 {zstats['grid_zone_dominance']['team2_zones']}"
        f" zona ({zstats['grid_zone_dominance']['team2_pct']}%)")
    print("="*50)


if __name__ == "__main__":
    main()