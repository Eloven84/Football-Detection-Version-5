import os
import cv2
import numpy as np
import tensorflow as tf
from trackers import Tracker
from team_assigner import TeamAssigner
from team_ball_control import TeamBallControl
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from keypoint_detector import KeypointDetector
from homography import HomographyCalculator
from tactical_map import TacticalMapRenderer
from utils import read_video, save_video

def is_keypoints_spread_enough(keypoints, min_spread_px=100):
    """Cek apakah keypoints tersebar cukup untuk homography yang akurat."""
    if len(keypoints) < 4:
        return False
    pts = list(keypoints.values())
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    return spread_x > min_spread_px and spread_y > min_spread_px

def is_valid_map_pos(pos, canvas_w, canvas_h, margin=10):
    """Pastikan titik hasil transform ada di dalam canvas."""
    if pos is None:
        return False
    x, y = pos
    return (-margin < x < canvas_w + margin and
            -margin < y < canvas_h + margin)

# def smooth_ball_trail(positions, window=9):
#         xs = [p[0] if p else None for p in positions]
#         ys = [p[1] if p else None for p in positions]
#         smoothed = []
#         for i in range(len(positions)):
#             half = window // 2
#             win_x = [xs[j] for j in range(max(0, i-half), min(len(xs), i+half+1)) if xs[j] is not None]
#             win_y = [ys[j] for j in range(max(0, i-half), min(len(ys), i+half+1)) if ys[j] is not None]
#             if win_x and win_y:
#                 smoothed.append((int(np.median(win_x)), int(np.median(win_y))))
#             else:
#                 smoothed.append(None)
#         return smoothed

def smooth_ball_trail(positions, window=9, max_jump_px=50):
    """Smooth + filter lompatan ekstrem."""
    # Filter lompatan ekstrem dulu sebelum smooth
    filtered = []
    prev = None
    for p in positions:
        if p is None:
            filtered.append(None)
            continue
        if prev is not None:
            dist = ((p[0]-prev[0])**2 + (p[1]-prev[1])**2) ** 0.5
            if dist > max_jump_px:
                filtered.append(None)  # anggap tidak valid, jangan sambung
                prev = None
                continue
        filtered.append(p)
        prev = p

    # Lanjut median smooth seperti sebelumnya
    xs = [p[0] if p else None for p in filtered]
    ys = [p[1] if p else None for p in filtered]
    smoothed = []
    for i in range(len(filtered)):
        half = window // 2
        win_x = [xs[j] for j in range(max(0, i-half), min(len(xs), i+half+1)) if xs[j] is not None]
        win_y = [ys[j] for j in range(max(0, i-half), min(len(ys), i+half+1)) if ys[j] is not None]
        if win_x and win_y:
            smoothed.append((int(np.median(win_x)), int(np.median(win_y))))
        else:
            smoothed.append(None)
    return smoothed

def main():
<<<<<<< HEAD
    video_path = 'input_video/input_1.mp4'
=======
    video_path = 'input_video/08fd33_4.mp4'
>>>>>>> bd1fd8f5b7f624cc10479119a1ab2a2c69be4c72

    # ── 1. Read Video ────────────────────────────────────────────────────────
    video_frames = read_video(video_path)
    print(f"Total frames: {len(video_frames)}")

    # ── 2. Object Tracking ───────────────────────────────────────────────────
    stub_name = os.path.basename(video_path).replace('.mp4', '_stub.pkl')
    stub_path = f"stubs/{stub_name}"

    tracker = Tracker(
        model_path=r'models/RFDETR Result Dataset With Augmentation Version 2/checkpoint_best_total.pth',
        ball_model_path=r'models/best_yolo8s_ball-detection.pt'
    )
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path=stub_path
    )
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    tracker.add_object_positions(tracks)

    # ── 3. Camera Movement ───────────────────────────────────────────────────
    stubs_name_cm = os.path.basename(video_path).replace('.mp4', '_cammove_stub.pkl')
    stub_path_cm  = f"stubs/{stubs_name_cm}"
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path=stub_path_cm
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # ── 4. Team Assignment ───────────────────────────────────────────────────
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks['player'], sample_frames=10)

    for frame_num, player_track in enumerate(tracks['player']):
        for player_id, track in player_track.items():
            # Hapus filter ini — biarkan semua player di-assign normal
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], player_id, is_goalkeeper=False
            )
            tracks['player'][frame_num][player_id]['team']       = team
            tracks['player'][frame_num][player_id]['team_color'] = \
                team_assigner.team_colors.get(team, (200, 200, 200))

    for frame_num, gk_track in enumerate(tracks['goalkeeper']):
        for gk_id, track in gk_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], gk_id, is_goalkeeper=True
            )
            tracks['goalkeeper'][frame_num][gk_id]['team']       = team
            tracks['goalkeeper'][frame_num][gk_id]['team_color'] = \
                team_assigner.team_colors.get(team, (200, 200, 200))

    print(f"[TeamAssigner] Total unique players assigned: {len(team_assigner.player_team_dict)}")

    # ── 5. Speed & Distance ──────────────────────────────────────────────────
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # ── 6. Ball Possession ───────────────────────────────────────────────────
    player_assigner = PlayerBallAssigner()
    team_control    = TeamBallControl()

    for frame_num, player_track in enumerate(tracks['player']):
        ball_frame = tracks['ball'][frame_num]
        if not ball_frame:
            team_control.update(-1, player_track)
            continue
        ball_bbox       = ball_frame[1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        team_control.update(assigned_player, tracks['player'][frame_num])
        if assigned_player != -1:
            tracks['player'][frame_num][assigned_player]['has_ball'] = True

    team_ball_control = np.array(team_control.get_sequence())

    # ── 7. Keypoint Detection & Homography ──────────────────────────────────
    keypoint_detector = KeypointDetector(
        model_path='models/soccana_keypoint/Model/weights/best.pt',
        confidence_threshold=0.5    # sesuaikan rekomendasi model
    )
    homography_calc = HomographyCalculator(
        min_keypoints=4,   
        common_kp_threshold=4,
        movement_threshold=2
    )
    tactical_renderer = TacticalMapRenderer()

    print("Detecting pitch keypoints...")
    all_keypoints = keypoint_detector.detect_keypoints_batch(video_frames)

    all_keypoints_dict = []
    frame_h, frame_w = video_frames[0].shape[:2]

    for kp_list in all_keypoints:
        if isinstance(kp_list, dict):
            kp_dict = kp_list
        else:
            kp_dict = {i: kp_list[i] for i in range(len(kp_list))}
        
        # Filter: hanya simpan keypoint yang koordinatnya ada di dalam frame
        filtered = {}
        for kid, (x, y) in kp_dict.items():
            if 0 <= x < frame_w and 0 <= y < frame_h:
                filtered[kid] = (x, y)
        
        all_keypoints_dict.append(filtered)

    # Debug: visualisasi keypoint frame 0
    debug_frame = video_frames[683].copy()
    for idx, (x, y) in all_keypoints_dict[683].items():
<<<<<<< HEAD
        xi, yi = int(x), int(y)
        cv2.circle(debug_frame, (xi, yi), 6, (0, 255, 0), -1)
        cv2.putText(debug_frame, str(idx), (xi + 5, yi - 5),
=======
        cv2.circle(debug_frame, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(debug_frame, str(idx), (x+5, y-5),
>>>>>>> bd1fd8f5b7f624cc10479119a1ab2a2c69be4c72
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("debug_keypoints_frame683.jpg", debug_frame)
    print("[DEBUG] Saved debug_keypoints_frame683.jpg")

    # ══════════════════════════════════════════════════════════════════════
    # ── 7.5 BARU: Kumpulkan semua ball map pos & smooth ──────────────────
    # ══════════════════════════════════════════════════════════════════════
    print("Computing ball trajectory...")

    # Pass pertama: hitung H dan kumpulkan semua posisi bola
    all_ball_map_pos = []
    homography_calc_pass1 = HomographyCalculator(
        min_keypoints=4,
        common_kp_threshold=4,
        movement_threshold=2
    )
    for frame_num in range(len(video_frames)):
        current_kp = all_keypoints_dict[frame_num]
        H = homography_calc_pass1.get_homography(current_kp)

        ball_frame = tracks['ball'][frame_num]
        ball_pos = None
        if ball_frame:
            bpos = ball_frame[1].get('position_adjusted') or ball_frame[1].get('position')
            if bpos:
                bp = homography_calc_pass1.transform_point(
                    bpos, H,
                    canvas_w=tactical_renderer.canvas_w,
                    canvas_h=tactical_renderer.canvas_h,
                    margin=0  # ← ketat, tidak boleh keluar sama sekali
                )
                if is_valid_map_pos(bp, tactical_renderer.canvas_w,
                                    tactical_renderer.canvas_h):
                    ball_pos = bp
        all_ball_map_pos.append(ball_pos)

    all_ball_map_pos_smooth = smooth_ball_trail(all_ball_map_pos, window=9)
    print(f"Ball trajectory computed: {sum(1 for p in all_ball_map_pos_smooth if p)} valid positions")

    # ── 8. Build Tactical Map Per Frame ─────────────────────────────────────
    tactical_frames = []

    for frame_num in range(len(video_frames)):
        current_kp = all_keypoints_dict[frame_num]
        H = homography_calc.get_homography(current_kp)  # ← pakai homography_calc utama

        player_positions = {}
        for player_id, track in tracks['player'][frame_num].items():
            pos = track.get('position_adjusted') or track.get('position')
            if pos is None:
                continue
            map_pos = homography_calc.transform_point(
                pos, H,
                canvas_w=tactical_renderer.canvas_w,
                canvas_h=tactical_renderer.canvas_h,
                margin=5
            )
            if not is_valid_map_pos(map_pos, tactical_renderer.canvas_w,
                                    tactical_renderer.canvas_h):
                map_pos = None
            player_positions[player_id] = {
                'map_pos': map_pos,
                'team':    track.get('team', -1),
                'role':    'player'
            }

        for gk_id, track in tracks['goalkeeper'][frame_num].items():
            pos = track.get('position_adjusted') or track.get('position')
            if pos is None:
                continue
            map_pos = homography_calc.transform_point(pos, H)
            if not is_valid_map_pos(map_pos, tactical_renderer.canvas_w,
                                    tactical_renderer.canvas_h):
                map_pos = None
            player_positions[f"gk_{gk_id}"] = {
                'map_pos': map_pos,
                'team':    track.get('team', -1),
                'role':    'goalkeeper'
            }

        # Ball pos pakai versi smooth
        ball_pos = all_ball_map_pos_smooth[frame_num]

        # Trail kumulatif dari frame 0 sampai sekarang
        trail_so_far = all_ball_map_pos_smooth[:frame_num + 1]

        tac_frame = tactical_renderer.render(
            player_positions=player_positions,
            ball_position=ball_pos,
            ball_trail=trail_so_far,   # ← tambahan baru
        )
        tactical_frames.append(tac_frame)

    print(f"[Homography] Total H matrix updates: {homography_calc.update_count}")

    # ── 9. Draw Annotations ──────────────────────────────────────────────────
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # ── 10. Side-by-side: Video | Tactical Map ───────────────────────────────
    frame_h, frame_w = video_frames[0].shape[:2]

    tac_display_h = frame_h
    tac_display_w = int(
        tac_display_h * tactical_renderer.canvas_w / tactical_renderer.canvas_h
    )

    combined_frames = []
    for frame_num in range(len(output_video_frames)):
        tac_resized = cv2.resize(
            tactical_frames[frame_num],
            (tac_display_w, tac_display_h)
        )
        divider = np.ones((frame_h, 4, 3), dtype=np.uint8) * 180
        combined = np.hstack([output_video_frames[frame_num], divider, tac_resized])
        combined_frames.append(combined)

    # ── 11. Save Output ──────────────────────────────────────────────────────
    output_name = os.path.basename(video_path).replace('.mp4', '_output.mp4')
    save_video(combined_frames, f"output_videos/{output_name}")
    print(f"Saved to output_videos/{output_name}")


if __name__ == "__main__":
    main()