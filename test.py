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
from homography.homography_calculator import HomographyCalculator
from tactical_map.tactical_map_renderer import TacticalMapRenderer
from utils import read_video, save_video


def main():
    video_path = 'input_video/08fd33_4.mp4'

    # ── 1. Read Video ────────────────────────────────────────────────────────
    video_frames = read_video(video_path)
    print(f"Total frames: {len(video_frames)}")

    # ── 2. Object Tracking ───────────────────────────────────────────────────
    stub_name = os.path.basename(video_path).replace('.mp4', '_stub.pkl')
    stub_path = f"stubs/{stub_name}"

    tracker = Tracker(
        r'models/RFDETR Result Dataset With Augmentation Version 2/checkpoint_best_total.pth'
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
        model_path='model_pitch',
        confidence_threshold=0.1    # naikkan dari 0.1 → 0.5
    )
    homography_calc = HomographyCalculator(
        min_keypoints=4,    # turunkan dari 4 → 3
        common_kp_threshold=3,
        movement_threshold=2
    )
    tactical_renderer = TacticalMapRenderer()

    print("Detecting pitch keypoints...")
    all_keypoints = keypoint_detector.detect_keypoints_batch(video_frames)

    # all_keypoints_dict = []
    # for kp_list in all_keypoints:
    #     if isinstance(kp_list, dict):
    #         all_keypoints_dict.append(kp_list)
    #     else:
    #         kp_dict = {i: kp_list[i] for i in range(len(kp_list))}
    #         all_keypoints_dict.append(kp_dict)

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

    counts = [len(kp) for kp in all_keypoints_dict]
    print(f"[DEBUG] Keypoints per frame — min:{min(counts)}, "
        f"max:{max(counts)}, avg:{sum(counts)/len(counts):.1f}")

    # Debug: visualisasi keypoint frame 0
    debug_frame = video_frames[0].copy()
    for idx, (x, y) in all_keypoints_dict[0].items():
        cv2.circle(debug_frame, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(debug_frame, str(idx), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("debug_keypoints_frame0.jpg", debug_frame)
    print("[DEBUG] Saved debug_keypoints_frame0.jpg")

    # ── 8. Build Tactical Map Per Frame ─────────────────────────────────────
    tactical_frames = []

    for frame_num in range(len(video_frames)):
        current_kp = all_keypoints_dict[frame_num]
        H = homography_calc.get_homography(current_kp)

        player_positions = {}

        for player_id, track in tracks['player'][frame_num].items():
            pos = track.get('position_adjusted') or track.get('position')
            if pos is None:
                continue
            map_pos = homography_calc.transform_point(pos, H) if H is not None else None
            player_positions[player_id] = {
                'map_pos': map_pos,
                'team':    track.get('team', -1),
                'role':    'player'
            }

        for gk_id, track in tracks['goalkeeper'][frame_num].items():
            pos = track.get('position_adjusted') or track.get('position')
            if pos is None:
                continue
            map_pos = homography_calc.transform_point(pos, H) if H is not None else None
            player_positions[f"gk_{gk_id}"] = {
                'map_pos': map_pos,
                'team':    track.get('team', -1),
                'role':    'goalkeeper'
            }

        ball_pos   = None
        ball_frame = tracks['ball'][frame_num]
        if ball_frame:
            bpos = ball_frame[1].get('position_adjusted') or ball_frame[1].get('position')
            if bpos and H is not None:
                ball_pos = homography_calc.transform_point(bpos, H)

        tac_frame = tactical_renderer.render(
            player_positions   = player_positions,
            ball_position      = ball_pos,
            keypoint_positions = None
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