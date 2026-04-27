import os
import numpy as np
from trackers import Tracker
from team_assigner import TeamAssigner
from view_transformer import ViewTransformer
from team_ball_control import TeamBallControl
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from utils import read_video, save_video


def main():
    video_path = 'input_video/input_1.mp4'

    video_frames = read_video(video_path)
    print(f"Total frames: {len(video_frames)}")

    stub_name = os.path.basename(video_path).replace('.mp4', '_stub.pkl')
    stub_path = f"stubs/{stub_name}"

    tracker = Tracker(
        r'models/RFDETR Result Dataset With Augmentation Version 1/best_'
    )

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path=stub_path
    )

    # ✅ Interpolasi bola — isi frame kosong
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    tracker.add_object_positions(tracks)

    # Camera Movement
    stubs_name_cm = os.path.basename(video_path).replace('.mp4', '_cammove_stub.pkl')
    stub_path_cm  = f"stubs/{stubs_name_cm}"
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path=stub_path_cm
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Speed & Distance
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Team Assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks['player'], sample_frames=10)

    for frame_num, player_track in enumerate(tracks['player']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id,
                is_goalkeeper=False
            )
            tracks['player'][frame_num][player_id]['team']       = team
            tracks['player'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Lakukan hal yang sama untuk goalkeeper
    for frame_num, gk_track in enumerate(tracks['goalkeeper']):
        for gk_id, track in gk_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                gk_id,
                is_goalkeeper=True
            )
            tracks['goalkeeper'][frame_num][gk_id]['team']       = team
            tracks['goalkeeper'][frame_num][gk_id]['team_color'] = team_assigner.team_colors[team]

    print(f"[TeamAssigner] Total unique players assigned: {len(team_assigner.player_team_dict)}")

    # Ball Assignment & Possession
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

    # Draw — tidak ada team_assigner argument lagi, semua sudah di tracks
    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        team_ball_control
    )
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    output_name = os.path.basename(video_path).replace('.mp4', '_output.mp4')
    save_video(output_video_frames, f"output_videos/{output_name}")


if __name__ == "__main__":
    main()