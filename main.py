import os
os.environ['OMP_NUM_THREADS']       = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner          # ✅ dari folder team_assigner
from player_ball_assigner import PlayerBallAssigner  # ✅ dari folder player_ball_assigner

def main():
    VIDEO_NAME = '08fd33_4'

    input_path   = f'input_video/{VIDEO_NAME}.mp4'
    stub_path    = f'stubs/tracks_stub_{VIDEO_NAME}.pkl'
    cropped_path = f'output_videos/cropped_image_{VIDEO_NAME}.jpg'
    output_path  = f'output_videos/output_{VIDEO_NAME}.avi'

    video_frames = read_video(input_path)
    print(f"Video        : {input_path}")
    print(f"Total frames : {len(video_frames)}")

    tracker = Tracker(
        r'E:\FOLDER TUGAS AKHIR 2\Football2\models\RFDETR Result Dataset With Augmentation Version 2\checkpoint_best_total.pth'
    )

    os.makedirs('stubs', exist_ok=True)
    os.makedirs('output_videos', exist_ok=True)

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=stub_path
    )

    # Interpolasi posisi bola
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    for cls_name, frame_list in tracks.items():
        total_detections = sum(len(f) for f in frame_list)
        print(f"  {cls_name}: {total_detections} total deteksi")

    # Save cropped image pemain pertama
    for track_id, player in tracks['players'][0].items():
        bbox          = player['bbox']
        frame         = video_frames[0]
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        cv2.imwrite(cropped_path, cropped_image)
        break

    # Assign Team Color dari 30 frame pertama
    team_assigner      = TeamAssigner()
    all_players_sample = {}
    for frame_num in range(min(30, len(video_frames))):
        all_players_sample.update(tracks['players'][frame_num])

    print(f"Jumlah sampel pemain: {len(all_players_sample)}")
    team_assigner.assign_team_color(video_frames[0], all_players_sample)

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track_info in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track_info['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team']       = team
            tracks['players'][frame_num][player_id]['team_color'] = \
                team_assigner.team_colors[team]

    # Assign Ball ke Pemain Terdekat
    player_assigner   = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_data = tracks['ball'][frame_num].get(1, None)
        if ball_data is None:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue

        ball_bbox       = ball_data['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team']
            )
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)
    print(f"Team 1 ball control: {np.sum(team_ball_control == 1)} frames")
    print(f"Team 2 ball control: {np.sum(team_ball_control == 2)} frames")

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, output_path)
    print(f"Video tersimpan: {output_path}")

if __name__ == '__main__':
    main()