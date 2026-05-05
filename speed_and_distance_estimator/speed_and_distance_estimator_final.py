import sys
sys.path.append('../')
from utils import measure_distance
import numpy as np

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate   = 24        # sesuaikan dengan FPS video kamu
        self.MAX_SPEED_KMH = 36.0    # sprint manusia ~36 km/h, bola bisa lebih

    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Gunakan position_adjusted (sudah dalam satuan cm pitch) kalau ada.
        Kalau tidak ada, fallback ke pixel dengan pixel_to_meter kasar.
        """
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object in ('ball', 'referee'):
                continue

            number_of_frames = len(object_tracks)

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
                if last_frame == frame_num:
                    continue

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_pos = (object_tracks[frame_num][track_id].get('position_adjusted')
                                 or object_tracks[frame_num][track_id].get('position'))
                    end_pos   = (object_tracks[last_frame][track_id].get('position_adjusted')
                                 or object_tracks[last_frame][track_id].get('position'))

                    if start_pos is None or end_pos is None:
                        continue

                    distance_raw = measure_distance(start_pos, end_pos)

                    # position_adjusted sudah dalam cm → konversi ke meter
                    # Kalau masih pixel, pakai fallback 0.05 m/px (lebih konservatif)
                    if object_tracks[frame_num][track_id].get('position_adjusted'):
                        distance_m = distance_raw / 100.0   # cm → meter
                    else:
                        distance_m = distance_raw * 0.05    # fallback pixel

                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    if time_elapsed <= 0:
                        continue

                    speed_mps = distance_m / time_elapsed
                    speed_kmh = speed_mps * 3.6

                    # Clamp ke batas fisiologis — jangan hardcode jadi 45
                    # Kalau > MAX, kemungkinan noise tracking, skip saja
                    if speed_kmh > self.MAX_SPEED_KMH:
                        continue   # ← skip, bukan clamp — data noise lebih baik dibuang

                    total_distance.setdefault(object, {})
                    total_distance[object].setdefault(track_id, 0)
                    total_distance[object][track_id] += distance_m

                    for fn in range(frame_num, last_frame):
                        if track_id not in tracks[object][fn]:
                            continue
                        tracks[object][fn][track_id]['speed']    = speed_kmh
                        tracks[object][fn][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        return frames