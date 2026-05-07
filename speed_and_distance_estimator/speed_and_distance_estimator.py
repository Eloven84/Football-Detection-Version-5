import sys
sys.path.append('../')
from utils import measure_distance


class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window    = 5
        self.frame_rate      = 30
        self.pixel_to_meter  = 0.1   # fallback; ganti kalau ada homography akurat

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object in ('ball', 'referee'):
                continue

            number_of_frames = len(object_tracks)

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_pos = (object_tracks[frame_num][track_id].get('position_adjusted')
                                 or object_tracks[frame_num][track_id].get('position'))
                    end_pos   = (object_tracks[last_frame][track_id].get('position_adjusted')
                                 or object_tracks[last_frame][track_id].get('position'))

                    if start_pos is None or end_pos is None:
                        continue

                    distance_px  = measure_distance(start_pos, end_pos)
                    distance_m   = distance_px * self.pixel_to_meter
                    time_elapsed = (last_frame - frame_num) / self.frame_rate

                    if time_elapsed <= 0:
                        continue

                    speed_mps = distance_m / time_elapsed
                    speed_kmh = speed_mps * 3.6

                    if speed_kmh > 45:
                        speed_kmh  = 45.0
                        speed_mps  = speed_kmh / 3.6
                        distance_m = speed_mps * time_elapsed

                    total_distance.setdefault(object, {})
                    total_distance[object].setdefault(track_id, 0)
                    total_distance[object][track_id] += distance_m

                    for fn in range(frame_num, last_frame):
                        if track_id not in tracks[object][fn]:
                            continue
                        tracks[object][fn][track_id]['speed']    = speed_kmh
                        tracks[object][fn][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        """Stub — drawing dilakukan di Tracker._draw_speed_distance()."""
        return frames