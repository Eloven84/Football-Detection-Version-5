import os
import cv2
import numpy as np
import pickle
import pandas as pd
from rfdetr import RFDETRBase
import supervision as sv
from PIL import Image
from utils import get_center_of_bbox, get_foot_position

full_classes = ['player-ball-goalkeeper-referee-QfA4', 'ball', 'goalkeeper', 'player', 'referee']
class_names  = full_classes[1:]  # ['ball', 'goalkeeper', 'player', 'referee']

class Tracker:
    def __init__(self, model_path):
        self.model = RFDETRBase(pretrain_weights=model_path)
        self.model.optimize_for_inference()

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=120,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        # --- Referee locking ---
        # Hanya 1 referee di DALAM lapangan yang dikunci.
        # Referee di LUAR lapangan tetap digambar tanpa limit.
        self.locked_referee_id   = None   # track_id wasit di dalam lapangan
        self.referee_vote        = {}     # {track_id: vote_count}
        self.VOTE_WINDOW         = 20     # kumpulkan vote N frame sebelum kunci
        self.known_referee_ids = set()  # semua referee yang sudah dikonfirmasi

        # --- ID Freezing untuk player & goalkeeper ---
        self.known_player_ids     = set()
        self.known_goalkeeper_ids = set()
        self.MAX_PLAYERS     = 25
        self.MAX_GOALKEEPERS = 4

    # ------------------------------------------------------------------ #
    #  DETECTION                                                           #
    # ------------------------------------------------------------------ #

    def detect_frames(self, frames):
        detections_list = []
        for frame in frames:
            pil_image  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = self.model.predict(pil_image, threshold=0.3)

            if detections is not None and len(detections) > 0:
                new_ids = []
                for raw_id in detections.class_id:
                    mapped = int(raw_id) - 1
                    mapped = max(0, min(mapped, len(class_names) - 1))
                    new_ids.append(mapped)
                detections.class_id = np.array(new_ids)

            detections_list.append(detections)
        return detections_list

    # ------------------------------------------------------------------ #
    #  HELPER                                                              #
    # ------------------------------------------------------------------ #

    def _is_inside_field(self, bbox, frame_shape, margin=0.08):
        """True kalau center bbox ada di dalam area lapangan (bukan pinggir frame)."""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (margin * w < cx < (1 - margin) * w) and \
               (margin * h < cy < (1 - margin) * h)

    # ------------------------------------------------------------------ #
    #  TRACKING                                                            #
    # ------------------------------------------------------------------ #

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                print(f"[Tracker] Loaded tracks from stub: {stub_path}")
                return pickle.load(f)

        detections_list = self.detect_frames(frames)

        tracks = {"ball": [], "goalkeeper": [], "player": [], "referee": []}

        for frame_num, detections in enumerate(detections_list):
            frame_tracks = {name: {} for name in tracks}

            if detections is None or len(detections) == 0:
                for name in tracks:
                    tracks[name].append(frame_tracks[name])
                continue

            frame_shape = frames[frame_num].shape

            # ── Pisahkan bola ──────────────────────────────────────────
            ball_mask = detections.class_id == class_names.index('ball')
            non_ball  = detections[~ball_mask]

            # ── Ball ──────────────────────────────────────────────────
            ball_dets = detections[ball_mask]
            if len(ball_dets) > 0:
                frame_tracks['ball'][1] = {"bbox": ball_dets.xyxy[0].tolist()}

            # ── ByteTrack untuk semua non-ball ────────────────────────
            if len(non_ball) > 0:
                tracked = self.tracker.update_with_detections(non_ball)

                for i in range(len(tracked.xyxy)):
                    bbox     = tracked.xyxy[i].tolist()
                    cls_id   = int(tracked.class_id[i])
                    track_id = int(tracked.tracker_id[i])
                    cls_name = class_names[cls_id]
                    inside   = self._is_inside_field(bbox, frame_shape)

                    # ── Referee logic ──────────────────────────────────
                    if cls_name == 'referee':
                        # Tambahkan validasi confidence jika deteksi sangat goyah
                        # pastikan bbox valid
                        if bbox[2] - bbox[0] < 5 or bbox[3] - bbox[1] < 5: 
                            continue
                        if inside:
                            if self.locked_referee_id is None:
                                self.referee_vote[track_id] = \
                                    self.referee_vote.get(track_id, 0) + 1

                                if self.referee_vote[track_id] >= self.VOTE_WINDOW:
                                    self.locked_referee_id = track_id
                                    self.known_referee_ids.add(track_id)  # ✅ daftarkan
                                    print(f"[Tracker] Referee LOCKED (dalam lapangan): "
                                        f"track_id={track_id} (frame {frame_num})")

                            if self.locked_referee_id == track_id:
                                frame_tracks['referee'][track_id] = {"bbox": bbox}
                            else:
                                # ID lain dideteksi sebagai referee di dalam → paksa jadi player
                                # tapi hanya kalau bukan referee yang sudah dikenal dari luar
                                if track_id not in self.known_referee_ids:
                                    if track_id not in self.known_player_ids:
                                        if len(self.known_player_ids) < self.MAX_PLAYERS:
                                            self.known_player_ids.add(track_id)
                                    if track_id in self.known_player_ids:
                                        frame_tracks['player'][track_id] = {"bbox": bbox}
                                else:
                                    # Ini referee luar yang kebetulan masuk lapangan sebentar
                                    frame_tracks['referee'][track_id] = {"bbox": bbox}
                        else:
                            # Referee di LUAR lapangan
                            if track_id not in self.known_referee_ids:
                                self.known_referee_ids.add(track_id)  # ✅ daftarkan segera
                                print(f"[Tracker] Referee CONFIRMED (luar lapangan): track_id={track_id}")
                            frame_tracks['referee'][track_id] = {"bbox": bbox}

                    # ── Goalkeeper logic ───────────────────────────────
                    elif cls_name == 'goalkeeper':
                        if track_id not in self.known_goalkeeper_ids:
                            if len(self.known_goalkeeper_ids) >= self.MAX_GOALKEEPERS:
                                # Overflow → masukkan sebagai player
                                if track_id not in self.known_player_ids:
                                    if len(self.known_player_ids) < self.MAX_PLAYERS:
                                        self.known_player_ids.add(track_id)
                                if track_id in self.known_player_ids:
                                    frame_tracks['player'][track_id] = {"bbox": bbox}
                                continue
                            self.known_goalkeeper_ids.add(track_id)
                        frame_tracks['goalkeeper'][track_id] = {"bbox": bbox}

                    # ── Player logic ───────────────────────────────────
                    elif cls_name == 'player':
                        if track_id not in self.known_player_ids:
                            if len(self.known_player_ids) >= self.MAX_PLAYERS:
                                continue
                            self.known_player_ids.add(track_id)
                        frame_tracks['player'][track_id] = {"bbox": bbox}

            for name in tracks:
                tracks[name].append(frame_tracks[name])

        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            print(f"[Tracker] Saved tracks to stub: {stub_path}")

        return tracks

    # ------------------------------------------------------------------ #
    #  INTERPOLASI BOLA                                                    #
    # ------------------------------------------------------------------ #

    def interpolate_ball_positions(self, ball_positions):
        """Isi frame bola yang kosong dengan interpolasi linear."""
        ball_positions_list = [x.get(1, {}).get('bbox', []) for x in ball_positions]

        has_any = any(len(b) == 4 for b in ball_positions_list)
        if not has_any:
            print("[WARNING] Tidak ada bola terdeteksi, interpolasi dilewati.")
            return ball_positions

        ball_positions_list = [
            b if len(b) == 4 else [None, None, None, None]
            for b in ball_positions_list
        ]

        df = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()
        df = df.ffill()

        return [
            {1: {"bbox": row}} if None not in row else {}
            for row in df.to_numpy().tolist()
        ]

    # ------------------------------------------------------------------ #
    #  POSITIONS                                                           #
    # ------------------------------------------------------------------ #

    def add_object_positions(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_name == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_name][frame_num][track_id]['position'] = position

    # ------------------------------------------------------------------ #
    #  DRAWING                                                             #
    # ------------------------------------------------------------------ #

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict     = tracks['player'][frame_num]
            goalkeeper_dict = tracks['goalkeeper'][frame_num]
            referee_dict    = tracks['referee'][frame_num]
            ball_dict       = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                color = tuple(map(int, color.tolist() if hasattr(color, 'tolist') else color))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 255, 0))

            for track_id, gk in goalkeeper_dict.items():
                color = gk.get('team_color', (0, 255, 255))
                color = tuple(map(int, color.tolist() if hasattr(color, 'tolist') else color))
                frame = self.draw_ellipse(frame, gk['bbox'], color, track_id)
                if gk.get('has_ball', False):
                    frame = self.draw_triangle(frame, gk['bbox'], (0, 255, 0))

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255), track_id)

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_frames.append(frame)

        return output_frames

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx = (x1 + x2) // 2
        w  = x2 - x1
        cv2.ellipse(frame, (cx, y2), (int(w/2), int(0.35*w)),
                    0, -45, 235, color, 2, cv2.LINE_4)
        if track_id is not None:
            rw, rh = 40, 20
            cv2.rectangle(frame, (cx-rw//2, y2+5), (cx+rw//2, y2+5+rh), color, cv2.FILLED)
            cv2.putText(frame, str(track_id), (cx-rw//2+2, y2+5+rh-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        x1, y1, x2, _ = [int(v) for v in bbox]
        cx = (x1 + x2) // 2
        tri = np.array([[cx, y1-10], [cx-10, y1-20], [cx+10, y1-20]])
        cv2.drawContours(frame, [tri], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [tri], 0, (0,0,0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255,255,255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        ctrl = team_ball_control[:frame_num+1]
        t1 = (ctrl == 1).sum() / len(ctrl) * 100
        t2 = (ctrl == 2).sum() / len(ctrl) * 100
        cv2.putText(frame, f"Team 1 Possession: {t1:.1f}%", (1360, 890),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Possession: {t2:.1f}%", (1360, 940),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        return frame