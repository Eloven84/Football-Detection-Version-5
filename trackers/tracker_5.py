import os
import cv2
import numpy as np
import pickle
import pandas as pd
from rfdetr import RFDETRBase
import supervision as sv
from PIL import Image
from ultralytics import YOLO
from utils import get_center_of_bbox, get_foot_position

full_classes = ['player-ball-goalkeeper-referee-QfA4', 'ball', 'goalkeeper', 'player', 'referee']
class_names  = full_classes[1:]  # ['ball', 'goalkeeper', 'player', 'referee']

# ── Annotation style constants ─────────────────────────────────────────────────
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_STAT = 0.42
FONT_THICK_STAT = 1
STAT_BG_ALPHA   = 0.72
STAT_PAD_X      = 5
STAT_PAD_Y      = 3


class Tracker:
    def __init__(self, model_path, ball_model_path=None):
        self.model = RFDETRBase(pretrain_weights=model_path)
        self.ball_model = YOLO(ball_model_path) if ball_model_path else None
        self.model.optimize_for_inference()

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=120,
            minimum_matching_threshold=0.7,
            frame_rate=30
        )

        # ── Referee locking ────────────────────────────────────────────────────
        self.locked_referee_id  = None
        self.referee_vote       = {}
        self.VOTE_WINDOW        = 5
        self.known_referee_ids  = set()

        # ── ID management ──────────────────────────────────────────────────────
        self.known_player_ids     = set()
        self.known_goalkeeper_ids = set()
        self.MAX_PLAYERS          = 30
        self.MAX_GOALKEEPERS      = 6

    # ──────────────────────────────────────────────────────────────────────────
    #  DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect_frames(self, frames):
        detections_list = []
        for frame in frames:
            pil_image  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = self.model.predict(pil_image, threshold=0.3)

            if detections is not None and len(detections) > 0:
                n = len(detections.xyxy)

                # ── Strip mismatched data fields ──
                clean_data = {}
                for k, v in detections.data.items():
                    if hasattr(v, '__len__') and len(v) == n:
                        clean_data[k] = v
                detections.data = clean_data

                # ── Remap class_id only if 1-based ──
                if len(detections.class_id) > 0:
                    max_id = int(detections.class_id.max())
                    if max_id >= len(class_names):
                        detections.class_id = np.clip(
                            detections.class_id.astype(int) - 1,
                            0,
                            len(class_names) - 1
                        )

            detections_list.append(detections)
        return detections_list

    # ──────────────────────────────────────────────────────────────────────────
    #  HELPER
    # ──────────────────────────────────────────────────────────────────────────

    def _is_inside_field(self, bbox, frame_shape, margin=0.12):
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (margin * w < cx < (1 - margin) * w) and \
               (margin * h < cy < (1 - margin) * h)

    # ──────────────────────────────────────────────────────────────────────────
    #  TRACKING
    # ──────────────────────────────────────────────────────────────────────────

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

            if len(detections.class_id) != len(detections.xyxy):
                print(f"[WARNING] Frame {frame_num}: class_id length mismatch "
                      f"({len(detections.class_id)} vs {len(detections.xyxy)}), skipping frame")
                for name in tracks:
                    tracks[name].append(frame_tracks[name])
                continue

            frame_shape = frames[frame_num].shape

            # ── Pisahkan bola ──────────────────────────────────────────────
            ball_mask = detections.class_id == class_names.index('ball')
            non_ball  = detections[~ball_mask]

            # ── Ball via YOLO ──────────────────────────────────────────────
            ball_yolo = None
            if self.ball_model is not None:
                yolo_result = self.ball_model.predict(
                    frames[frame_num], conf=0.5, verbose=False, device='mps'
                )[0]
                best_conf = 0
                for box in yolo_result.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    if w > 60 or h > 60:
                        continue
                    if w < 5 or h < 5:
                        continue
                    aspect = w / h if h > 0 else 0
                    if not (0.5 < aspect < 2.0):
                        continue
                    if conf > best_conf:
                        best_conf = conf
                        ball_yolo = {"bbox": [x1, y1, x2, y2]}

            if ball_yolo:
                frame_tracks['ball'][1] = ball_yolo

            # ── ByteTrack untuk semua non-ball ─────────────────────────────
            if len(non_ball) > 0:
                tracked = self.tracker.update_with_detections(non_ball)

                for i in range(len(tracked.xyxy)):
                    bbox     = tracked.xyxy[i].tolist()
                    cls_id   = int(tracked.class_id[i])
                    track_id = int(tracked.tracker_id[i])
                    cls_name = class_names[cls_id]
                    inside   = self._is_inside_field(bbox, frame_shape)

                    if cls_name == 'referee':
                        if self._is_inside_field(bbox, frame_shape):
                            if self.locked_referee_id is None:
                                self.referee_vote[track_id] = self.referee_vote.get(track_id, 0) + 1
                                if self.referee_vote[track_id] >= self.VOTE_WINDOW:
                                    self.locked_referee_id = track_id
                                    self.known_referee_ids.add(track_id)
                                    self.known_player_ids.discard(track_id)
                                    print(f"[Tracker] Referee LOCKED (inside field): track_id={track_id} (frame {frame_num})")
                            if self.locked_referee_id == track_id:
                                frame_tracks['referee'][track_id] = {"bbox": bbox}
                            elif track_id in self.known_referee_ids:
                                frame_tracks['referee'][track_id] = {"bbox": bbox}
                            else:
                                pass
                        else:
                            if track_id not in self.known_referee_ids:
                                self.referee_vote[track_id] = self.referee_vote.get(track_id, 0) + 1
                                if self.referee_vote[track_id] >= self.VOTE_WINDOW:
                                    self.known_referee_ids.add(track_id)
                                    if self.locked_referee_id is None and self._is_inside_field(bbox, frame_shape):
                                        self.locked_referee_id = track_id
                                else:
                                    continue
                            frame_tracks['referee'][track_id] = {"bbox": bbox}

                    elif cls_name == 'goalkeeper':
                        self.known_goalkeeper_ids.add(track_id)
                        frame_tracks['goalkeeper'][track_id] = {"bbox": bbox}

                    elif cls_name == 'player':
                        if track_id not in self.known_player_ids:
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

    # ──────────────────────────────────────────────────────────────────────────
    #  INTERPOLASI BOLA
    # ──────────────────────────────────────────────────────────────────────────

    def interpolate_ball_positions(self, ball_positions):
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

    # ──────────────────────────────────────────────────────────────────────────
    #  POSITIONS
    # ──────────────────────────────────────────────────────────────────────────

    def add_object_positions(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if not bbox or len(bbox) != 4:
                        continue
                    if object_name == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    if position[0] <= 0 and position[1] <= 0:
                        continue
                    tracks[object_name][frame_num][track_id]['position'] = position

    # ──────────────────────────────────────────────────────────────────────────
    #  DRAWING — method lama (tidak diubah, tetap berfungsi)
    # ──────────────────────────────────────────────────────────────────────────

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Method anotasi original — tetap dipertahankan agar kompatibel
        dengan pemanggilan dari main.py / kode lama.
        Warna diambil dari track['team_color'] yang sudah di-set TeamAssigner.
        """
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
                frame = self._draw_speed_distance(frame, player, player['bbox'])

            for track_id, gk in goalkeeper_dict.items():
                color = gk.get('team_color', (0, 255, 255))
                color = tuple(map(int, color.tolist() if hasattr(color, 'tolist') else color))
                frame = self.draw_ellipse(frame, gk['bbox'], color, track_id)
                if gk.get('has_ball', False):
                    frame = self.draw_triangle(frame, gk['bbox'], (0, 255, 0))
                frame = self._draw_speed_distance(frame, gk, gk['bbox'])

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255), track_id)

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_frames.append(frame)

        return output_frames

    # ──────────────────────────────────────────────────────────────────────────
    #  DRAWING — method baru dengan warna dari user (app.py)
    # ──────────────────────────────────────────────────────────────────────────

    def draw_annotations_with_colors(
        self,
        video_frames,
        tracks,
        team_ball_control,
        annotation_colors: dict | None = None,
        gk_colors:         dict | None = None,
        referee_color:     tuple       = (0, 255, 255),
    ):
        """
        Versi draw_annotations yang menerima warna bounding box dari luar
        (dikirim user via form upload di index.html → app.py → pipeline).

        Parameters
        ----------
        annotation_colors : dict {team_id: (B, G, R)}
            Warna ellipse untuk player biasa per tim.
            Contoh: {1: (30, 80, 220), 2: (220, 80, 30)}
            Jika None, fallback ke team_color dari track (perilaku lama).

        gk_colors : dict {team_id: (B, G, R)}
            Warna ellipse khusus untuk goalkeeper per tim.
            Contoh: {1: (85, 173, 246), 2: (72, 187, 120)}
            Jika None, fallback ke team_color dari track.

        referee_color : tuple (B, G, R)
            Warna ellipse untuk wasit.
            Default: (0, 255, 255) — cyan, sama dengan perilaku lama.
        """
        # Pastikan semua warna berupa tuple int (bukan numpy array)
        def _safe_color(c):
            if c is None:
                return None
            return tuple(int(v) for v in (c.tolist() if hasattr(c, 'tolist') else c))

        # Normalisasi annotation_colors dan gk_colors sekali di awal
        ann_c = {k: _safe_color(v) for k, v in annotation_colors.items()} \
            if annotation_colors else {}
        gk_c  = {k: _safe_color(v) for k, v in gk_colors.items()} \
            if gk_colors else {}
        ref_c = _safe_color(referee_color) or (0, 255, 255)

        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict     = tracks['player'][frame_num]
            goalkeeper_dict = tracks['goalkeeper'][frame_num]
            referee_dict    = tracks['referee'][frame_num]
            ball_dict       = tracks['ball'][frame_num]

            # ── Player ────────────────────────────────────────────────────
            for track_id, player in player_dict.items():
                team = player.get('team', -1)

                # Prioritas: warna dari user → warna dari TeamAssigner → default
                if team in ann_c:
                    color = ann_c[team]
                else:
                    raw = player.get('team_color', (0, 0, 255))
                    color = _safe_color(raw)

                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 255, 0))
                frame = self._draw_speed_distance(frame, player, player['bbox'])

            # ── Goalkeeper ────────────────────────────────────────────────
            for track_id, gk in goalkeeper_dict.items():
                team = gk.get('team', -1)

                # Prioritas: gk_colors dari user → team_color dari TeamAssigner → cyan
                if team in gk_c:
                    color = gk_c[team]
                else:
                    raw = gk.get('team_color', (0, 255, 255))
                    color = _safe_color(raw)

                frame = self.draw_ellipse(frame, gk['bbox'], color, track_id)
                if gk.get('has_ball', False):
                    frame = self.draw_triangle(frame, gk['bbox'], (0, 255, 0))
                frame = self._draw_speed_distance(frame, gk, gk['bbox'])

            # ── Referee ───────────────────────────────────────────────────
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], ref_c, track_id)

            # ── Ball ──────────────────────────────────────────────────────
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_frames.append(frame)

        return output_frames

    # ──────────────────────────────────────────────────────────────────────────
    #  SPEED & DISTANCE ANNOTATION
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_speed_distance(self, frame, track_info, bbox):
        speed    = track_info.get('speed')
        distance = track_info.get('distance')

        if speed is None and distance is None:
            return frame

        x1, y1, x2, y2 = [int(v) for v in bbox]

        lines = []
        if speed is not None:
            lines.append(f"{speed:.1f} km/h")
        if distance is not None:
            lines.append(f"{distance:.1f} m")

        if not lines:
            return frame

        fs    = FONT_SCALE_STAT
        thick = FONT_THICK_STAT
        sizes = [cv2.getTextSize(ln, FONT, fs, thick)[0] for ln in lines]
        max_w = max(s[0] for s in sizes)
        line_h = sizes[0][1]
        total_h = len(lines) * (line_h + STAT_PAD_Y * 2) + STAT_PAD_Y

        box_w  = max_w + STAT_PAD_X * 2
        box_x1 = max(0, (x1 + x2) // 2 - box_w // 2)
        box_x2 = min(frame.shape[1] - 1, box_x1 + box_w)
        box_y2 = max(total_h, y1 - 4)
        box_y1 = max(0, box_y2 - total_h)

        roi = frame[box_y1:box_y2, box_x1:box_x2]
        if roi.size > 0:
            bg = roi.copy()
            bg[:] = (20, 20, 20)
            frame[box_y1:box_y2, box_x1:box_x2] = cv2.addWeighted(
                bg, STAT_BG_ALPHA, roi, 1 - STAT_BG_ALPHA, 0
            )

        text_x = box_x1 + STAT_PAD_X
        for i, (line, (tw, th)) in enumerate(zip(lines, sizes)):
            ty = box_y1 + STAT_PAD_Y + (i + 1) * (th + STAT_PAD_Y)
            ty = min(ty, frame.shape[0] - 2)
            cv2.putText(frame, line, (text_x + 1, ty + 1),
                        FONT, fs, (0, 0, 0), thick + 1, cv2.LINE_AA)
            cv2.putText(frame, line, (text_x, ty),
                        FONT, fs, (255, 255, 255), thick, cv2.LINE_AA)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  DRAW PRIMITIVES
    # ──────────────────────────────────────────────────────────────────────────

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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        x1, y1, x2, _ = [int(v) for v in bbox]
        cx = (x1 + x2) // 2
        tri = np.array([[cx, y1-10], [cx-10, y1-20], [cx+10, y1-20]])
        cv2.drawContours(frame, [tri], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [tri], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        ctrl = team_ball_control[:frame_num+1]
        t1 = (ctrl == 1).sum() / len(ctrl) * 100
        t2 = (ctrl == 2).sum() / len(ctrl) * 100
        cv2.putText(frame, f"Team 1 Possession: {t1:.1f}%", (1360, 890),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Possession: {t2:.1f}%", (1360, 940),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return frame