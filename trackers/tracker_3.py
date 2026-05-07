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
            lost_track_buffer=120,           # ← turun dari 120 → 30 supaya ID tidak
                                            #   di-recycle terlalu cepat setelah occlusion
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
                new_ids = []
                for raw_id in detections.class_id:
                    mapped = int(raw_id) - 1
                    mapped = max(0, min(mapped, len(class_names) - 1))
                    new_ids.append(mapped)
                detections.class_id = np.array(new_ids)

            detections_list.append(detections)
        return detections_list

    # ──────────────────────────────────────────────────────────────────────────
    #  HELPER
    # ──────────────────────────────────────────────────────────────────────────

    def _is_inside_field(self, bbox, frame_shape, margin=0.12):
        """
        True kalau center bbox ada di dalam area lapangan (bukan pinggir frame).
        margin=0.12 (naik dari 0.08) supaya referee di pinggir garis putih
        tetap dianggap 'luar lapangan'.
        """
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

            frame_shape = frames[frame_num].shape

            # ── Pisahkan bola ──────────────────────────────────────────────
            ball_mask = detections.class_id == class_names.index('ball')
            non_ball  = detections[~ball_mask]

            # # ── Ball ──────────────────────────────────────────────────────
            # ball_dets = detections[ball_mask]
            # if len(ball_dets) > 0:
            #     frame_tracks['ball'][1] = {"bbox": ball_dets.xyxy[0].tolist()}

            # ── Ball ──────────────────────────────────────────────────
            ball_yolo = None
            if self.ball_model is not None:
                yolo_result = self.ball_model.predict(frames[frame_num], conf=0.5, verbose=False, device='mps')[0]  # ← naikkan conf
                best_conf = 0
                for box in yolo_result.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # ── Filter 1: ukuran bbox harus wajar (bola tidak besar) ──
                    w, h = x2 - x1, y2 - y1
                    if w > 60 or h > 60:   # skip kalau terlalu besar
                        continue
                    if w < 5 or h < 5:     # skip kalau terlalu kecil
                        continue
                    
                    # ── Filter 2: aspek rasio harus mendekati bulat ──
                    aspect = w / h if h > 0 else 0
                    if not (0.5 < aspect < 2.0):
                        continue
                        
                    if conf > best_conf:
                        best_conf = conf
                        ball_yolo = {"bbox": [x1, y1, x2, y2]}

            if ball_yolo:
                frame_tracks['ball'][1] = ball_yolo
            # else:
            #     ball_dets = detections[ball_mask]
            #     if len(ball_dets) > 0:
            #         frame_tracks['ball'][1] = {"bbox": ball_dets.xyxy[0].tolist()}  # ← fallback RFDETR

            # ── ByteTrack untuk semua non-ball ─────────────────────────────
            if len(non_ball) > 0:
                tracked = self.tracker.update_with_detections(non_ball)

                for i in range(len(tracked.xyxy)):
                    bbox     = tracked.xyxy[i].tolist()
                    cls_id   = int(tracked.class_id[i])
                    track_id = int(tracked.tracker_id[i])
                    cls_name = class_names[cls_id]
                    inside   = self._is_inside_field(bbox, frame_shape)

                    # ── Di dalam get_object_tracks, ganti blok Referee logic dengan ini: ──

                    # ── Referee logic ────────────────────────────────────────────────────
                    if cls_name == 'referee':
                        if self._is_inside_field(bbox, frame_shape):
                            # Logika VOTING untuk wasit di dalam lapangan
                            if self.locked_referee_id is None:
                                self.referee_vote[track_id] = self.referee_vote.get(track_id, 0) + 1
                                
                                # Jika sudah memenuhi syarat, kunci wasit
                                if self.referee_vote[track_id] >= self.VOTE_WINDOW:
                                    self.locked_referee_id = track_id
                                    self.known_referee_ids.add(track_id)
                                    # ── TAMBAHAN: hapus dari known_player_ids kalau sempat masuk ──
                                    self.known_player_ids.discard(track_id)
                                    print(f"[Tracker] Referee LOCKED (inside field): track_id={track_id} (frame {frame_num})")

                            # Jika ini wasit yang sudah terkunci, daftarkan
                            if self.locked_referee_id == track_id:
                                frame_tracks['referee'][track_id] = {"bbox": bbox}
                            
                            # Jika ini referee dari luar yang masuk lapangan, tetap izinkan
                            elif track_id in self.known_referee_ids:
                                frame_tracks['referee'][track_id] = {"bbox": bbox}
                            
                            # Jika ini adalah "deteksi palsu" di dalam lapangan, paksa jadi player
                            else:
                                # if track_id not in self.known_player_ids:
                                #     if len(self.known_player_ids) < self.MAX_PLAYERS:
                                #         self.known_player_ids.add(track_id)
                                
                                # if track_id in self.known_player_ids:
                                #     frame_tracks['player'][track_id] = {"bbox": bbox}
                                pass

                        else:
                            # Logika untuk Referee di LUAR lapangan
                            if track_id not in self.known_referee_ids:
                                self.referee_vote[track_id] = self.referee_vote.get(track_id, 0) + 1
                                if self.referee_vote[track_id] >= self.VOTE_WINDOW:
                                    self.known_referee_ids.add(track_id)
                                    if self.locked_referee_id is None and self._is_inside_field(bbox, frame_shape):
                                        self.locked_referee_id = track_id
                                # Selama belum confirmed: skip saja, jangan paksa jadi player
                                # kecuali kamu yakin 100% itu bukan referee
                                else:
                                    continue  # ← tunggu dulu, jangan assign ke mana-mana
                            
                            frame_tracks['referee'][track_id] = {"bbox": bbox}

                    elif cls_name == 'goalkeeper':
                        self.known_goalkeeper_ids.add(track_id)
                        frame_tracks['goalkeeper'][track_id] = {"bbox": bbox}

                    # ── Player logic ────────────────────────────────────────
                    elif cls_name == 'player':
                        if track_id not in self.known_player_ids:
                            self.known_player_ids.add(track_id)  # tambah saja, tanpa batas ketat
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
                    
                    # 1. Validasi BBOX dasar: pastikan bbox tidak kosong
                    if not bbox or len(bbox) != 4:
                        continue
                        
                    # 2. Hitung posisi
                    if object_name == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    
                    # 3. Filter "False Positive" (posisi 0,0 atau negatif)
                    # Sering terjadi jika fungsi mapping gagal mengolah input
                    if position[0] <= 0 and position[1] <= 0:
                        # Opsi: berikan peringatan atau lewati agar tidak muncul di peta
                        continue
                    
                    # 4. Assign hanya jika posisi valid
                    tracks[object_name][frame_num][track_id]['position'] = position

    # ──────────────────────────────────────────────────────────────────────────
    #  DRAWING
    # ──────────────────────────────────────────────────────────────────────────

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