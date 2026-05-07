
import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        h, w = frame.shape[:2]
        diag = np.sqrt(h**2 + w**2)
        self.minimum_distance = diag * 0.002    # auto threshold based on the diagonal length of the frame
        # self.minimum_distance = 5    # the amount of movement required so it wont be so little, if the movement less than 5 we can consider that the camera is not moving

        # self.lk_params = dict(
        #     winSize = (15, 15),
        #     maxLevel = 2,      # pyramids to downscale the image to get larger features, downscale twice
        #     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        # )

        scale = max(w, h) / 1000
        self.lk_params = dict(
            winSize = (int(15*scale), int(15*scale)),
            maxLevel = 3 if scale > 1 else 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # create mask to avoid unwanted features
        
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # mask_features = np.zeros_like(first_frame_grayscale)
        # mask_features[:, 0:20] = 1      # get the banner from the top
        # mask_features[:, 900:1050] = 1      # get the banner from the bottom
        h = frame.shape[0]
        mask_features = np.zeros_like(first_frame_grayscale)

        top_ratio = 0.03       # top 3%
        bottom_ratio = 0.15    # bottom 15%

        mask_features[:, :int(h * top_ratio)] = 1
        mask_features[:, int(h * (1 - bottom_ratio)):] = 1


        # self.features = dict(
        #     maxCorners = 100,   # maximum amount of corners we can utilize
        #     qualityLevel = 0.3, # the higher the level better the features, but the lesser amount of features you can get
        #     minDistance = 3,    # in pixel
        #     blockSize = 7,      # search size of the picture
        #     mask = mask_features
        # )

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = np.var(frame_gray)

        if variance < 200:   # low detail video
            maxCorners = 200
            qualityLevel = 0.01
        else:
            maxCorners = 100
            qualityLevel = 0.03

        minDistance = max(3, int(min(h, w) * 0.005))

        self.features = dict(
            maxCorners = maxCorners,
            qualityLevel = qualityLevel,
            minDistance = minDistance,
            blockSize = 7,
            mask = mask_features
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # read the stub when we have it
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # crate camera movement per frame --> calculate camera movement
        camera_movement = [[0,0]] * len(frames)     # it will be 0,0 for all frames when we initialize it

        # convert image into a gray image to extract the image
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)    # track corner features, ** to expand the dictionary into the parameters

        for frame_num in range (1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # see where the new features are
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            # measure the disctance between the old and new features, and whether there's a distance in the camera movement or not
            max_distance = 0      # get maximum distance between any two features
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):    # in order to zip any 2 lists we have to zip it first
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance :
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
            
            if max_distance > self.minimum_distance or frame_num % 10 == 0:  # if the movement is significant or every 10 frames we update the camera movement
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    # def draw_camera_movement(self, frames, camera_movement_per_frame):
    #     output_frames = []

    #     for frame_num, frame in enumerate (frames):
    #         frame = frame.copy()    # in order not to contaminate what was inputed to the function
            
    #         overlay = frame.copy()
    #         cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), -1)
    #         alpha = 1
    #         cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
            
    #         x_movement, y_movement = camera_movement_per_frame[frame_num]
    #         frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,0,0), 3)
    #         frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,0,0), 3)

    #         output_frames.append(frame)

    #     return output_frames

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            h, w = frame.shape[:2]

            # --- AUTO-SCALE UI ELEMENTS ---
            scale = max(w, h) / 1080   # normalized scale vs FullHD

            rect_w = int(w * 0.28)     # 28% of width
            rect_h = int(h * 0.10)     # 10% of height

            font_scale = 0.5 * scale
            thickness = max(1, int(2 * scale))

            # --- DRAW AUTOMATIC RECTANGLE ---
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (int(w * 0.01), int(h * 0.02)),                   # auto margin
                (int(w * 0.01) + rect_w, int(h * 0.02) + rect_h),
                (255, 255, 255),
                -1
            )
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # --- PLACE TEXT DYNAMICALLY ---
            x_movement, y_movement = camera_movement_per_frame[frame_num]

            text_x = int(w * 0.03)
            text_y1 = int(h * 0.05)
            text_y2 = int(h * 0.09)

            cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}",
                        (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness)

            cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}",
                        (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness)

            output_frames.append(frame)

        return output_frames
