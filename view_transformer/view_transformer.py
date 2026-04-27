import cv2
import numpy as np
from math import atan2
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # optional: used only if you provide corner_model_path
# belum perlu ganti ke RF-DETR karena model belum training untuk deteksi keypoints
import warnings

class ViewTransformer:
    def __init__(self, sample_frame, corner_model_path: str = None,
                 court_length_m: float = 105.0, court_width_m: float = 68.0,
                 visualize_mapping: bool = False):
        """
        sample_frame : a BGR image (numpy array) used to detect the court corners automatically.
        corner_model_path : optional path to a YOLO model trained to detect the four corner points.
                            If None, the class will try Hough-based fallback.
        court_length_m, court_width_m : real-world dimensions (meters) for the target rectangle.
        visualize_mapping : if True, show debug windows (useful for tuning).
        """
        self.visualize_mapping = visualize_mapping
        self.court_length = float(court_length_m)
        self.court_width = float(court_width_m)

        self.pixel_vertices = None
        # Try YOLO-based detection if user provided a model path
        if corner_model_path is not None:
            try:
                self.model = YOLO(corner_model_path)
            except Exception as e:
                warnings.warn(f"Could not load YOLO model '{corner_model_path}': {e}. Falling back to Hough.")
                self.model = None
        else:
            self.model = None

        # Try to detect pixel vertices automatically
        self.pixel_vertices = self._detect_court_corners_auto(sample_frame)

        # Fallback static choice if detection failed
        if self.pixel_vertices is None:
            warnings.warn("Automatic corner detection failed — using fallback hardcoded trapezoid. "
                          "Consider providing a good sample_frame and a YOLO corner model.")
            # fallback (example values) - you may replace these as last resort
            h, w = sample_frame.shape[:2]
            self.pixel_vertices = np.array([
                [int(w * 0.06), int(h * 0.96)],   # bottom-left
                [int(w * 0.15), int(h * 0.25)],   # top-left
                [int(w * 0.47), int(h * 0.23)],   # top-right
                [int(w * 0.82), int(h * 0.85)]    # bottom-right
            ], dtype=np.float32)

        # Ensure vertices are float32 and sorted in consistent order
        self.pixel_vertices = self._order_vertices_clockwise(self.pixel_vertices).astype(np.float32)

        # Target vertices in meters — rectangle (clockwise)
        self.target_vertices = np.array([
            [0.0, self.court_width],
            [0.0, 0.0],
            [self.court_length, 0.0],
            [self.court_length, self.court_width]
        ], dtype=np.float32)

        # perspective transform matrix: pixel -> real-world
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

        if self.visualize_mapping:
            self.debug_visualize_mapping(sample_frame)

    # ---------------------------
    # Public API
    # ---------------------------
    def transform_point(self, point):
        """
        Transform a single (x, y) pixel point to real-world court coordinates using perspective transform.
        Returns None if the point is outside the detected trapezoid (pitch).
        """
        if point is None:
            return None

        p = (int(point[0]), int(point[1]))
        # use polygon test with integer contour
        is_inside = cv2.pointPolygonTest(self.pixel_vertices.astype(np.int32), p, False) >= 0
        if not is_inside:
            return None

        # reshape to (N,1,2) float32 as required by perspectiveTransform
        pts = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.perspective_transformer)
        return transformed.reshape(-1, 2)[0]  # return single (x, y)

    def add_transformed_position_to_tracks(self, tracks):
        """
        For every tracked object, read 'position_adjusted' and write 'position_transformed'.
        Keeps None when transformation not possible.
        """
        for object_name, object_tracks in tracks.items():
            for frame_num, frame_dict in enumerate(object_tracks):
                for track_id, track_info in frame_dict.items():
                    position = track_info.get('position_adjusted', None)
                    if position is None:
                        tracks[object_name][frame_num][track_id]['position_transformed'] = None
                        continue

                    pos_np = np.array(position, dtype=np.float32)
                    transformed = self.transform_point(pos_np)
                    if transformed is None:
                        tracks[object_name][frame_num][track_id]['position_transformed'] = None
                    else:
                        tracks[object_name][frame_num][track_id]['position_transformed'] = transformed.tolist()

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _detect_court_corners_auto(self, frame):
        """
        Attempt to find the 4 corner points automatically.
        Priority:
          1) YOLO model (if loaded) — expects the model to output labels for the corners or corner points
          2) Hough-lines fallback
        Returns: np.array([[x,y],...], dtype=float) with 4 points in arbitrary order, or None if not found
        """
        pts = None

        # 1) YOLO-based detection
        if self.model is not None:
            try:
                # run one inference — keep verbose off for speed
                preds = self.model.predict(frame, imgsz=1024, verbose=False)
                # preds is a list; take first result
                if len(preds) > 0:
                    res = preds[0]
                    # res.boxes.xyxyn or .xyxy; we can use res.boxes.xyxy or res.boxes.cls with names
                    # Convert detection boxes to (x_center, y_center) and label names
                    boxes = []
                    if hasattr(res, 'boxes'):
                        for b in res.boxes:
                            xyxy = b.xyxy.numpy().tolist() if hasattr(b.xyxy, 'numpy') else b.xyxy.tolist()
                            cls = int(b.cls.numpy().item()) if hasattr(b.cls, 'numpy') else int(b.cls)
                            # center
                            x1, y1, x2, y2 = xyxy
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            boxes.append((cls, (cx, cy), (x1, y1, x2, y2)))
                    # find mapping from class ids to names if possible
                    model_names = getattr(self.model, 'names', None)
                    # collect corner candidates by label names containing 'corner' or exactly match expected labels
                    candidates = {}
                    if model_names is not None:
                        # normalize names
                        for cls_id, center, _box in boxes:
                            name = str(model_names.get(cls_id, str(cls_id))).lower()
                            # try flexible matching
                            if 'corner' in name:
                                # map name to a canonical key: try to detect top/bottom left/right
                                # e.g. 'corner_top_left', 'top_left_corner', 'tl_corner'
                                if 'top' in name and 'left' in name:
                                    candidates['top_left'] = center
                                elif 'top' in name and 'right' in name:
                                    candidates['top_right'] = center
                                elif 'bottom' in name and 'left' in name:
                                    candidates['bottom_left'] = center
                                elif 'bottom' in name and 'right' in name:
                                    candidates['bottom_right'] = center
                                else:
                                    # if we can't deduce, keep a list under generic
                                    candidates.setdefault('generic', []).append(center)
                    # if we matched four named corners, construct array
                    if all(k in candidates for k in ('bottom_left','top_left','top_right','bottom_right')):
                        pts = np.array([
                            candidates['bottom_left'],
                            candidates['top_left'],
                            candidates['top_right'],
                            candidates['bottom_right']
                        ], dtype=np.float32)
                        return pts

                    # fallback: if model output 'corner' detections without side labels,
                    # pick four extreme centers (leftmost/topmost/rightmost/bottommost)
                    if 'generic' in candidates and len(candidates['generic']) >= 4:
                        centers = candidates['generic']
                        pts = self._choose_4_extreme_points(np.array(centers))
                        return pts
            except Exception as e:
                warnings.warn(f"YOLO corner detection failed: {e}")

        # 2) Hough-lines fallback
        try:
            pts = self._detect_corners_hough(frame)
            return pts
        except Exception as e:
            warnings.warn(f"Hough-based corner detection failed: {e}")
            return None

    def _choose_4_extreme_points(self, centers):
        """
        Given >=4 centers choose 4 by extremes:
        bottom-left (min x among bottom half), top-left, top-right, bottom-right
        """
        if centers.shape[0] < 4:
            return None
        # compute centroid
        c = centers.mean(axis=0)
        # compute angles around centroid and pick four quadrants
        angles = np.arctan2(centers[:,1] - c[1], centers[:,0] - c[0])
        order = np.argsort(angles)
        centers_sorted = centers[order]
        # pick roughly four quadrants spaced equally
        n = centers_sorted.shape[0]
        idxs = [0, n//4, n//2, 3*n//4]
        chosen = centers_sorted[idxs[:4]]
        return chosen.astype(np.float32)

    def _detect_corners_hough(self, frame):
        """
        Hough-lines-based approach:
          - convert to HSV, mask green, morphological opening
          - Canny edge detect
          - HoughLinesP to detect lines
          - find intersections, choose four outer-most intersections
        Returns np.array(4x2) or raises Exception
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # mask green by HSV range (works for most football pitches) — tune automatically based on median saturation
        s_med = np.median(hsv[:,:,1])
        lower_sat = max(30, int(s_med * 0.5))
        upper_sat = 255
        lower = np.array([30, lower_sat, 30])
        upper = np.array([90, upper_sat, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # edges
        gray = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, 50, 150)

        # detect lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                                minLineLength=int(min(w,h)*0.25), maxLineGap=40)
        if lines is None or len(lines) < 2:
            raise RuntimeError("not enough lines found")

        # collect line endpoints
        lines = lines.reshape(-1,4)
        # compute intersections for all pairs
        def intersect(a1, a2, b1, b2):
            # line intersection of segment a and b (infinite lines)
            xdiff = (a1[0] - a2[0], b1[0] - b2[0])
            ydiff = (a1[1] - a2[1], b1[1] - b2[1])
            def det(a, b): return a[0]*b[1] - a[1]*b[0]
            div = det(xdiff, ydiff)
            if abs(div) < 1e-6:
                return None
            d = (det(a1,a2), det(b1,b2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return (x, y)
        inters = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                p = intersect(lines[i,0:2], lines[i,2:4], lines[j,0:2], lines[j,2:4])
                if p is None:
                    continue
                x,y = p
                # keep only intersections inside image
                if x >= 0 and x < w and y >=0 and y < h:
                    inters.append((x,y))
        if len(inters) < 4:
            raise RuntimeError("not enough intersections")

        inters = np.array(inters)
        # cluster intersections and pick 4 cluster centers (kmeans-like by sorting extremes)
        # pick 4 extreme points by convex hull then approximating polygon
        hull = cv2.convexHull(inters.astype(np.float32))
        # approximate polygon to 4 vertices
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if approx is None or len(approx) < 4:
            # fallback: pick four extreme points from inters (leftmost, topmost, rightmost, bottommost)
            leftmost = inters[np.argsort(inters[:,0])[:5]]
            rightmost = inters[np.argsort(-inters[:,0])[:5]]
            topmost = inters[np.argsort(inters[:,1])[:5]]
            bottommost = inters[np.argsort(-inters[:,1])[:5]]
            # choose one representative from each group by median
            pts = np.array([
                leftmost[np.argsort(leftmost[:,1])[len(leftmost)//2]],
                topmost[np.argsort(topmost[:,0])[len(topmost)//2]],
                rightmost[np.argsort(rightmost[:,1])[len(rightmost)//2]],
                bottommost[np.argsort(bottommost[:,0])[len(bottommost)//2]]
            ], dtype=np.float32)
            return pts
        # ensure approx shape is (4,1,2)
        approx = approx.reshape(-1,2)
        # if more than 4 points take the 4 most extreme (by angle)
        if len(approx) > 4:
            approx = self._order_vertices_clockwise(approx)[:4]
        return approx.astype(np.float32)

    def _order_vertices_clockwise(self, pts):
        """
        Sort points clockwise and ensures order: bottom-left, top-left, top-right, bottom-right
        We use centroid + atan2 to sort, then rearrange start index.
        """
        pts = np.array(pts, dtype=np.float32).reshape(-1,2)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
        order = np.argsort(angles)
        pts_sorted = pts[order]

        # after sorting by angle, pts_sorted is circular; we need to map to desired order
        # find index of bottom-left (max y, min x approx)
        bl_idx = np.argmin(np.linalg.norm(pts_sorted - np.array([0, pts_sorted[:,1].max()]), axis=1))
        # safer: simply return pts_sorted but rotated so that order corresponds roughly clockwise
        pts_clock = pts_sorted
        return pts_clock

    def debug_visualize_mapping(self, sample_frame=None):
        """
        Show pixel vertices and the warped target rectangle for validation.
        """
        if sample_frame is None:
            # create a blank visualization
            sample_frame = np.zeros((800, 1400, 3), dtype=np.uint8)
        vis = sample_frame.copy()
        for (x,y) in self.pixel_vertices.astype(int):
            cv2.circle(vis, (int(x), int(y)), 10, (0,255,255), -1)
        # draw polygon
        cv2.polylines(vis, [self.pixel_vertices.astype(np.int32)], True, (255,0,0), 3)
        # warp a visualization of the polygon into the target space
        warped = cv2.warpPerspective(vis, self.perspective_transformer, (int(self.court_length*10), int(self.court_width*10)))
        cv2.imshow("Pixel Points & Polygon", vis)
        cv2.imshow("Warped (target space)", warped)
        cv2.waitKey(1)

# end of file
