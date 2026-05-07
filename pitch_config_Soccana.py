# pitch_config.py — versi baru HORIZONTAL

PITCH_WIDTH_M  = 105.0
PITCH_HEIGHT_M = 68.0

# Canvas horizontal: gawang di kiri dan kanan
# x_m [0..105] → px (horizontal), y_m [0..68] → py (vertikal)
CANVAS_W = 1050   # 10 px per meter
CANVAS_H = 680
MARGIN_X = 50
MARGIN_Y = 50
FIELD_W  = CANVAS_W - 2 * MARGIN_X   # 950
FIELD_H  = CANVAS_H - 2 * MARGIN_Y   # 580

def m_to_px(x_m, y_m):
    """
    x_m: 0..105 (panjang lapangan, horizontal di canvas)
    y_m: 0..68  (lebar lapangan, vertikal di canvas)
    """
    px = MARGIN_X + int(x_m / PITCH_WIDTH_M  * FIELD_W)
    py = MARGIN_Y + int(y_m / PITCH_HEIGHT_M * FIELD_H)
    return (px, py)

PITCH_KEYPOINTS = {
    # Field Boundaries
    0:  m_to_px(0,     0),       # sideline_top_left
    9:  m_to_px(105,   0),       # sideline_bottom_left  ← PERHATIKAN
    16: m_to_px(0,    68),       # sideline_top_right
    25: m_to_px(105,  68),       # sideline_bottom_right

    # Left Penalty Area
    1:  m_to_px(0,    13.84),    # big_rect_left_top_pt1
    2:  m_to_px(0,    54.16),    # big_rect_left_top_pt2
    3:  m_to_px(16.5, 13.84),    # big_rect_left_bottom_pt1
    4:  m_to_px(16.5, 54.16),    # big_rect_left_bottom_pt2

    # Right Penalty Area
    17: m_to_px(105,  13.84),    # big_rect_right_top_pt1
    18: m_to_px(105,  54.16),    # big_rect_right_top_pt2
    19: m_to_px(88.5, 13.84),    # big_rect_right_bottom_pt1
    20: m_to_px(88.5, 54.16),    # big_rect_right_bottom_pt2

    # Left Goal Area
    5:  m_to_px(0,    24.84),    # small_rect_left_top_pt1
    6:  m_to_px(0,    43.16),    # small_rect_left_top_pt2
    7:  m_to_px(5.5,  24.84),    # small_rect_left_bottom_pt1
    8:  m_to_px(5.5,  43.16),    # small_rect_left_bottom_pt2

    # Right Goal Area
    21: m_to_px(105,  24.84),    # small_rect_right_top_pt1
    22: m_to_px(105,  43.16),    # small_rect_right_top_pt2
    23: m_to_px(99.5, 24.84),    # small_rect_right_bottom_pt1
    24: m_to_px(99.5, 43.16),    # small_rect_right_bottom_pt2

    # Center Elements
    11: m_to_px(52.5,  0),       # center_line_top
    12: m_to_px(52.5, 68),       # center_line_bottom
    15: m_to_px(52.5, 34),       # field_center

    # Center Circle
    13: m_to_px(52.5, 34 - 9.15),   # center_circle_top
    14: m_to_px(52.5, 34 + 9.15),   # center_circle_bottom
    27: m_to_px(52.5 - 9.15, 34),   # center_circle_left
    28: m_to_px(52.5 + 9.15, 34),   # center_circle_right

    # Semicircles
    10: m_to_px(16.5, 34),       # left_semicircle_right
    26: m_to_px(88.5, 34),       # right_semicircle_left
}