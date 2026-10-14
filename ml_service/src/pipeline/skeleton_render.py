# skeleton_render.py - Render a hand skeleton onto a plain white canvas
#
# Shared by the offline Kaggle-photo dataset builder (build_skeleton_dataset.py)
# and the live re-recording script (1b_collect_skeleton_images.py), so the
# rendering is pixel-for-pixel identical between training data and whatever
# gets saved next - no drift between "how training images look" and "how
# freshly recorded images look" is exactly the class of bug this project
# has already been bitten by once (see the num_hands=1/2 mismatch).
import cv2
import numpy as np

CANVAS_SIZE = 400

# Standard 21-point MediaPipe hand skeleton edges: 4 per finger (wrist ->
# tip) plus 3 palm-base connections between adjacent fingers' MCP joints.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),             # palm base
]


def draw_skeleton(canvas, pts_xy, offset_x=0, offset_y=0):
    """Draw a green 3px-line skeleton with red 2px joint dots onto canvas
    (mutated in place). pts_xy: 21 (x, y) pixel coordinates."""
    pts2d = [(int(round(x)) + offset_x, int(round(y)) + offset_y) for x, y in pts_xy]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, pts2d[a], pts2d[b], (0, 255, 0), 3)
    for x, y in pts2d:
        cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)
    return canvas


def render_skeleton_canvas(hands_pts_xy, size=CANVAS_SIZE, margin=40):
    """Build a fresh size x size white canvas with one or two hand
    skeletons centered and scaled to fit together. ISL fingerspelling mixes
    one-handed and two-handed letters (see isl_alphabet_reference.py), so
    this must handle either.

    hands_pts_xy: either a single hand's 21 (x, y) points, or a list of 1-2
    hands' 21-point sets, in any source scale (e.g. straight from a
    MediaPipe detection on a raw photo or webcam frame) - rescaled here to
    fit the canvas consistently regardless of original resolution or hand
    size, and regardless of one hand vs two.
    """
    hands_pts_xy = np.asarray(hands_pts_xy, dtype=np.float32)
    if hands_pts_xy.ndim == 2:
        hands_pts_xy = hands_pts_xy[np.newaxis, ...]  # single hand -> (1, 21, 2)

    all_pts = hands_pts_xy.reshape(-1, 2)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    span = max(x_max - x_min, y_max - y_min, 1e-6)
    scale = (size - 2 * margin) / span
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    canvas = np.full((size, size, 3), 255, np.uint8)
    for hand_pts in hands_pts_xy:
        scaled = [((x - cx) * scale + size / 2, (y - cy) * scale + size / 2) for x, y in hand_pts]
        draw_skeleton(canvas, scaled)
    return canvas
