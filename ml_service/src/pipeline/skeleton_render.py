# skeleton_render.py - Render one or two hand skeletons onto a plain white canvas
#
# This is the single source of truth for what the skeleton CNN sees. The
# offline dataset builder (scripts/process_original_images.py), the trainer
# (scripts/train_skeleton_cnn.py) and live inference (src/main.py) all call
# render_hands(), so training images and live frames are pixel-for-pixel
# identical given the same landmarks.
#
# ISL fingerspelling is mostly two-handed, and what separates many letters is
# where one hand touches the other. Both hands are therefore drawn together
# on one canvas, scaled as a unit so their relative position is preserved.
# Hands are ordered by on-screen position (leftmost wrist first) rather than
# MediaPipe's Left/Right handedness label, because that label is unreliable
# exactly when the hands touch or cross - which is most ISL letters.
import cv2
import numpy as np

IMG_SIZE = 128
MARGIN_RATIO = 0.1

# BGR. One color per position-ordered hand so the CNN can tell which fingers
# belong to which hand where the two skeletons overlap.
HAND_COLORS = [(0, 160, 0), (200, 80, 0)]
JOINT_COLOR = (0, 0, 255)

# Standard 21-point MediaPipe hand skeleton edges: 4 per finger (wrist ->
# tip) plus 3 palm-base connections between adjacent fingers' MCP joints.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm base
]


def _as_hand_list(hands):
    """Accept one hand (21, >=2) or a sequence of hands (k, 21, >=2).

    Returns a list of (21, 3) float32 arrays. A missing z column is filled
    with zeros; MediaPipe's z is smaller (more negative) for the hand closer
    to the camera.
    """
    if hands is None or len(hands) == 0:
        return []
    arr = np.asarray(hands, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.ndim != 3 or arr.shape[1] != 21 or arr.shape[2] < 2:
        raise ValueError(f'Expected hands shaped (k, 21, 2+), got {arr.shape}')
    if arr.shape[2] == 2:
        zeros = np.zeros((arr.shape[0], 21, 1), dtype=np.float32)
        arr = np.concatenate([arr, zeros], axis=2)
    return [hand[:, :3].astype(np.float32) for hand in arr]


def order_hands(hands):
    """Sort hands left-to-right by wrist x position."""
    return sorted(hands, key=lambda hand: float(hand[0, 0]))


def _hand_distance(a, b):
    return float(np.linalg.norm(a[:, :2] - b[:, :2]))


def _assign_to(screen, reference):
    """Put the two screen-sorted hands into reference's slot order."""
    keep = _hand_distance(screen[0], reference[0]) + _hand_distance(screen[1], reference[1])
    swap = _hand_distance(screen[0], reference[1]) + _hand_distance(screen[1], reference[0])
    return [screen[1], screen[0]] if swap < keep else list(screen)


class HandOrderStabilizer:
    """Hold hand identity steady while the wrists cross.

    Each training photo is ordered left-to-right once. Sorting every live
    frame the same way swaps the two colors whenever the wrists pass or
    merely jitter past each other, so the picture flickers and the letter
    never settles. While the wrists are close, each hand is followed from
    where it was last frame (and where it was heading). Once the wrists are
    clearly apart again, left-to-right order is used, which is what the
    training photos use.
    """

    def __init__(self, margin_ratio=0.15):
        self.margin_ratio = margin_ratio
        self.prev = None
        self.prev2 = None
        self.holding = False

    def reset(self):
        self.prev = None
        self.prev2 = None
        self.holding = False

    def apply(self, hands):
        hands = _as_hand_list(hands)
        if len(hands) != 2:
            self.reset()
            return hands

        screen = order_hands(hands)
        flat = np.concatenate([hand[:, :2] for hand in screen])
        span = max(float((flat.max(axis=0) - flat.min(axis=0)).max()), 1e-6)
        gap = float(screen[1][0, 0] - screen[0][0, 0])
        # Wider margin to leave than to enter, so a wrist sitting on the
        # boundary cannot swap the slots on every frame.
        limit = span * (self.margin_ratio * 2 if self.holding else self.margin_ratio)
        if self.prev is None or gap >= limit:
            self.holding = False
            ordered = screen
        else:
            self.holding = True
            reference = self.prev
            if self.prev2 is not None:
                reference = [2 * previous - before for previous, before in zip(self.prev, self.prev2)]
            ordered = _assign_to(screen, reference)
        self.prev2 = self.prev
        self.prev = [hand.copy() for hand in ordered]
        return ordered


def canonical_points(hands, margin_ratio=MARGIN_RATIO, ordered=False):
    """Map hands into a shared unit square.

    hands: one or two hands' 21 points in any consistent pixel scale (x and y
    must use the same units, e.g. lm.x * width, lm.y * height).
    Returns (num_hands, 21, 2) in [margin, 1 - margin], scaled as a unit so
    the hands' relative position and size are preserved. z is not included;
    the caller uses it only to decide which hand is drawn on top.

    Pass ordered=True when the caller already applied HandOrderStabilizer,
    so a held identity is not immediately re-sorted away.
    """
    hands = _as_hand_list(hands)
    if not ordered:
        hands = order_hands(hands)
    if not hands:
        return np.zeros((0, 21, 2), dtype=np.float32)

    stacked = np.stack([hand[:, :2] for hand in hands])
    flat = stacked.reshape(-1, 2)
    mins, maxs = flat.min(axis=0), flat.max(axis=0)
    span = max(float((maxs - mins).max()), 1e-6)
    center = (mins + maxs) / 2
    scale = (1 - 2 * margin_ratio) / span
    return ((stacked - center) * scale + 0.5).astype(np.float32)


def render_hands(hands, size=IMG_SIZE, margin_ratio=MARGIN_RATIO, ordered=False):
    """Draw one or two hand skeletons on a size x size white BGR canvas.

    The hand with the larger mean z (farther from the camera) is drawn
    first, so the nearer hand stays visible where the two overlap. Color
    still follows slot order, not draw order.
    """
    canvas = np.full((size, size, 3), 255, np.uint8)
    raw = _as_hand_list(hands)
    if not ordered:
        raw = order_hands(raw)
    if not raw:
        return canvas

    points = canonical_points(raw, margin_ratio, ordered=True) * size
    thickness = max(1, round(size / 64))
    radius = max(1, round(size / 64))
    # MediaPipe z is smaller for the closer hand. Draw far hands first.
    draw_order = sorted(range(len(raw)), key=lambda i: float(raw[i][:, 2].mean()), reverse=True)

    for hand_idx in draw_order:
        color = HAND_COLORS[hand_idx % len(HAND_COLORS)]
        pts = [(int(round(x)), int(round(y))) for x, y in points[hand_idx]]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], color, thickness, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(canvas, (x, y), radius, JOINT_COLOR, -1, cv2.LINE_AA)
    return canvas


def render_skeleton_canvas(hands_pts_xy, size=400, margin=40):
    """Legacy entry point used by the older collection scripts."""
    return render_hands(hands_pts_xy, size=size, margin_ratio=margin / size)
