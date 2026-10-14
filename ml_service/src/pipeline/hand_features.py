# hand_features.py - Derived geometric features (finger curl, thumb/palm relations)
#
# Raw landmark coordinates only implicitly encode how bent a finger is or
# where the thumb sits relative to the palm - the exact properties that
# separate visually similar closed-fist letters (A/C/I/M/O). Computing them
# explicitly gives the classifier a much more direct signal for that specific
# confusion cluster. Applied identically at training time (2_preprocess_and_augment.py)
# and inference time (src/main.py) so the model always sees the same feature space.
import numpy as np

RAW_FEATURES = 126  # 21 landmarks * (x, y, z) * 2 hands - what the frontend sends
RAW_FEATURES_PER_HAND = 63
DERIVED_FEATURES_PER_HAND = 8  # 5 finger-curl angles + 3 distances
MODEL_FEATURES = RAW_FEATURES + DERIVED_FEATURES_PER_HAND * 2  # 142 - what the model consumes

# MediaPipe hand landmark indices for each finger's (MCP, PIP/IP, TIP)
_FINGER_JOINTS = {
    'thumb': (1, 2, 4),
    'index': (5, 6, 8),
    'middle': (9, 10, 12),
    'ring': (13, 14, 16),
    'pinky': (17, 18, 20),
}
_FINGERTIPS = (4, 8, 12, 16, 20)
_PALM_POINTS = (5, 9, 13, 17)  # index/middle/ring/pinky MCPs


def _angle(a, b, c):
    """Angle at point b (radians), between vectors b->a and b->c."""
    ba = a - b
    bc = c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-8 or nc < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def derive_hand_features(hand_raw_63):
    """Compute 8 extra geometric features from one hand's 63-dim (21x3) vector.

    Returns zeros if the hand slot is empty (all-zero input, i.e. that hand
    wasn't detected this frame).
    """
    hand_raw_63 = np.asarray(hand_raw_63, dtype=np.float32)
    if not np.any(hand_raw_63):
        return np.zeros(DERIVED_FEATURES_PER_HAND, dtype=np.float32)

    pts = hand_raw_63.reshape(21, 3)

    curls = np.array([
        _angle(pts[mcp], pts[mid], pts[tip])
        for mcp, mid, tip in _FINGER_JOINTS.values()
    ], dtype=np.float32)

    palm_center = pts[list(_PALM_POINTS)].mean(axis=0)
    thumb_tip, index_tip = pts[4], pts[8]
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
    thumb_palm_dist = np.linalg.norm(thumb_tip - palm_center)

    tips = pts[list(_FINGERTIPS[1:])]  # index/middle/ring/pinky tips (exclude thumb)
    spread = max(
        np.linalg.norm(tips[i] - tips[j])
        for i in range(len(tips)) for j in range(i + 1, len(tips))
    )

    return np.concatenate([curls, [thumb_index_dist, thumb_palm_dist, spread]]).astype(np.float32)


def add_derived_features(raw_vector_126):
    """(126,) raw landmarks -> (142,) with derived features appended per hand.

    Layout: [raw_126, left_hand_derived_8, right_hand_derived_8]
    """
    raw_vector_126 = np.asarray(raw_vector_126, dtype=np.float32)
    left_extra = derive_hand_features(raw_vector_126[:RAW_FEATURES_PER_HAND])
    right_extra = derive_hand_features(raw_vector_126[RAW_FEATURES_PER_HAND:RAW_FEATURES])
    return np.concatenate([raw_vector_126, left_extra, right_extra]).astype(np.float32)
