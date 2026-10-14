# preprocessor.py - Wrist-relative, scale-normalized feature encoding
#
# Not imported by src/main.py - the frontend (landmarkNormalizer.js) already
# normalizes before sending. Kept correct and in sync with that encoding for
# any offline tooling that needs to reproduce it server-side.
import numpy as np

FEATURES_PER_HAND = 21 * 3  # (x, y, z)
FEATURES = FEATURES_PER_HAND * 2  # left + right


def normalize_hand(landmarks_21):
    """Wrist-center and scale-normalize one hand's 21 landmarks.

    Input: (21, 3) - [x, y, z] coordinates
    Output: (63,) - wrist-relative, unit-scale-normalized coordinates
    """
    lm = np.array(landmarks_21, dtype=np.float32)
    wrist = lm[0]
    centered = lm - wrist
    scale = np.max(np.linalg.norm(centered, axis=1)) + 1e-6
    return (centered / scale).flatten()


def normalize_landmarks(left_hand_21, right_hand_21):
    """Build the 126-dim model input from up to two hands.

    Pass None for a hand that wasn't detected; its slot is zero-filled.
    Order matches extract_kaggle_dataset.py / landmarkNormalizer.js: left
    hand first, right hand second.
    """
    left = normalize_hand(left_hand_21) if left_hand_21 is not None else np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right = normalize_hand(right_hand_21) if right_hand_21 is not None else np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    return np.concatenate([left, right])


def reshape_for_model(feature_vector):
    """Add the batch dimension for a single-frame model input.

    Input: (126,)
    Output: (1, 126)
    """
    return np.asarray(feature_vector, dtype=np.float32).reshape(1, FEATURES)