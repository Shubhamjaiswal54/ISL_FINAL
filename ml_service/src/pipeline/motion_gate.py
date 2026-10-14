# motion_gate.py - Rejects mid-transition frames using frame-to-frame landmark velocity
#
# The classifier only ever saw held static poses during training, so a frame
# captured while the hand is moving between two letters (e.g. spelling "CAT")
# sits outside anything it learned and produces an arbitrary, sometimes
# high-confidence, wrong prediction. Gating on motion instead of trying to
# make the model itself robust to transitions means no new training data is
# needed - we simply stop trusting the classifier's output while the hand is
# demonstrably still moving, and let it through once the pose settles.
import numpy as np


class MotionGate:
    """Tracks frame-to-frame hand velocity to detect mid-transition motion.

    Landmarks arriving here are already wrist-relative and unit-scale
    normalized per hand (see preprocessor.py / landmarkNormalizer.js), so a
    fixed velocity_threshold is comparable across users, hand sizes, and
    distance from the camera.
    """

    def __init__(self, velocity_threshold=0.06, ema_alpha=0.5):
        self.velocity_threshold = velocity_threshold
        self.ema_alpha = ema_alpha
        self.prev_landmarks = None
        self.velocity = 0.0

    def update(self, raw_landmarks_126):
        """Feed the current frame's raw landmark vector.

        Returns True if the hand is settled (safe to trust the classifier),
        False if it's still in transition (first frame counts as transition -
        there's no velocity signal yet).
        """
        raw_landmarks_126 = np.asarray(raw_landmarks_126, dtype=np.float32)
        if self.prev_landmarks is None:
            self.prev_landmarks = raw_landmarks_126
            self.velocity = 0.0
            return False

        frame_delta = float(np.linalg.norm(raw_landmarks_126 - self.prev_landmarks))
        self.velocity = self.ema_alpha * frame_delta + (1 - self.ema_alpha) * self.velocity
        self.prev_landmarks = raw_landmarks_126
        return self.velocity < self.velocity_threshold

    def reset(self):
        self.prev_landmarks = None
        self.velocity = 0.0
