# motion_gate.py - Rejects mid-transition frames using landmark speed
#
# The classifier only ever saw held poses during training, so a frame
# captured while the hands move between two letters sits outside anything it
# learned and can produce a confident wrong prediction. Rather than trust it,
# we track how fast the landmarks are moving and only let the prediction
# through once the pose settles.
import numpy as np


class MotionGate:
    """Tracks landmark speed in the renderer's unit square (see
    skeleton_render.canonical_points), so the threshold is independent of
    hand size and distance from the camera. Moving the whole pose without
    changing its shape doesn't count as motion - the rendered canvas, and so
    the prediction, doesn't change either."""

    def __init__(self, speed_threshold=0.6, ema_alpha=0.5):
        self.speed_threshold = speed_threshold
        self.ema_alpha = ema_alpha
        self.reset()

    def reset(self):
        self.prev_points = None
        self.prev_time = None
        self.speed = 0.0

    def update(self, canonical_pts, now):
        """canonical_pts: (num_hands, 21, 2). Returns True if the pose is
        settled. The first frame, and any frame where the number of hands
        changed, counts as moving - there is no speed signal yet."""
        canonical_pts = np.asarray(canonical_pts, dtype=np.float32)
        if self.prev_points is None or self.prev_points.shape != canonical_pts.shape:
            self.prev_points, self.prev_time, self.speed = canonical_pts, now, 0.0
            return False

        dt = max(now - self.prev_time, 1e-3)
        displacement = float(np.linalg.norm(canonical_pts - self.prev_points, axis=-1).mean())
        self.speed = self.ema_alpha * (displacement / dt) + (1 - self.ema_alpha) * self.speed
        self.prev_points, self.prev_time = canonical_pts, now
        return self.speed < self.speed_threshold
