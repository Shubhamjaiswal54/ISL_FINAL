# control_signs.py - Space and backspace, read from the hand pose
#
# These are not letters. They are held still so a letter in the middle of
# being formed does not fire them.
#   space     both hands open, fingers spread
#   backspace one hand, index finger pointing to the left of the camera picture
import numpy as np

HOLD_SECONDS = 0.55

_FINGERS = (
    (8, 6),    # index tip, pip
    (12, 10),  # middle
    (16, 14),  # ring
    (20, 18),  # pinky
)


def _distance(hand, a, b):
    return float(np.linalg.norm(hand[a, :2] - hand[b, :2]))


def _palm(hand):
    return max(_distance(hand, 0, 9), 1.0)


def _extended(hand, tip, joint):
    return _distance(hand, 0, tip) > _distance(hand, 0, joint) * 1.12


def _thumb_extended(hand):
    return _distance(hand, 0, 4) > _distance(hand, 0, 3) * 1.2


def _fingers_open(hand):
    return all(_extended(hand, tip, joint) for tip, joint in _FINGERS)


def _open_hand(hand):
    palm = _palm(hand)
    spread = _distance(hand, 8, 20)
    return _fingers_open(hand) and _thumb_extended(hand) and spread > palm * 0.7


def _point_left(hand):
    """Index finger toward the left edge of the picture the signer sees."""
    palm = _palm(hand)
    pointing = float(hand[8, 0]) < float(hand[0, 0]) - palm * 0.55
    others_curled = not any(_extended(hand, tip, joint) for tip, joint in _FINGERS[1:])
    thumb_down = not _thumb_extended(hand)
    return _extended(hand, 8, 6) and others_curled and thumb_down and pointing


def detect_control(hands):
    """Return 'space', 'backspace', or None."""
    if not hands:
        return None
    if len(hands) >= 2 and _open_hand(hands[0]) and _open_hand(hands[1]):
        return 'space'
    if len(hands) == 1 and _point_left(hands[0]):
        return 'backspace'
    return None


class ControlLatch:
    """Fire a held control sign once, then wait until the hands leave it."""

    def __init__(self, hold=HOLD_SECONDS):
        self.hold = hold
        self.name = None
        self.since = None
        self.fired = False

    def reset(self):
        self.name = None
        self.since = None
        self.fired = False

    def update(self, name, now):
        """Returns (pose is being held, it just completed the hold)."""
        if name != self.name:
            self.name = name
            self.since = now if name else None
            self.fired = False
        if not name:
            return False, False
        just_fired = False
        if not self.fired and now - self.since >= self.hold:
            self.fired = True
            just_fired = True
        return True, just_fired
