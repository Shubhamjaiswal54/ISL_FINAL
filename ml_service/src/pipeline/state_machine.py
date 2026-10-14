# state_machine.py - Confidence thresholding, debounce, idle detection
import time


class ISLStateMachine:
    """State machine for ISL gesture decoding and sentence construction.

    Handles:
    - Confidence thresholding for word acceptance
    - Debounce window to prevent identical repeated predictions
    - Idle detection to flush accumulated words to NLP engine

    Timing is wall-clock based (seconds), not frame-counted. The frontend
    detects on an uncapped requestAnimationFrame loop, so "frames" arrive at
    whatever rate the viewer's display refreshes (60Hz, 120Hz, ...) - a
    frame-count threshold silently means a different amount of real time for
    every viewer. Seconds don't have that problem.
    """

    def __init__(self, confidence_threshold=0.85,
                 debounce_seconds=0.4,
                 idle_seconds=2.5,
                 low_confidence_seconds=0.2,
                 gap_reset_seconds=0.5):
        self.confidence_threshold = confidence_threshold
        self.debounce_seconds = debounce_seconds
        self.idle_seconds = idle_seconds
        self.low_confidence_seconds = low_confidence_seconds
        # How long between update() calls counts as "the hand left and came
        # back" rather than normal continuous frames.
        self.gap_reset_seconds = gap_reset_seconds

        # State variables
        self.word_list = []  # Accumulated gloss words
        self.last_prediction = None
        self.is_idle = False
        self.low_confidence_since = None
        self.stable_prediction = None
        self.stable_since = None
        self.idle_since = None
        self.last_update_time = None

    def update(self, predicted_label, confidence, now=None):
        """Process a new prediction and update state.

        Args:
            predicted_label: String class label (e.g., 'A', 'IDLE')
            confidence: Float confidence score from model
            now: Optional timestamp (seconds); defaults to time.time()

        Returns:
            Boolean: True if a word was accepted this frame
        """
        now = time.time() if now is None else now
        accepted = False

        # update() only ever runs when the frontend actually sends a frame,
        # and it sends nothing at all while no hand is detected - so a long
        # real-world pause (hand lowered, then raised again) leaves
        # last_prediction/stable_prediction frozen at their pre-pause values
        # instead of naturally resetting. Without this, re-signing the exact
        # same letter you last signed - after any pause long enough for the
        # hand to leave frame - would silently fail to register, since
        # nothing ran during the gap to clear last_prediction the way a
        # genuine low-confidence frame would have.
        if self.last_update_time is not None and now - self.last_update_time > self.gap_reset_seconds:
            self.stable_prediction = None
            self.stable_since = None
            self.last_prediction = None
        self.last_update_time = now

        if confidence < self.confidence_threshold:
            if self.low_confidence_since is None:
                self.low_confidence_since = now
            self.stable_prediction = None
            self.stable_since = None
            if now - self.low_confidence_since >= self.low_confidence_seconds:
                # A short release/pause allows the same letter to be entered again.
                self.last_prediction = None
            return False

        self.low_confidence_since = None
        if predicted_label != self.stable_prediction:
            self.stable_prediction = predicted_label
            self.stable_since = now

        # Require a stable prediction held for debounce_seconds before adding one letter.
        if now - self.stable_since >= self.debounce_seconds and predicted_label != self.last_prediction:
            self.last_prediction = predicted_label
            if predicted_label != 'IDLE':
                self.word_list.append(predicted_label)
                accepted = True

        # Idle detection: if IDLE class detected continuously for idle_seconds
        if predicted_label == 'IDLE':
            if self.idle_since is None:
                self.idle_since = now
            if now - self.idle_since >= self.idle_seconds:
                # Flush the word list to NLP engine
                self.is_idle = True
                self.word_list = []
                self.idle_since = None
                self.last_prediction = None
        else:
            self.idle_since = None  # Reset idle timer on non-IDLE

        return accepted

    def get_sentence_glosses(self):
        """Get the current accumulated word list.

        Returns:
            list of strings - the accumulated gloss sequence
        """
        return self.word_list.copy()

    def reset(self):
        """Reset the state machine."""
        self.word_list = []
        self.last_prediction = None
        self.is_idle = False
        self.low_confidence_since = None
        self.stable_prediction = None
        self.stable_since = None
        self.idle_since = None
        self.last_update_time = None
