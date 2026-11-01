# state_machine.py - Letter acceptance, word boundaries, and sentence flushing
import time


class ISLStateMachine:
    """Turns a stream of per-frame letter predictions into words and sentences.

    - A letter is accepted once the same confident prediction holds for
      debounce_seconds. The same letter can't repeat until confidence dips
      or the hands leave the frame, so a held sign isn't entered twice.
    - Hands out of frame for word_gap_seconds ends the current word.
    - Hands out of frame for sentence_gap_seconds ends the sentence:
      hand_absent() returns the finished words for the NLP step.

    Timing is wall-clock based, since the frontend's frame rate depends on
    the viewer's device.
    """

    def __init__(self, confidence_threshold=0.8,
                 debounce_seconds=0.35,
                 low_confidence_seconds=0.2,
                 gap_reset_seconds=0.5,
                 word_gap_seconds=1.0,
                 sentence_gap_seconds=3.0):
        self.confidence_threshold = confidence_threshold
        self.debounce_seconds = debounce_seconds
        self.low_confidence_seconds = low_confidence_seconds
        self.gap_reset_seconds = gap_reset_seconds
        self.word_gap_seconds = word_gap_seconds
        self.sentence_gap_seconds = sentence_gap_seconds
        self.reset()

    def reset(self):
        self.words = []
        self.letters = []
        self.hand_absent_since = None
        self.last_hand_time = None
        self._clear_debounce()

    def _clear_debounce(self):
        self.last_accepted = None
        self.stable_prediction = None
        self.stable_since = None
        self.low_confidence_since = None

    def _close_word(self):
        if self.letters:
            self.words.append(''.join(self.letters))
            self.letters = []

    def update(self, predicted_label, confidence, now=None):
        """Feed one frame with hands visible. Returns True if a letter was
        accepted this frame."""
        now = time.time() if now is None else now
        self.hand_absent_since = None

        # Frames stopped arriving without explicit hand-absent frames (e.g. a
        # dropped connection): treat it like the hands were lowered, so the
        # same letter can be signed again.
        if self.last_hand_time is not None and now - self.last_hand_time > self.gap_reset_seconds:
            self._clear_debounce()
        self.last_hand_time = now

        if confidence < self.confidence_threshold:
            if self.low_confidence_since is None:
                self.low_confidence_since = now
            self.stable_prediction = None
            self.stable_since = None
            if now - self.low_confidence_since >= self.low_confidence_seconds:
                self.last_accepted = None
            return False

        self.low_confidence_since = None
        if predicted_label != self.stable_prediction:
            self.stable_prediction = predicted_label
            self.stable_since = now

        if (now - self.stable_since >= self.debounce_seconds
                and predicted_label != self.last_accepted
                and predicted_label != 'IDLE'):
            self.last_accepted = predicted_label
            self.letters.append(predicted_label)
            return True
        return False

    def hand_absent(self, now=None):
        """Feed one frame with no hands visible. Returns the finished
        sentence's words once the sentence gap elapses, otherwise None.

        The gap is measured from the last frame that had hands, so a dropped
        or coalesced "no hands" message still counts the real absence.
        """
        now = time.time() if now is None else now
        if self.hand_absent_since is None:
            self.hand_absent_since = self.last_hand_time if self.last_hand_time is not None else now
            self._clear_debounce()

        elapsed = now - self.hand_absent_since
        if elapsed >= self.word_gap_seconds:
            self._close_word()
        if elapsed >= self.sentence_gap_seconds and self.words:
            words = self.words
            self.reset()
            return words
        return None

    def finish(self):
        """Close the current word and return whatever was spelled. Used when
        the signer presses Stop, which ends the camera instead of sending
        more "no hands" frames."""
        self._close_word()
        words = self.words[:]
        self.reset()
        return words

    def get_sentence_glosses(self):
        """Finished words plus the word currently being spelled."""
        return self.words + ([''.join(self.letters)] if self.letters else [])

    @property
    def current_word(self):
        return ''.join(self.letters)
