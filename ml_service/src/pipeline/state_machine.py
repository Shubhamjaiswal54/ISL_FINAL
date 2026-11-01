# state_machine.py - Letter acceptance, word boundaries, and sentence flushing
import time


class ISLStateMachine:
    """Turns a stream of per-frame letter predictions into words.

    - A letter is accepted once the same confident prediction holds for
      debounce_seconds. The same letter can't repeat until confidence dips
      or the hands leave the frame, so a held sign isn't entered twice.
    - Space commits the letters spelled so far as one word.
    - Speak is what turns those words into a sentence. Lowering the hands
      does not end a word or a sentence.

    Timing is wall-clock based, since the frontend's frame rate depends on
    the viewer's device.
    """

    def __init__(self, confidence_threshold=0.5,
                 debounce_seconds=0.35,
                 low_confidence_seconds=0.2,
                 gap_reset_seconds=0.5):
        self.confidence_threshold = confidence_threshold
        self.debounce_seconds = debounce_seconds
        self.low_confidence_seconds = low_confidence_seconds
        self.gap_reset_seconds = gap_reset_seconds
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
        """Hands left the frame. The same letter can be signed again.

        This does not end the word or the sentence. Space and Speak do that.
        """
        now = time.time() if now is None else now
        if self.hand_absent_since is None:
            self.hand_absent_since = self.last_hand_time if self.last_hand_time is not None else now
            self._clear_debounce()
        return None

    def space(self, chosen=None):
        """Commit the letters spelled so far.

        The stored word is the spelling itself. `chosen` is used only when
        the signer taps a suggestion.
        """
        from src.pipeline.spell_suggest import commit_spelling, suggest_words
        if not self.letters:
            return False
        raw = ''.join(self.letters)
        allowed = {raw.lower(), *suggest_words(raw)}
        if isinstance(chosen, str) and chosen.strip().lower() in allowed:
            word = chosen.strip().lower()
        else:
            word = commit_spelling(raw)
        self.last_accepted = self.letters[-1]
        self.letters = []
        self.stable_prediction = None
        self.stable_since = None
        if word:
            self.words.append(word)
        return True

    def accept_word(self):
        """Thumbs up: keep the word being spelled, not a suggestion."""
        return self.space()

    def delete_character(self):
        """Backspace: drop the last letter, or the last word if none is open."""
        if self.letters:
            self.last_accepted = self.letters.pop()
            self.stable_prediction = None
            self.stable_since = None
            return True
        if self.words:
            self.words.pop()
            return True
        return False

    def finish(self):
        """Close the current word and return whatever was spelled, then clear
        the session. The Stop button does not call this."""
        self._close_word()
        words = self.words[:]
        self.reset()
        return words

    def backspace(self, index=None):
        """Remove one word from the predicted list.

        With no index, the last word goes. That last entry is the word still
        being spelled, when there is one. The letter that was just removed
        stays blocked until the pose changes, so a sign the user is still
        holding does not type itself again.
        """
        total = len(self.words) + (1 if self.letters else 0)
        if total == 0:
            return False
        if type(index) is not int or index < 0 or index >= total:
            index = total - 1
        if self.letters and index == len(self.words):
            self.last_accepted = self.letters[-1]
            self.letters = []
            self.stable_prediction = None
            self.stable_since = None
            return True
        del self.words[index]
        return True

    def get_sentence_glosses(self):
        """Finished words plus the word currently being spelled."""
        return self.words + ([''.join(self.letters)] if self.letters else [])

    @property
    def current_word(self):
        return ''.join(self.letters)
