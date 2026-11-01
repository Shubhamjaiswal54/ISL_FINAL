# spell_suggest.py - nearest English word for a fingerspelled token
#
# Word list: google-10000-english-usa-no-swears, most common first.
# https://github.com/first20hours/google-10000-english (public domain)
import os
from functools import lru_cache

_WORDS_PATH = os.path.join(os.path.dirname(__file__), 'common_words.txt')


def _load():
    order = []
    rank = {}
    with open(_WORDS_PATH, encoding='utf-8') as handle:
        for line in handle:
            word = line.strip().lower()
            if word and word not in rank:
                rank[word] = len(order)
                order.append(word)
    by_length = {}
    by_first = {}
    for word in order:
        by_length.setdefault(len(word), []).append(word)
        by_first.setdefault(word[0], []).append(word)
    return rank, by_length, by_first


RANK, BY_LENGTH, BY_FIRST = _load()


def _levenshtein(left, right, limit):
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        row_min = i
        for j, right_char in enumerate(right, 1):
            cost = 0 if left_char == right_char else 1
            best = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost)
            current.append(best)
            if best < row_min:
                row_min = best
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _letters(raw):
    return ''.join(char for char in str(raw).lower() if char.isalpha())


# Enough to scroll through, not the whole dictionary on a one-letter prefix.
_MAX_SUGGESTIONS = 80


@lru_cache(maxsize=2048)
def suggest_words(raw):
    """Autocomplete strip for the word being spelled.

    Every dictionary word that starts with the spelling is included, most
    common first, the way a phone keyboard lists completions. A misspelling
    with no such words, such as "hallo", lists the nearest real words instead,
    with the same-length correction ("hello") first.
    """
    word = _letters(raw)
    if not word or any(char.isdigit() for char in str(raw)):
        return []

    prefix_hits = [candidate for candidate in BY_FIRST.get(word[0], []) if candidate.startswith(word)]
    prefix_hits.sort(key=lambda candidate: RANK[candidate])
    # "hel" continues into help and hello. "hallo" only continues into halloween,
    # which is the typo plus several extra letters, so that should not outrank hello.
    close_prefixes = [candidate for candidate in prefix_hits if len(candidate) - len(word) <= 2]

    corrections = []
    seen = set(prefix_hits)
    for candidate in BY_FIRST.get(word[0], []):
        if candidate in seen or abs(len(candidate) - len(word)) > 1:
            continue
        if _levenshtein(word, candidate, 1) <= 1:
            corrections.append(candidate)
            seen.add(candidate)
    corrections.sort(key=lambda candidate: (abs(len(candidate) - len(word)), RANK[candidate]))

    if not close_prefixes and not corrections and len(word) >= 5:
        for length in range(len(word) - 2, len(word) + 3):
            for candidate in BY_LENGTH.get(length, []):
                if candidate in seen:
                    continue
                if _levenshtein(word, candidate, 2) <= 2:
                    corrections.append(candidate)
                    seen.add(candidate)
        corrections.sort(key=lambda candidate: (abs(len(candidate) - len(word)), RANK[candidate]))

    if word not in RANK and corrections and not close_prefixes:
        ordered = corrections + prefix_hits
    else:
        # Once help and hello are on the strip, a shorter typo such as "he" is noise.
        ordered = prefix_hits + [candidate for candidate in corrections if len(candidate) >= len(word)]
    return ordered[:_MAX_SUGGESTIONS]


def commit_spelling(raw):
    """The spelled letters, unchanged. A suggestion is used only if the signer taps it."""
    text = ''.join(str(raw).split())
    if any(char.isdigit() for char in text):
        return text.lower()
    return _letters(text)
