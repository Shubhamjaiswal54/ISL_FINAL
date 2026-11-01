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


@lru_cache(maxsize=2048)
def suggest_words(raw):
    """Up to three likely English words for a fingerspelled token.

    A real dictionary word is left alone. Otherwise a one-letter misspelling
    such as "hallo" suggests "hello", and a short prefix such as "hel"
    suggests words that start that way.
    """
    word = _letters(raw)
    if len(word) < 2 or any(char.isdigit() for char in str(raw)):
        return []
    if word in RANK:
        return []

    found = []
    seen = set()

    def add(candidate):
        if candidate in seen or candidate == word:
            return
        seen.add(candidate)
        found.append(candidate)

    def order(candidate):
        # A wrong letter ("hallo" -> "hello") beats a shorter, more common word.
        if len(candidate) == len(word):
            group = 0
        elif len(candidate) > len(word):
            group = 1
        else:
            group = 2
        return (group, RANK[candidate])

    for candidate in BY_FIRST.get(word[0], []):
        if abs(len(candidate) - len(word)) > 1:
            continue
        if _levenshtein(word, candidate, 1) <= 1:
            add(candidate)

    for candidate in BY_FIRST.get(word[0], []):
        if candidate.startswith(word) and 0 < len(candidate) - len(word) <= 2:
            add(candidate)

    if not found and len(word) >= 5:
        for length in range(len(word) - 2, len(word) + 3):
            for candidate in BY_LENGTH.get(length, []):
                if _levenshtein(word, candidate, 2) <= 2:
                    add(candidate)

    found.sort(key=order)
    return found[:3]


def commit_spelling(raw):
    """The word to store when Space is pressed."""
    text = ''.join(str(raw).split())
    if any(char.isdigit() for char in text):
        return text.lower()
    word = _letters(text)
    if not word:
        return ''
    options = suggest_words(text)
    if options:
        return options[0]
    return word
