# isl_alphabet_reference.py - Expected hand count per ISL letter
#
# Sourced from the user-provided ISL manual alphabet reference chart.
# ISL fingerspelling mixes one-handed and two-handed letters - it is NOT
# uniformly two-handed - so validation/filtering must check the expected
# count per letter, not a fixed "always 1" or "always 2" rule.
ONE_HANDED_LETTERS = {'C', 'I', 'L', 'O', 'U', 'V'}
TWO_HANDED_LETTERS = {'A', 'B', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R',
                       'S', 'T', 'W', 'X', 'Y', 'Z'}

EXPECTED_HAND_COUNT = {letter: 1 for letter in ONE_HANDED_LETTERS}
EXPECTED_HAND_COUNT.update({letter: 2 for letter in TWO_HANDED_LETTERS})

assert set(EXPECTED_HAND_COUNT) == set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), \
    'isl_alphabet_reference.py must cover all 26 letters exactly once'


def expected_hand_count(letter):
    return EXPECTED_HAND_COUNT.get(letter.upper())
