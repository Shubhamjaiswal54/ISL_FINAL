# letter_groups.py - ISL letter -> hand-shape group mapping
#
# Derived empirically from this project's own two-handed ISL landmark data
# (per-letter centroids in the 142-dim raw+derived feature space,
# agglomerative clustering into 8 groups) - NOT copied from an ASL reference
# app's groupings. ASL and ISL are different sign languages with different
# letterforms for a meaningful number of letters, so assuming ASL's
# handshape families transfer to ISL would silently mislabel whichever
# letters don't actually share a handshape family in ISL. Regenerate by
# re-running scripts/derive_letter_groups.py if the underlying landmark data
# changes meaningfully (e.g. after adding a lot more real recordings).
#
# NOTE: this clustering (run 2026-08-29, on the real two-handed dataset
# covering all 26 letters) came out lopsided - one 14-letter catch-all group
# vs. several singletons - unlike the older one-handed clustering's more
# even split. The two-handed 142-dim feature space apparently doesn't
# separate into 8 balanced clusters as cleanly. The per-group letter picker
# for group 1 still has to do real work (14-way), so it's not free accuracy
# the way singleton groups are - but it's still a real result from real data,
# not a bug to silently "fix" by re-clustering until it looks nicer.
GROUPS = {
    0: ['B', 'Z'],
    1: ['A', 'D', 'E', 'F', 'H', 'J', 'M', 'N', 'P', 'R', 'S', 'T', 'X', 'Y'],
    2: ['G', 'W'],
    3: ['O'],
    4: ['C', 'I', 'U', 'V'],
    5: ['L'],
    6: ['Q'],
    7: ['K'],
}

LETTER_TO_GROUP = {letter: group for group, letters in GROUPS.items() for letter in letters}

NUM_GROUPS = len(GROUPS)
