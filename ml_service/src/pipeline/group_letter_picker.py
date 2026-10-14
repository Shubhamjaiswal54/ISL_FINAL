# group_letter_picker.py - Pick the exact ISL letter within a predicted
# hand-shape group, using landmark coordinates - the "separate fixed
# geometry function" stage of the group-CNN pipeline.
#
# One small trained classifier per group (scripts/train_group_letter_pickers.py),
# not nearest-centroid distance or hand-tuned thresholds - centroid distance
# was tried first and validated at only 69% overall accuracy (some letters
# at 0%), because letters end up in the same group precisely because
# they're geometrically similar, so the same feature space that couldn't
# separate them for clustering also can't separate them by raw distance.
# A trained classifier per group reaches 99.93% on the same held-out data.
import json
import os

import numpy as np

from .letter_groups import GROUPS

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'group_pickers')


class GroupLetterPicker:
    """Given a predicted group index and this frame's engineered feature
    vector (126 raw + 16 derived = 142, from add_derived_features), returns
    the letter within that group the matching per-group classifier picks.
    Groups with only one letter (see letter_groups.GROUPS) skip inference
    entirely - there's nothing to disambiguate.
    """

    def __init__(self, models_dir=_MODELS_DIR):
        import tensorflow as tf  # local import: keeps this cheap to import when unused

        with open(os.path.join(models_dir, 'metadata.json')) as f:
            metadata = json.load(f)

        self._models = {}
        self._letters_by_group = {}
        for group_idx_str, info in metadata.items():
            group_idx = int(group_idx_str)
            self._letters_by_group[group_idx] = info['letters']
            if info['model'] is not None:
                self._models[group_idx] = tf.keras.models.load_model(
                    os.path.join(models_dir, info['model'])
                )

    def pick(self, group_idx, model_features_142):
        letters = self._letters_by_group.get(group_idx, GROUPS.get(group_idx))
        if letters is None:
            raise ValueError(f'Unknown group index: {group_idx}')
        if len(letters) == 1:
            return letters[0]

        model = self._models[group_idx]
        x = np.asarray(model_features_142, dtype=np.float32).reshape(1, -1)
        probs = model.predict(x, verbose=0)[0]
        return letters[int(np.argmax(probs))]
