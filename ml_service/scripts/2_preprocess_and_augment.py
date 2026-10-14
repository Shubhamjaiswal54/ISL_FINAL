# 2_preprocess_and_augment.py
# Single-frame dataset assembly: merge classes, geometric augmentation, class balancing
import sys
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline.hand_features import add_derived_features
from src.pipeline.isl_alphabet_reference import expected_hand_count

FEATURES = 126  # 21 landmarks * (x, y, z) * 2 hands - raw, on-disk sample format


def _load_class_samples(folder, label):
    """Load one class's raw 126-dim samples, dropping any whose detected
    hand count doesn't match what that letter actually requires.

    ISL fingerspelling mixes one-handed and two-handed letters (see
    isl_alphabet_reference.py) - a sample with the wrong hand count for its
    letter (e.g. a stray second hand in frame during a one-handed letter, or
    a missed hand during a two-handed one) describes an input live inference
    can never correctly produce for that class, so it's discarded instead of
    trained on. `label` is None/non-letter (e.g. IDLE) skips this check -
    any detected hand(s) are accepted, only fully-empty samples are dropped.
    """
    need_hands = expected_hand_count(label)
    samples = []
    dropped = 0
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.npy'):
            continue
        sample = np.load(os.path.join(folder, filename)).astype(np.float32)
        if sample.shape != (FEATURES,):
            continue
        left_present = np.any(sample[:63])
        right_present = np.any(sample[63:126])
        num_hands = int(left_present) + int(right_present)
        if need_hands is not None:
            if num_hands != need_hands:
                dropped += 1
                continue
        elif num_hands == 0:
            dropped += 1
            continue
        samples.append(sample)
    if dropped:
        print(f'  {folder}: dropped {dropped} wrong-hand-count sample(s)')
    return samples


def load_data(alphabet_dir='data/alphabet_landmarks', extra_dir='data/raw_dataset',
              actions_path='data/actions.json'):
    """Load per-class single-frame samples.

    Letters A-Z come from `alphabet_dir` (Kaggle-derived, via
    extract_kaggle_dataset.py). Any class also (or only) collected live with
    1_collect_landmarks.py lives under `extra_dir/<label>/*.npy` - for a new
    class (e.g. IDLE) it's picked up automatically; for an existing letter
    it's merged in *alongside* the Kaggle images, so your own webcam samples
    actually get used instead of being shadowed by the Kaggle folder.
    """
    with open(actions_path) as f:
        labels = list(json.load(f))

    if os.path.isdir(extra_dir):
        for name in sorted(os.listdir(extra_dir)):
            folder = os.path.join(extra_dir, name)
            if not os.path.isdir(folder) or name.upper() in labels:
                continue
            if not any(f.endswith('.npy') for f in os.listdir(folder)):
                continue  # e.g. leftover raw Kaggle image folders, not collected samples
            labels.append(name.upper())

    X, y = [], []
    for idx, label in enumerate(labels):
        class_samples = []
        primary_folder = os.path.join(alphabet_dir, label)
        if os.path.isdir(primary_folder):
            class_samples.extend(_load_class_samples(primary_folder, label))
        extra_folder = os.path.join(extra_dir, label.lower())
        if os.path.isdir(extra_folder):
            class_samples.extend(_load_class_samples(extra_folder, label))
        X.extend(class_samples)
        y.extend([idx] * len(class_samples))

    return np.array(X, dtype=np.float32), np.array(y), labels


def _rotate_xy(coords_xy, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return coords_xy @ rot.T


def augment_sample(sample, rng, rotation_range=0.26, scale_range=0.1, noise_std=0.02):
    """Apply small rotation, scale, and coordinate jitter to one 126-dim sample.

    Landmarks are wrist-relative and scale-normalized per hand, ordered
    [left(63), right(63)]. This mimics the pose/distance/camera-angle
    variation a live webcam sees, which the static Kaggle image set doesn't
    capture on its own.
    """
    out = sample.copy()
    angle = rng.uniform(-rotation_range, rotation_range)
    scale = rng.uniform(1 - scale_range, 1 + scale_range)

    for start in (0, 63):
        hand = out[start:start + 63].reshape(21, 3)
        if not np.any(hand):
            continue  # empty hand slot - leave zeroed
        hand[:, :2] = _rotate_xy(hand[:, :2], angle)
        hand *= scale
        hand += rng.normal(0.0, noise_std, size=hand.shape).astype(np.float32)
        out[start:start + 63] = hand.flatten()

    return out


def _balanced_augmented_train_set(X_train, y_train, num_classes, augment_multiplier, rng):
    """Balance classes and multiply data via augmentation.

    Every class is grown to `target` samples: the real samples are kept
    as-is, and the shortfall is filled with augmented copies drawn (with
    replacement) from that class's real samples. `target` itself scales
    with `augment_multiplier`, so even the majority class gets extra
    augmented variety instead of just being repeated verbatim.
    """
    per_class = {c: X_train[y_train == c] for c in range(num_classes)}
    non_empty = [s for s in per_class.values() if len(s) > 0]
    if not non_empty:
        raise RuntimeError('No training samples found for any class')
    target = max(len(s) for s in non_empty) * augment_multiplier

    X_out, y_out = [], []
    for cls, samples in per_class.items():
        if len(samples) == 0:
            continue
        X_out.extend(samples)
        y_out.extend([cls] * len(samples))

        needed = target - len(samples)
        if needed > 0:
            picks = rng.integers(0, len(samples), size=needed)
            X_out.extend(augment_sample(samples[i], rng) for i in picks)
            y_out.extend([cls] * needed)

    return np.array(X_out, dtype=np.float32), np.array(y_out)


def _synthesize_transition_samples(per_class_real, class_pool, num_samples, rng, alpha_range=(0.3, 0.7)):
    """Synthesize new samples by blending two different letters' real
    samples - a hand mid-transition between two signs isn't well
    represented by either sign alone or by jittered copies of a resting
    pose, but a blend of two real letters is geometrically close to what
    that transition actually looks like. Requires no new recording: built
    entirely from data already collected, labeled by the caller (IDLE).
    """
    synthetic = []
    for _ in range(num_samples):
        c1, c2 = rng.choice(class_pool, size=2, replace=False)
        pool1, pool2 = per_class_real[c1], per_class_real[c2]
        s1 = pool1[rng.integers(0, len(pool1))]
        s2 = pool2[rng.integers(0, len(pool2))]
        alpha = rng.uniform(*alpha_range)
        synthetic.append(((1 - alpha) * s1 + alpha * s2).astype(np.float32))
    return np.array(synthetic, dtype=np.float32)


def _compute_ood_stats(X_real, y_real, num_classes, percentile=99, margin=1.3):
    """Per-class nearest-centroid distance stats, for rejecting inputs that
    don't resemble any trained letter (e.g. a resting hand, mid-transition,
    or any pose that isn't a real sign). Computed on real (unaugmented)
    samples only, so it reflects genuine pose variation, not synthetic noise.
    """
    centroids = np.zeros((num_classes, X_real.shape[1]), dtype=np.float32)
    thresholds = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        samples = X_real[y_real == c]
        if len(samples) == 0:
            continue
        centroid = samples.mean(axis=0)
        centroids[c] = centroid
        dists = np.linalg.norm(samples - centroid, axis=1)
        thresholds[c] = float(np.percentile(dists, percentile)) if len(dists) > 1 else float(dists.max())
    return {
        'centroids': centroids.tolist(),
        'thresholds': thresholds.tolist(),
        'margin': margin,
    }


def preprocess_dataset(alphabet_dir='data/alphabet_landmarks', extra_dir='data/raw_dataset',
                        actions_path='data/actions.json', output_dir='data/processed',
                        models_dir='models', test_size=0.15, augment_multiplier=5, seed=42):
    X, y, labels = load_data(alphabet_dir, extra_dir, actions_path)
    if len(X) == 0:
        raise RuntimeError(f'No samples found under {alphabet_dir} or {extra_dir}')

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    rng = np.random.default_rng(seed)

    # If IDLE is a trained class, teach it what a hand mid-transition
    # between two DIFFERENT letters looks like - not just what a static
    # resting pose looks like (jittered copies of real IDLE recordings
    # already cover that). See _synthesize_transition_samples.
    if 'IDLE' in labels:
        idle_idx = labels.index('IDLE')
        non_idle = [c for c in range(len(labels)) if c != idle_idx and np.any(y_train == c)]
        per_class_real = {c: X_train_raw[y_train == c] for c in non_idle}
        if len(non_idle) >= 2:
            real_idle_count = int(np.sum(y_train == idle_idx))
            num_synthetic = max(real_idle_count, 200)
            synthetic_idle = _synthesize_transition_samples(per_class_real, non_idle, num_synthetic, rng)
            X_train_raw = np.concatenate([X_train_raw, synthetic_idle], axis=0)
            y_train = np.concatenate([y_train, np.full(len(synthetic_idle), idle_idx, dtype=y_train.dtype)])

    # OOD rejection stats from the real (pre-augmentation) training split, in
    # the same derived feature space the model uses.
    os.makedirs(models_dir, exist_ok=True)
    real_train_derived = np.array([add_derived_features(x) for x in X_train_raw], dtype=np.float32)
    with open(os.path.join(models_dir, 'ood_centroids.json'), 'w') as f:
        json.dump(_compute_ood_stats(real_train_derived, y_train, len(labels)), f)

    X_train_raw, y_train = _balanced_augmented_train_set(X_train_raw, y_train, len(labels), augment_multiplier, rng)

    # Shuffle so training batches aren't ordered by class
    perm = rng.permutation(len(X_train_raw))
    X_train_raw, y_train = X_train_raw[perm], y_train[perm]

    # Add derived geometric features (finger curl, thumb/palm distances) -
    # raw landmarks are what get augmented; the model consumes raw + derived,
    # computed identically here and in src/main.py.
    X_train = np.array([add_derived_features(x) for x in X_train_raw], dtype=np.float32)
    X_test = np.array([add_derived_features(x) for x in X_test_raw], dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    # Persist the final (possibly IDLE-extended) label order for the training script.
    with open(actions_path, 'w') as f:
        json.dump(labels, f, indent=2)

    return X_train, y_train, X_test, y_test, labels


def main():
    X_train, y_train, X_test, y_test, labels = preprocess_dataset()

    print(f'Classes ({len(labels)}): {labels}')
    print(f'Training samples: {X_train.shape[0]} (balanced + augmented)')
    print(f'Training shape: {X_train.shape}')
    print(f'Test samples: {X_test.shape[0]} (real, unaugmented, natural class distribution)')
    print(f'Test shape: {X_test.shape}')
    print('Preprocessed data saved to data/processed/')


if __name__ == '__main__':
    main()
