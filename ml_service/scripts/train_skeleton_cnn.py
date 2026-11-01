# train_skeleton_cnn.py
# Flat 26-letter ISL classifier on rendered two-hand skeleton images.
#
# Input: data/isl_landmarks.npz from scripts/process_original_images.py.
# Each training sample is augmented at the landmark level and rendered by
# src/pipeline/skeleton_render.render_hands - the same function live
# inference uses - so there is no train/inference drawing mismatch.
#
# Outputs (models/):
#   isl_skeleton_cnn.keras       trained Keras model
#   isl_skeleton_cnn.tflite      float16 TFLite model served by src/main.py
#   isl_skeleton_labels.json     class index -> letter
#   isl_skeleton_report.json     test accuracy, per-letter scores, confusion
#                                matrix, and confusion-derived letter groups
#
# Usage (from ml_service/):
#   python scripts/train_skeleton_cnn.py
import argparse
import json
import math
import os
import sys

import numpy as np
import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix

ML_SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ML_SERVICE_DIR)
from src.pipeline.skeleton_render import IMG_SIZE, render_hands
from src.pipeline.skeleton_augment import augment_hands

MODELS_DIR = os.path.join(ML_SERVICE_DIR, 'models')


def load_dataset(path, max_per_class=None):
    data = np.load(path)
    points, counts, labels, orders = data['points'], data['hand_counts'], data['labels'], data['orders']
    if max_per_class:
        keep = np.concatenate([np.flatnonzero(labels == c)[:max_per_class] for c in np.unique(labels)])
        points, counts, labels, orders = points[keep], counts[keep], labels[keep], orders[keep]
    hands = [list(p[:n]) for p, n in zip(points, counts)]
    return hands, labels, orders


def _split_one_class(idx, orders, val_frac, test_frac, gap_ms, boundary):
    """Hold out later captures of one letter.

    A gap of `gap_ms` in capture time starts a new session. With three or
    more sessions, the last is the test set and the one before it is
    validation, so a burst of near-identical photos can't sit on both sides
    of the cut. Otherwise the latest fraction is held out and `boundary`
    photos on each side of a cut are discarded.
    With no session gap, fewer than 8 photos all stay in training: forcing a
    fraction split there leaves the letter with nothing to learn from.
    """
    idx = idx[np.argsort(orders[idx], kind='stable')]
    if len(idx) == 0:
        return [], [], []

    sessions, current = [], [idx[0]]
    for previous, current_idx in zip(idx, idx[1:]):
        if orders[current_idx] - orders[previous] > gap_ms:
            sessions.append(current)
            current = [current_idx]
        else:
            current.append(current_idx)
    sessions.append(current)
    if len(sessions) >= 3:
        return ([i for session in sessions[:-2] for i in session], list(sessions[-2]), list(sessions[-1]))
    if len(idx) < 8:
        return list(idx), [], []

    n = len(idx)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    gap = min(boundary, max(0, (n - n_test - n_val - 1) // 2))
    while gap > 0 and n - n_test - n_val - 2 * gap < 1:
        gap -= 1
    if n - n_test - n_val - 2 * gap < 1:
        return list(idx), [], []

    test = list(idx[n - n_test:])
    val_end = n - n_test - gap
    val = list(idx[val_end - n_val:val_end])
    train = list(idx[:val_end - n_val - gap])
    return train, val, test


def split_by_capture_order(labels, orders, val_frac, test_frac, gap_ms=30_000, boundary=8):
    train, val, test = [], [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        class_train, class_val, class_test = _split_one_class(
            idx, orders, val_frac, test_frac, gap_ms, boundary)
        train.extend(class_train)
        val.extend(class_val)
        test.extend(class_test)
    return np.array(train, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int)


class SkeletonBatches(keras.utils.PyDataset):
    """Renders skeleton images on the fly; fresh augmentation every epoch."""

    def __init__(self, hands, y, indices, batch_size, img_size, augment, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.hands, self.y = hands, y
        self.indices = np.array(indices)
        self.batch_size, self.img_size = batch_size, img_size
        self.augment, self.seed = augment, seed
        self.epoch = 0
        if augment:
            np.random.default_rng(seed).shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, batch_idx):
        batch = self.indices[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        # Seeded per (epoch, batch) so worker threads never share an RNG.
        rng = np.random.default_rng((self.seed, self.epoch, batch_idx))
        images = np.stack([
            render_hands(augment_hands(self.hands[i], rng) if self.augment else self.hands[i],
                         size=self.img_size)
            for i in batch
        ]).astype(np.float32)
        return images, self.y[batch]

    def on_epoch_end(self):
        self.epoch += 1
        if self.augment:
            np.random.default_rng((self.seed, self.epoch)).shuffle(self.indices)


def build_model(num_classes, img_size):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def suggest_groups(confusion_rates, class_names, threshold):
    """Union letters whose mutual confusion rate reaches `threshold`. These
    are the groups a two-stage (group -> letter) model would use - derived
    from what this model actually confuses, not from finger-shape intuition."""
    parent = list(range(len(class_names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            if confusion_rates[i, j] + confusion_rates[j, i] >= threshold:
                parent[find(i)] = find(j)

    groups = {}
    for i, name in enumerate(class_names):
        groups.setdefault(find(i), []).append(name)
    return sorted(groups.values(), key=lambda g: (-len(g), g))


def export_tflite(model, tflite_path):
    try:
        import tensorflow as tf
    except ImportError:
        print('TensorFlow not installed - skipping TFLite export.')
        return False
    if keras.backend.backend() != 'tensorflow':
        print(f'Keras backend is {keras.backend.backend()!r} - TFLite export needs the tensorflow backend.')
        return False

    saved_model_dir = os.path.join(MODELS_DIR, 'isl_skeleton_saved_model')
    model.export(saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    with open(tflite_path, 'wb') as f:
        f.write(converter.convert())
    print(f'TFLite model saved to {tflite_path}')
    return True


def main():
    parser = argparse.ArgumentParser(description='Train the ISL skeleton CNN.')
    parser.add_argument('--data', default=os.path.join(ML_SERVICE_DIR, 'data', 'isl_landmarks.npz'))
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--img-size', type=int, default=IMG_SIZE)
    parser.add_argument('--val-frac', type=float, default=0.10)
    parser.add_argument('--test-frac', type=float, default=0.15)
    parser.add_argument('--group-threshold', type=float, default=0.03,
                        help='Mutual confusion rate at which two letters are grouped')
    parser.add_argument('--max-per-class', type=int, default=None, help='Subsample for quick experiments')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--session-gap', type=float, default=30.0,
                        help='Seconds of quiet that separate recording sessions')
    args = parser.parse_args()

    hands, labels, orders = load_dataset(args.data, args.max_per_class)
    class_names = sorted(np.unique(labels).tolist())
    y = np.array([class_names.index(label) for label in labels], dtype=np.int32)
    train_idx, val_idx, test_idx = split_by_capture_order(
        labels, orders, args.val_frac, args.test_frac, gap_ms=args.session_gap * 1000)
    print(f'Classes ({len(class_names)}): {class_names}')
    print(f'Samples: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}')
    if len(test_idx):
        unscored = [name for name in class_names if name not in set(labels[test_idx])]
        if unscored:
            print(f'No held-out photos for {unscored} (too few samples); trained, but not scored.')

    loader_kwargs = dict(workers=args.workers, use_multiprocessing=False)
    train_data = SkeletonBatches(hands, y, train_idx, args.batch_size, args.img_size, augment=True, **loader_kwargs)
    val_data = (SkeletonBatches(hands, y, val_idx, args.batch_size, args.img_size, augment=False, **loader_kwargs)
                if len(val_idx) else None)
    callbacks = []
    if val_data is not None:
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6),
        ]

    model = build_model(len(class_names), args.img_size)
    model.summary()
    model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks)

    y_true = y_pred = None
    if len(test_idx):
        test_data = SkeletonBatches(hands, y, test_idx, args.batch_size, args.img_size, augment=False, **loader_kwargs)
        probs = model.predict(test_data, verbose=0)
        y_true, y_pred = y[test_idx], probs.argmax(axis=1)
    accuracy, cm, groups = None, None, [[name] for name in class_names]
    if y_true is not None and len(y_true):
        accuracy = float((y_true == y_pred).mean())
        print(f'\nHeld-out test accuracy (last {args.test_frac:.0%} of each letter): {accuracy:.4f}\n')
        print(classification_report(y_true, y_pred, labels=range(len(class_names)),
                                    target_names=class_names, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        confused = sorted(
            ((rates[i, j], class_names[i], class_names[j])
             for i in range(len(class_names)) for j in range(len(class_names)) if i != j and rates[i, j] > 0),
            reverse=True,
        )[:10]
        if confused:
            print('Most confused (true -> predicted):')
            for rate, true_name, pred_name in confused:
                print(f'  {true_name} -> {pred_name}: {rate:.1%}')

        groups = suggest_groups(rates, class_names, args.group_threshold)
        multi = [g for g in groups if len(g) > 1]
        if multi:
            print(f'\nLetters confused with each other at >= {args.group_threshold:.0%} '
                  f'(candidate groups for a two-stage model): {multi}')
        else:
            print('\nNo letter pairs are confused often enough to justify a grouped model.')
    else:
        print('No held-out photos - skipping the test report.')

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'isl_skeleton_labels.json'), 'w') as f:
        json.dump({str(i): name for i, name in enumerate(class_names)}, f, indent=2)
    with open(os.path.join(MODELS_DIR, 'isl_skeleton_report.json'), 'w') as f:
        json.dump({
            'test_accuracy': accuracy,
            'img_size': args.img_size,
            'classes': class_names,
            'confusion_matrix': cm.tolist() if cm is not None else [],
            'suggested_groups': groups,
            'group_threshold': args.group_threshold,
        }, f, indent=2)

    try:
        model.save(os.path.join(MODELS_DIR, 'isl_skeleton_cnn.keras'))
    except ImportError as error:
        print(f'Could not save .keras file ({error}).')
    export_tflite(model, os.path.join(MODELS_DIR, 'isl_skeleton_cnn.tflite'))


if __name__ == '__main__':
    main()
