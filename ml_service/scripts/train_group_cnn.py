# train_group_cnn.py
# Trains the 8-group hand-shape CNN (skeleton-image input, matching the ASL
# reference app's architecture exactly), using this project's own
# ISL-derived groups (src/pipeline/letter_groups.py) instead of the ASL
# reference app's groupings - this project's data is Indian Sign Language,
# and ASL/ISL letterforms don't reliably share handshape families.
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline.letter_groups import LETTER_TO_GROUP, NUM_GROUPS

try:
    import tf_keras as target_keras
except ImportError:
    import tensorflow.keras as target_keras
    print('WARNING: tf_keras not installed - falling back to tf.keras. The '
          'saved .h5 may not load via `from tf_keras.models import load_model` '
          'in the target app; install tf_keras for exact compatibility.')

IMG_SIZE = 400


def list_samples(data_dirs=('data/skeleton_dataset', 'data/skeleton_dataset_own')):
    """Walk each data_dir/<LETTER>/*.jpg and map to its ISL hand-shape group.

    Merges the Kaggle-derived set and any personal recordings from
    scripts/1b_collect_skeleton_images.py automatically, so re-recording
    more letters just means re-running this script - no manual merging.
    """
    paths, groups = [], []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        for letter in sorted(os.listdir(data_dir)):
            letter_dir = os.path.join(data_dir, letter)
            if not os.path.isdir(letter_dir) or letter not in LETTER_TO_GROUP:
                continue
            group = LETTER_TO_GROUP[letter]
            for filename in sorted(os.listdir(letter_dir)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(letter_dir, filename))
                    groups.append(group)
    return paths, np.array(groups)


def make_dataset(paths, labels, batch_size=32, shuffle=True):
    def _load(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method='nearest')
        image = tf.cast(image, tf.float32)  # raw 0-255 pixels, no rescaling layer
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes=NUM_GROUPS, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    layers = target_keras.layers
    model = target_keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(96, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(data_dirs=('data/skeleton_dataset', 'data/skeleton_dataset_own'),
          output_path='models/isl_group_cnn.h5',
          epochs=25, batch_size=32, test_size=0.15, seed=42):
    paths, groups = list_samples(data_dirs)
    if not paths:
        raise RuntimeError(f'No skeleton images found under {data_dirs}')

    present_groups = sorted(set(groups.tolist()))
    print(f'Samples: {len(paths)} | groups present: {present_groups} (need all {NUM_GROUPS})')
    missing = set(range(NUM_GROUPS)) - set(present_groups)
    if missing:
        raise RuntimeError(
            f'Groups with zero samples: {sorted(missing)} - every group needs at least '
            'one covered letter with image data before training can proceed.'
        )

    train_paths, test_paths, train_groups, test_groups = train_test_split(
        paths, groups, test_size=test_size, random_state=seed, stratify=groups
    )
    print(f'Train: {len(train_paths)} | Test: {len(test_paths)}')

    train_ds = make_dataset(train_paths, train_groups, batch_size=batch_size, shuffle=True)
    test_ds = make_dataset(test_paths, test_groups, batch_size=batch_size, shuffle=False)

    model = build_model()
    model.summary()

    callbacks = target_keras.callbacks
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=[early_stop, reduce_lr])

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f'Test group accuracy: {acc:.4f}')

    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    print(classification_report(test_groups, y_pred, labels=list(range(NUM_GROUPS))))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f'Model saved to {output_path}')
    return model


if __name__ == '__main__':
    train()
