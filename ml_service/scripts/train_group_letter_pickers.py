# train_group_letter_pickers.py
# Trains one small classifier per hand-shape group to pick the exact letter
# within it, replacing nearest-centroid distance (validated at only 69%
# overall accuracy, with some letters at 0% - centroid distance is too
# crude to separate letters that were clustered together precisely because
# they're geometrically similar). Reuses the same engineered feature data
# already built for the main MLP (data/processed/*.npy).
import os
import sys
import json
import numpy as np
from tensorflow import keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline.letter_groups import GROUPS


def train_pickers(data_dir='data/processed', actions_path='data/actions.json',
                   output_dir='models/group_pickers'):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    labels = json.load(open(actions_path))
    label_to_idx = {l: i for i, l in enumerate(labels)}

    os.makedirs(output_dir, exist_ok=True)
    metadata = {}
    overall_correct = 0
    overall_total = 0

    for group_idx, letters in GROUPS.items():
        if len(letters) <= 1:
            metadata[str(group_idx)] = {'letters': letters, 'model': None}
            print(f'Group {group_idx} {letters}: single letter, no picker needed')
            continue

        class_indices = [label_to_idx[l] for l in letters]
        idx_to_local = {ci: i for i, ci in enumerate(class_indices)}

        train_mask = np.isin(y_train, class_indices)
        test_mask = np.isin(y_test, class_indices)
        Xtr = X_train[train_mask]
        ytr = np.array([idx_to_local[y] for y in y_train[train_mask]])
        Xte = X_test[test_mask]
        yte = np.array([idx_to_local[y] for y in y_test[test_mask]])

        model = keras.Sequential([
            keras.layers.Input(shape=(Xtr.shape[1],)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(len(letters), activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=40, batch_size=64,
                  callbacks=[early_stop], verbose=0)

        loss, acc = model.evaluate(Xte, yte, verbose=0)
        print(f'Group {group_idx} {letters}: test accuracy {acc:.4f} (n={len(Xte)})')
        overall_correct += acc * len(Xte)
        overall_total += len(Xte)

        model_path = os.path.join(output_dir, f'group_{group_idx}.keras')
        model.save(model_path)
        metadata[str(group_idx)] = {'letters': letters, 'model': f'group_{group_idx}.keras'}

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\nOverall within-group picker accuracy: {overall_correct/overall_total:.4f}')
    print(f'Saved group pickers to {output_dir}')


if __name__ == '__main__':
    train_pickers()
