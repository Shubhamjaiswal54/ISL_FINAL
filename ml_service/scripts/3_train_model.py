# 3_train_model.py
# Single-frame ISL hand-pose classifier: MLP with early stopping and evaluation
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import json


def load_data():
    """Load preprocessed data."""
    (X_train, y_train) = (np.load('data/processed/X_train.npy'),
                          np.load('data/processed/y_train.npy'))
    (X_test, y_test) = (np.load('data/processed/X_test.npy'),
                        np.load('data/processed/y_test.npy'))
    return (X_train, y_train), (X_test, y_test)


def build_model(num_classes, input_shape=(142,)):
    """Static hand-pose classifier.

    Each sample is one wrist-relative, scale-normalized frame (126 raw
    landmark values) plus 16 derived geometric features - finger-curl angles
    and thumb/palm distances per hand, computed in
    src/pipeline/hand_features.py - which make closed-fist letters (A/C/I/M/O)
    much easier to separate than raw coordinates alone. ISL fingerspelling is
    a set of static poses, not motion, so a small MLP on a single frame
    generalizes better than a recurrent model trained on an artificially
    tiled 30-frame sequence (which is what this replaced).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train():
    (X_train, y_train), (X_test, y_test) = load_data()
    with open('data/actions.json') as f:
        labels = json.load(f)
    num_classes = len(labels)

    print(f'Number of classes: {num_classes}')
    print(f'Train shape: {X_train.shape}')
    print(f'Test shape: {X_test.shape}')

    model = build_model(num_classes)

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest Accuracy: {accuracy:.4f}')

    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print('\nClassification Report:')
    present = sorted(set(y_test.tolist()) | set(y_pred_classes.tolist()))
    print(classification_report(
        y_test, y_pred_classes,
        labels=present, target_names=[labels[i] for i in present]
    ))

    # Save model
    model.save('models/isl_model.keras')
    with open('models/label_encoder.json', 'w') as f:
        json.dump({str(index): label for index, label in enumerate(labels)}, f, indent=2)
    print('Model saved to models/isl_model.keras')

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    print('Training history plot saved to training_history.png')


if __name__ == '__main__':
    train()
