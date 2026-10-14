# extract_kaggle_dataset.py
# Offline Batch Feature Extraction Pipeline for Kaggle ISL datasets
import os
import argparse
import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')
hands = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        # Must match the live app's numHands: 1 (useMediaPipe.js) exactly.
        # With num_hands=2, most images picked up a second incidental/
        # spurious hand detection alongside the real signing hand, so most
        # training samples ended up with BOTH the left and right hand slots
        # populated - a region of input space live inference (always
        # exactly one hand, one slot zero) never actually produces. That
        # train/inference mismatch, not classification capacity, was the
        # real cause of persistent live misclassification despite near-
        # perfect held-out test accuracy (the test set shared the same
        # contamination).
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
)

FEATURES_PER_HAND = 21 * 3  # (x, y, z) = 63
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # 126 (Left + Right)


def normalize_hand(landmarks):
    """Wrist-centered, scale-normalized 63-feature array (x, y, z)."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    centered = coords - wrist
    scale = np.max(np.linalg.norm(centered, axis=1)) + 1e-6
    normalized = centered / scale
    return normalized.flatten()


def extract_frame_features(image_path):
    """Processes one image and returns a 1D vector of shape (84,)."""
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros(TOTAL_FEATURES, dtype=np.float32)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = hands.detect(mp_image)

    left_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)

    if results.hand_landmarks and results.handedness:
        for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            normalized_vector = normalize_hand(hand_lms)
            if label == 'Left':
                left_hand = normalized_vector
            elif label == 'Right':
                right_hand = normalized_vector

    return np.concatenate([left_hand, right_hand])


def process_image_dataset(input_dir, output_dir='data/raw_dataset', actions_path='data/actions.json'):
    """Extract one normalized 84-feature frame from each image in A-Z folders."""
    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(
        d.upper() for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and len(d) == 1 and d.isalpha()
    )
    if not classes:
        raise ValueError(f'No alphabet folders found in {input_dir}. Expected folders A through Z.')

    with open(actions_path, 'w') as f:
        json.dump(classes, f, indent=2)

    for label in classes:
        class_dir = os.path.join(input_dir, label)
        if not os.path.isdir(class_dir):
            class_dir = os.path.join(input_dir, label.lower())
        save_class_dir = os.path.join(output_dir, label)
        os.makedirs(save_class_dir, exist_ok=True)

        image_files = sorted(
            os.path.join(class_dir, filename)
            for filename in os.listdir(class_dir)
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        )
        saved = 0
        for image_path in image_files:
            features = extract_frame_features(image_path)
            if not np.any(features):
                continue
            np.save(os.path.join(save_class_dir, f'sample_{saved:04d}.npy'), features)
            saved += 1
        print(f'{label}: saved {saved}/{len(image_files)} images')

    print(f'Feature extraction complete. Labels saved to {actions_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/alphabet_dataset')
    parser.add_argument('--output-dir', default='data/raw_dataset')
    parser.add_argument('--actions-path', default='data/actions.json')
    args = parser.parse_args()
    process_image_dataset(args.input_dir, args.output_dir, args.actions_path)