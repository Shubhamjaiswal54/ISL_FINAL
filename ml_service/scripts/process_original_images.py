# process_original_images.py
# Runs MediaPipe hand detection over the found dataset at
# data/original_images/<LETTER>/*.jpg (1000 real two-handed ISL photos per
# letter, consistent studio background) and feeds the results into the same
# two pipelines 1b_collect_skeleton_images.py feeds from live recording:
#   - landmark .npy -> data/raw_dataset/<gesture>/         (per-letter MLP)
#   - skeleton .jpg  -> data/skeleton_dataset_own/<LETTER>/ (group-CNN)
#
# Uses the same normalization (skeleton_render.py) and the same
# expected-hand-count validation (isl_alphabet_reference.py) as the live
# script, so this dataset's samples are indistinguishable downstream from
# ones collected via the webcam - no separate code path to keep in sync.
import cv2
import numpy as np
import os
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline.skeleton_render import render_skeleton_canvas
from src.pipeline.isl_alphabet_reference import expected_hand_count

ML_SERVICE_DIR = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH = os.path.join(ML_SERVICE_DIR, 'models', 'hand_landmarker.task')
SOURCE_DIR = os.path.join(ML_SERVICE_DIR, 'data', 'original_images')

FEATURES_PER_HAND = 21 * 3

landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
)


def normalize_hand(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    centered = coords - wrist
    scale = np.max(np.linalg.norm(centered, axis=1)) + 1e-6
    return (centered / scale).flatten()


def detect(image_bgr):
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(mp_image)

    if not results.hand_landmarks:
        return None, [], 0

    left_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    pixel_pts_per_hand = []
    for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
        label = handedness[0].category_name
        normalized_vector = normalize_hand(hand_lms)
        if label == 'Left':
            left_hand = normalized_vector
        elif label == 'Right':
            right_hand = normalized_vector
        pixel_pts_per_hand.append([(lm.x * w, lm.y * h) for lm in hand_lms])

    normalized = np.concatenate([left_hand, right_hand]).astype(np.float32)
    return normalized, pixel_pts_per_hand, len(pixel_pts_per_hand)


def process_letter(letter):
    letter_upper = letter.upper()
    need_hands = expected_hand_count(letter_upper)
    if need_hands is None:
        return None

    src_dir = os.path.join(SOURCE_DIR, letter)
    landmark_dir = os.path.join(ML_SERVICE_DIR, 'data', 'raw_dataset', letter.lower())
    skeleton_dir = os.path.join(ML_SERVICE_DIR, 'data', 'skeleton_dataset_own', letter_upper)
    os.makedirs(landmark_dir, exist_ok=True)
    os.makedirs(skeleton_dir, exist_ok=True)

    start_index = len([f for f in os.listdir(landmark_dir) if f.endswith('.npy')])

    saved = 0
    skipped_no_hand = 0
    skipped_wrong_count = 0
    filenames = sorted(os.listdir(src_dir))
    for i, filename in enumerate(filenames):
        path = os.path.join(src_dir, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        normalized, pixel_pts_per_hand, num_hands = detect(image)
        if normalized is None:
            skipped_no_hand += 1
            continue
        if num_hands != need_hands:
            skipped_wrong_count += 1
            continue

        idx = start_index + saved + 1
        np.save(os.path.join(landmark_dir, f'sample_{idx:04d}.npy'), normalized)
        skeleton = render_skeleton_canvas(pixel_pts_per_hand)
        cv2.imwrite(os.path.join(skeleton_dir, f'sample_{idx:04d}.jpg'), skeleton)
        saved += 1

        if (i + 1) % 200 == 0:
            print(f'  {letter_upper}: {i + 1}/{len(filenames)} images scanned, {saved} saved', flush=True)

    print(f'{letter_upper} (expects {need_hands}-hand): saved {saved}, '
          f'skipped {skipped_no_hand} no-hand, {skipped_wrong_count} wrong-count', flush=True)
    return saved


def main():
    letters = sorted(d for d in os.listdir(SOURCE_DIR)
                      if os.path.isdir(os.path.join(SOURCE_DIR, d)) and d.upper() in
                      __import__('string').ascii_uppercase)
    total = 0
    for letter in letters:
        saved = process_letter(letter)
        if saved is not None:
            total += saved
    print(f'\nTotal samples added: {total}')


if __name__ == '__main__':
    main()
