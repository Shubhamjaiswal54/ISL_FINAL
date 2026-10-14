# 1b_collect_skeleton_images.py
# Like 1_collect_landmarks.py, but also renders and saves a skeleton image
# per sample (green skeleton, red joints, white 400x400x3 canvas - see
# src/pipeline/skeleton_render.py) for the group-CNN pipeline
# (scripts/train_group_cnn.py). Saves both from the same detection - no
# extra recording session needed to feed both pipelines:
#   - landmark .npy -> data/raw_dataset/<gesture>/   (existing per-letter MLP)
#   - skeleton .jpg  -> data/skeleton_dataset_own/<LETTER>/  (group-CNN)
#
# ISL fingerspelling mixes one-handed and two-handed letters (see
# src/pipeline/isl_alphabet_reference.py) - detects up to 2 hands and only
# accepts a sample when the detected count matches what that letter expects,
# so an incomplete two-handed capture (or a stray extra hand during a
# one-handed letter) doesn't silently get saved as if it were correct.
import cv2
import numpy as np
import os
import time
import argparse
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline.skeleton_render import render_skeleton_canvas
from src.pipeline.isl_alphabet_reference import expected_hand_count

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')

FEATURES_PER_HAND = 21 * 3
TOTAL_FEATURES = FEATURES_PER_HAND * 2

landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,  # ISL fingerspelling needs up to 2 hands depending on the letter
        min_hand_detection_confidence=0.5,
    )
)


def normalize_hand(landmarks):
    """Wrist-centered, scale-normalized 63-feature array (x, y, z)."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    centered = coords - wrist
    scale = np.max(np.linalg.norm(centered, axis=1)) + 1e-6
    return (centered / scale).flatten()


def detect(image_bgr, timestamp_ms):
    """Returns (normalized_126_vector, pixel_pts_per_hand, num_hands_detected).
    pixel_pts_per_hand is a list of 1-2 hands' 21 (x, y) points each, in
    this frame's own pixel coordinates, for skeleton rendering. Returns
    (None, [], 0) if no hand is found."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gesture', required=True, help='Letter/gesture label (e.g., A, IDLE)')
    parser.add_argument('--samples', type=int, default=200, help='Number of samples to collect')
    args = parser.parse_args()

    gesture_lower = args.gesture.lower()
    gesture_upper = args.gesture.upper()
    need_hands = expected_hand_count(gesture_upper)  # None for non-letter gestures (e.g. IDLE) - no validation then

    ml_service_dir = os.path.join(os.path.dirname(__file__), '..')
    landmark_dir = os.path.join(ml_service_dir, 'data', 'raw_dataset', gesture_lower)
    skeleton_dir = os.path.join(ml_service_dir, 'data', 'skeleton_dataset_own', gesture_upper)
    os.makedirs(landmark_dir, exist_ok=True)
    os.makedirs(skeleton_dir, exist_ok=True)

    start_index = len([f for f in os.listdir(landmark_dir) if f.endswith('.npy')])

    # DirectShow is more reliable than the default MSMF backend on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Could not open the webcam (device index 0).')
        print('Close any other app/browser tab using the camera and try again.')
        return

    if need_hands is not None:
        print(f'"{gesture_upper}" expects {need_hands}-handed input - frames with a different hand '
              f'count are skipped automatically.')
    print(f'Collecting {args.samples} samples for "{args.gesture}" (landmarks + skeleton images)...')
    print('Press "Q" to quit early.')

    collected = 0
    skipped_wrong_count = 0
    consecutive_failures = 0
    start_time = time.time()
    while collected < args.samples:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= 30:
                print('ERROR: Camera stopped returning frames.')
                print('Close any other app/browser tab using the camera and try again.')
                break
            continue
        consecutive_failures = 0

        timestamp_ms = int((time.time() - start_time) * 1000)
        normalized, pixel_pts_per_hand, num_hands = detect(frame, timestamp_ms)
        if normalized is not None:
            if need_hands is not None and num_hands != need_hands:
                skipped_wrong_count += 1
            else:
                idx = start_index + collected + 1
                np.save(os.path.join(landmark_dir, f'sample_{idx:03d}.npy'), normalized)
                skeleton = render_skeleton_canvas(pixel_pts_per_hand)
                cv2.imwrite(os.path.join(skeleton_dir, f'sample_{idx:03d}.jpg'), skeleton)
                collected += 1
                print(f'Collected: {collected}/{args.samples}')

        status = f'Samples: {collected}/{args.samples}'
        if need_hands is not None:
            status += f'  (need {need_hands}-hand, skipped {skipped_wrong_count} wrong-count frames)'
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Collecting Landmarks + Skeleton Images', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'Collection complete: {collected} samples saved to {landmark_dir} and {skeleton_dir}')


if __name__ == '__main__':
    main()
