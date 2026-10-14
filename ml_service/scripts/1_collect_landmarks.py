# 1_collect_landmarks.py
# Standalone OpenCV + MediaPipe (Tasks API) landmark collection script
import cv2
import numpy as np
import os
import time
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')

FEATURES_PER_HAND = 21 * 3  # (x, y, z) = 63
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # 126 (Left + Right)

landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        # Must match the live app's numHands: 1 (useMediaPipe.js) exactly -
        # with num_hands=2, an incidental second hand in frame (your other
        # hand, resting nearby) gets detected and saved too, producing
        # samples with BOTH hand slots populated that live inference (one
        # hand, one slot always zero) never actually produces.
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
)


def normalize_hand(landmarks):
    """Wrist-centered, scale-normalized 63-feature array (x, y, z).

    Must match scripts/extract_kaggle_dataset.py and
    frontend/src/services/landmarkNormalizer.js exactly, since training
    data and live inference both flow through this same encoding.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    centered = coords - wrist
    scale = np.max(np.linalg.norm(centered, axis=1)) + 1e-6
    normalized = centered / scale
    return normalized.flatten()


def extract_landmarks(image_bgr, timestamp_ms):
    """Extract normalized landmarks for up to 2 hands from one BGR frame.

    Returns a 126-dim vector: [left_hand(63), right_hand(63)], with the
    missing hand's slot zero-filled, matching the training-time encoding.
    Returns None if no hand is detected.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not results.hand_landmarks:
        return None

    left_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)

    for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
        label = handedness[0].category_name  # "Left" or "Right"
        normalized_vector = normalize_hand(hand_lms)
        if label == 'Left':
            left_hand = normalized_vector
        elif label == 'Right':
            right_hand = normalized_vector

    return np.concatenate([left_hand, right_hand]).astype(np.float32)  # 126 values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gesture', required=True, help='Gesture label (e.g., HELLO, PLEASE, IDLE)')
    parser.add_argument('--samples', type=int, default=200, help='Number of samples to collect')
    args = parser.parse_args()

    gesture = args.gesture.lower()
    # Anchored to this script's own location (like MODEL_PATH above), not the
    # caller's working directory - running this from inside scripts/ instead
    # of ml_service/ used to silently save into scripts/data/raw_dataset/.
    ml_service_dir = os.path.join(os.path.dirname(__file__), '..')
    save_dir = os.path.join(ml_service_dir, 'data', 'raw_dataset', gesture)
    os.makedirs(save_dir, exist_ok=True)

    # DirectShow is more reliable than the default MSMF backend on Windows,
    # which can fail to grab frames (often because another app - e.g. a
    # browser tab still using the webcam - is holding the device).
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Could not open the webcam (device index 0).')
        print('Close any other app/browser tab using the camera and try again.')
        return

    print(f'Collecting {args.samples} samples for "{args.gesture}"...')
    print('Press "Q" to quit early.')

    collected = 0
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
        landmarks = extract_landmarks(frame, timestamp_ms)
        if landmarks is not None:
            # Save sample
            sample_path = os.path.join(save_dir, f'sample_{collected+1:03d}.npy')
            np.save(sample_path, landmarks)
            collected += 1
            print(f'Collected: {collected}/{args.samples}')

        # Display instructions
        cv2.putText(frame, f'Samples: {collected}/{args.samples}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Collecting Landmarks', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'Collection complete: {collected} samples saved to {save_dir}')


if __name__ == '__main__':
    main()
