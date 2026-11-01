# 1b_collect_skeleton_images.py
# Record webcam photos into data/original_images/<LETTER>/, the folder
# scripts/process_original_images.py reads. Saving the photo (instead of a
# landmark file) keeps this recording on the same path as the photos you
# already took: EXIF orientation, two-hand detection, and the skeleton
# rendering all happen in that one script.
#
# Usage (from ml_service/):
#   python scripts/1b_collect_skeleton_images.py --gesture A --samples 200
#   python scripts/process_original_images.py
import argparse
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ML_SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ML_SERVICE_DIR, 'models', 'hand_landmarker.task')


def main():
    parser = argparse.ArgumentParser(description='Record ISL letter photos from the webcam.')
    parser.add_argument('--gesture', required=True, help='Letter to record, e.g. A')
    parser.add_argument('--samples', type=int, default=200)
    args = parser.parse_args()

    letter = args.gesture.upper()
    if len(letter) != 1 or not letter.isalpha():
        raise SystemExit('--gesture must be a single letter A-Z')

    save_dir = os.path.join(ML_SERVICE_DIR, 'data', 'original_images', letter)
    os.makedirs(save_dir, exist_ok=True)

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit('Could not open the webcam.')

    print(f'Recording {args.samples} photos for {letter} into {save_dir}')
    print('A frame is saved only while at least one hand is visible. Press Q to stop.')
    print('Afterwards run: python scripts/process_original_images.py')

    collected = 0
    started = time.time()
    while collected < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int((time.time() - started) * 1000),
        )
        hands = len(result.hand_landmarks or [])
        if hands:
            collected += 1
            cv2.imwrite(os.path.join(save_dir, f'{int(time.time() * 1000)}_{collected:04d}.jpg'), frame)
        cv2.putText(frame, f'{letter}  {collected}/{args.samples}  hands:{hands}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Recording ISL photos', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f'Saved {collected} photos to {save_dir}')


if __name__ == '__main__':
    main()
