# build_skeleton_dataset.py
# Renders skeleton images (green 3px lines, red 2px joints, on a plain white
# 400x400x3 canvas) from the raw Kaggle photos, for the group-CNN pipeline.
# See src/pipeline/skeleton_render.py for the shared renderer and
# scripts/1b_collect_skeleton_images.py for the live-recording counterpart -
# both must render identically, or training/inference see different things.
import os
import sys
import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline.skeleton_render import render_skeleton_canvas

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')
hands = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,  # must match the live app's numHands: 1 exactly
        min_hand_detection_confidence=0.5,
    )
)


def build_dataset(input_dir='data/raw_dataset/Indian', output_dir='data/skeleton_dataset'):
    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(
        d.upper() for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and len(d) == 1 and d.isalpha()
    )
    if not classes:
        raise ValueError(f'No alphabet folders found in {input_dir}. Expected folders A through Z.')

    for label in classes:
        class_dir = os.path.join(input_dir, label)
        if not os.path.isdir(class_dir):
            class_dir = os.path.join(input_dir, label.lower())
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)

        image_files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        )
        saved = 0
        for filename in image_files:
            image = cv2.imread(os.path.join(class_dir, filename))
            if image is None:
                continue
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = hands.detect(mp_image)
            if not result.hand_landmarks:
                continue
            pts = [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]
            skeleton = render_skeleton_canvas(pts)
            cv2.imwrite(os.path.join(save_dir, f'{saved:04d}.jpg'), skeleton)
            saved += 1
        print(f'{label}: saved {saved}/{len(image_files)} skeleton images')

    print(f'Skeleton dataset built at {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/raw_dataset/Indian')
    parser.add_argument('--output-dir', default='data/skeleton_dataset')
    args = parser.parse_args()
    build_dataset(args.input_dir, args.output_dir)
