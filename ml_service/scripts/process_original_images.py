# process_original_images.py
# Photos -> two-hand landmark dataset for the ISL skeleton CNN.
#
# Reads data/original_images/<LETTER>/*.jpg, runs MediaPipe with up to two
# hands, and writes every usable detection to one file (data/isl_landmarks.npz)
# that scripts/train_skeleton_cnn.py renders and augments on the fly. Raw
# pixel-space points are stored (not rendered images) so augmentation can
# happen at the landmark level, before drawing, exactly like live inference.
#
# Usage (from ml_service/):
#   python scripts/process_original_images.py
#   python scripts/process_original_images.py --input-dir /path/to/photos --previews 10
import argparse
import os
import re
import sys
import urllib.request
from collections import Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ML_SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ML_SERVICE_DIR)
from src.pipeline.skeleton_render import render_hands
from src.pipeline.isl_alphabet_reference import expected_hand_count

MODEL_PATH = os.path.join(ML_SERVICE_DIR, 'models', 'hand_landmarker.task')
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
             'hand_landmarker/float16/1/hand_landmarker.task')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
MAX_IMAGE_SIDE = 1280  # detection quality plateaus well below phone-camera resolution


def ensure_landmarker_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f'Downloading MediaPipe hand landmarker to {MODEL_PATH} ...')
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except OSError as error:
        raise SystemExit(f'Download failed ({error}). Fetch it manually:\n'
                         f'  curl -L -o {MODEL_PATH} {MODEL_URL}')


def natural_key(name):
    """IMG_9.jpg before IMG_10.jpg, so file order follows capture order."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def create_landmarker():
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
    )


def read_image(path):
    """Load a photo upright. Phone cameras store rotation in EXIF, and
    cv2.imread ignores it, which feeds the detector sideways hands."""
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def detect_hands(landmarker, image_bgr):
    """Return a list of 0-2 hands, each (21, 3) in pixel units: x * width,
    y * height, z * width (MediaPipe's z shares x's scale)."""
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest > MAX_IMAGE_SIDE:
        factor = MAX_IMAGE_SIDE / longest
        image_bgr = cv2.resize(image_bgr, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_AREA)
        h, w = image_bgr.shape[:2]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return [
        np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand], dtype=np.float32)
        for hand in (result.hand_landmarks or [])
    ]


def process_dataset(input_dir, output_path, previews_dir, num_previews):
    ensure_landmarker_model()
    landmarker = create_landmarker()

    class_dirs = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')
    )
    if not class_dirs:
        raise SystemExit(f'No class folders found in {input_dir}. Expected A/, B/, ... Z/.')

    points, hand_counts, labels, files, orders = [], [], [], [], []
    for class_dir in class_dirs:
        label = class_dir.upper()
        need_hands = expected_hand_count(label)
        folder = os.path.join(input_dir, class_dir)
        # Capture time first, filename as the tiebreak. A random filename
        # still groups a later recording session after an earlier one.
        paths = sorted(
            (os.path.join(folder, name) for name in os.listdir(folder)
             if name.lower().endswith(IMAGE_EXTENSIONS)),
            key=lambda path: (os.path.getmtime(path), natural_key(os.path.basename(path))),
        )

        stats = Counter()
        detected_counts = Counter()
        saved_previews = 0
        for index, path in enumerate(paths):
            filename = os.path.basename(path)
            try:
                image = read_image(path)
            except Exception:
                stats['unreadable'] += 1
                continue

            hands = detect_hands(landmarker, image)
            detected_counts[len(hands)] += 1
            if not hands:
                stats['no_hand'] += 1
                continue
            # Kept even when the count doesn't match the letter chart. Live
            # signing sends whatever MediaPipe finds, including a resting
            # second hand or a hand hidden behind the other one.
            if need_hands is not None and len(hands) != need_hands:
                stats['other_count'] += 1

            padded = np.zeros((2, 21, 3), dtype=np.float32)
            padded[:len(hands)] = hands
            points.append(padded)
            hand_counts.append(len(hands))
            labels.append(label)
            files.append(os.path.join(class_dir, filename))
            orders.append(os.path.getmtime(path) * 1000.0 + index * 1e-3)
            stats['saved'] += 1

            if previews_dir and saved_previews < num_previews:
                os.makedirs(os.path.join(previews_dir, label), exist_ok=True)
                cv2.imwrite(os.path.join(previews_dir, label, f'{os.path.splitext(filename)[0]}.png'),
                            render_hands(hands, size=256))
                saved_previews += 1

            if (index + 1) % 250 == 0:
                print(f'  {label}: {index + 1}/{len(paths)} scanned', flush=True)

        counts = ', '.join(f'{k} hand(s): {v}' for k, v in sorted(detected_counts.items()))
        print(f'{label} (chart says {need_hands or "any"}-hand): saved {stats["saved"]}/{len(paths)}'
              f' | no hand {stats["no_hand"]}, other count {stats["other_count"]}'
              f', unreadable {stats["unreadable"]} | detected -> {counts}', flush=True)
        if paths and stats['other_count'] > len(paths) / 2:
            print(f'  Note: most {label} photos show a different hand count than '
                  f'isl_alphabet_reference.py ({need_hands}). They are kept, because live '
                  f'signing produces those frames too.', flush=True)

    landmarker.close()
    if not points:
        raise SystemExit('No usable detections - nothing saved.')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        points=np.stack(points),
        hand_counts=np.array(hand_counts, dtype=np.int8),
        labels=np.array(labels),
        files=np.array(files),
        orders=np.array(orders, dtype=np.float64),
    )
    print(f'\nSaved {len(points)} samples across {len(set(labels))} classes to {output_path}')
    if previews_dir:
        print(f'Skeleton previews (what the CNN sees) saved to {previews_dir}')


def main():
    parser = argparse.ArgumentParser(description='Extract two-hand ISL landmarks from photos.')
    parser.add_argument('--input-dir', default=os.path.join(ML_SERVICE_DIR, 'data', 'original_images'))
    parser.add_argument('--output', default=os.path.join(ML_SERVICE_DIR, 'data', 'isl_landmarks.npz'))
    parser.add_argument('--previews-dir', default=os.path.join(ML_SERVICE_DIR, 'data', 'skeleton_previews'))
    parser.add_argument('--previews', type=int, default=5, help='Rendered skeleton previews to save per letter')
    args = parser.parse_args()
    process_dataset(args.input_dir, args.output, args.previews_dir if args.previews > 0 else None,
                    args.previews)


if __name__ == '__main__':
    main()
