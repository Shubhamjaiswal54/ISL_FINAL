# render_skeletons_from_landmarks.py
# Renders skeleton images directly from already-recorded landmark .npy files
# (data/raw_dataset/<letter>/*.npy, from 1_collect_landmarks.py) instead of
# requiring a fresh webcam session. A rendered skeleton only needs the
# hand's relative geometry - which a wrist-relative, scale-normalized
# landmark vector already fully captures - not an original photo behind it.
# render_skeleton_canvas recenters/rescales whatever points it's given, so
# it doesn't matter that these were normalized for a different purpose.
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pipeline.skeleton_render import render_skeleton_canvas
from src.pipeline.letter_groups import LETTER_TO_GROUP

FEATURES = 126


def render_dataset(raw_dataset_dir='data/raw_dataset', output_dir='data/skeleton_dataset_own'):
    total_saved = 0
    for gesture_dir in sorted(os.listdir(raw_dataset_dir)):
        letter = gesture_dir.upper()
        if letter not in LETTER_TO_GROUP:
            continue  # skip IDLE, Kaggle leftovers ("Indian"), anything not a group letter

        folder = os.path.join(raw_dataset_dir, gesture_dir)
        if not os.path.isdir(folder):
            continue

        save_dir = os.path.join(output_dir, letter)
        os.makedirs(save_dir, exist_ok=True)

        saved = 0
        dropped = 0
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith('.npy'):
                continue
            sample = np.load(os.path.join(folder, filename)).astype(np.float32)
            if sample.shape != (FEATURES,):
                continue

            left = sample[:63]
            right = sample[63:126]
            left_present = np.any(left)
            right_present = np.any(right)
            if left_present and right_present:
                dropped += 1  # same two-hand contamination filter as the main pipeline
                continue
            if not left_present and not right_present:
                continue

            hand = left if left_present else right
            pts_xy = hand.reshape(21, 3)[:, :2]  # (x, y); z dropped, rendering is 2D

            skeleton = render_skeleton_canvas(pts_xy)
            cv2.imwrite(os.path.join(save_dir, f'{saved:04d}.jpg'), skeleton)
            saved += 1

        msg = f'{letter}: rendered {saved} skeleton image(s)'
        if dropped:
            msg += f' (dropped {dropped} two-hand-contaminated)'
        print(msg)
        total_saved += saved

    print(f'\nTotal: {total_saved} skeleton images rendered to {output_dir}')


if __name__ == '__main__':
    render_dataset()
