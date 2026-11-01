# skeleton_augment.py - Landmark-level augmentation for the skeleton CNN
#
# Applied to raw hand points before render_hands(), never to rendered pixels,
# so augmented samples are still valid skeletons drawn by the exact renderer
# live inference uses. The dataset is a single signer, so these transforms
# stand in for the variation other people bring: camera tilt, different hand
# proportions, and slightly different spacing between hands.
#
# Images are not mirrored. Several ISL letters are distinguished by which
# side the dominant hand is on, so teaching the model that a sign and its
# mirror are the same letter would erase that difference. Live landmarks are
# mirrored once, in the browser, to match a front-camera selfie.
import numpy as np


def augment_hands(hands, rng,
                  rotation_deg=15.0,
                  aspect_range=0.1,
                  hand_shift=0.03,
                  hand_scale_range=0.08,
                  jitter=0.006):
    """Return an augmented copy of hands: a list of (21, 3) point arrays in
    one shared pixel scale. z (depth) is kept, so the renderer still draws
    the nearer hand on top. Offsets are relative to the joint bounding box,
    so the transform strength doesn't depend on photo resolution."""
    prepared = []
    for hand in hands:
        hand = np.asarray(hand, dtype=np.float32)
        if hand.shape[1] == 2:
            hand = np.column_stack([hand, np.zeros(len(hand), dtype=np.float32)])
        prepared.append(hand[:, :3].copy())
    hands = prepared
    if not hands:
        return hands

    flat = np.concatenate([hand[:, :2] for hand in hands])
    center = flat.mean(axis=0)
    span = max(float((flat.max(axis=0) - flat.min(axis=0)).max()), 1e-6)

    angle = np.deg2rad(rng.uniform(-rotation_deg, rotation_deg))
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    aspect = np.array([1 + rng.uniform(-aspect_range, aspect_range), 1.0], dtype=np.float32)

    out = []
    for hand in hands:
        xy, z = hand[:, :2], hand[:, 2]
        if len(hands) > 1:
            # Small independent per-hand placement change: other signers
            # hold their hands at slightly different distances apart.
            wrist = xy[0].copy()
            scale = rng.uniform(1 - hand_scale_range, 1 + hand_scale_range)
            xy = (xy - wrist) * scale + wrist
            z = z * scale
            xy = xy + rng.normal(0.0, hand_shift * span, size=2)
        xy = ((xy - center) * aspect) @ rotation.T + center
        xy = xy + rng.normal(0.0, jitter * span, size=xy.shape)
        out.append(np.column_stack([xy, z]).astype(np.float32))
    return out
