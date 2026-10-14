# 5_pc_live_test.py
# Local live OpenCV webcam tester
import json
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')
FEATURES_PER_HAND = 21 * 3  # (x, y, z)
FEATURES = FEATURES_PER_HAND * 2  # left + right
SMOOTHING_FRAMES = 5

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='models/isl_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe (Tasks API)
landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
)

# Load label encoder
with open('models/label_encoder.json') as f:
    label_encoder = {int(k): v for k, v in json.load(f).items()}


def normalize_hand(landmarks):
    """Wrist-centered, scale-normalized 63-feature array (x, y, z)."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    centered = coords - wrist
    scale = np.max(np.linalg.norm(centered, axis=1)) + 1e-6
    return (centered / scale).flatten()


def extract_features(results):
    """Build the 126-dim [left(63), right(63)] vector for this frame."""
    left_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)
    right_hand = np.zeros(FEATURES_PER_HAND, dtype=np.float32)

    if results.hand_landmarks:
        for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
            label = handedness[0].category_name
            vector = normalize_hand(hand_lms)
            if label == 'Left':
                left_hand = vector
            elif label == 'Right':
                right_hand = vector

    return np.concatenate([left_hand, right_hand]).astype(np.float32)


# State machine variables
confidence_threshold = 0.85
debounce_counter = 0
debounce_window = 15
idle_counter = 0
idle_threshold = 45
prob_buffer = []

cap = cv2.VideoCapture(0)
start_time = time.time()

print('Live ISL Translation Test')
print('Press "Q" to quit.')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    features = extract_features(results)

    if np.any(features):
        # Single-frame inference - no window to fill
        input_data = features.reshape(1, FEATURES).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        prob_buffer.append(output_data)
        if len(prob_buffer) > SMOOTHING_FRAMES:
            prob_buffer.pop(0)
        smoothed = np.mean(prob_buffer, axis=0)

        predicted_idx = int(np.argmax(smoothed))
        confidence = float(smoothed[predicted_idx])
        predicted_label = label_encoder.get(predicted_idx, f'unknown_{predicted_idx}')

        cv2.putText(frame, f'Class: {predicted_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Conf: {confidence:.2f}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # State machine logic
        if confidence >= confidence_threshold:
            debounce_counter = 0

            if predicted_label == 'IDLE':
                idle_counter += 1
                if idle_counter >= idle_threshold:
                    print(f'Sentence flushed! IDLE detected for {idle_threshold} frames')
                    idle_counter = 0
            else:
                idle_counter = 0
        else:
            debounce_counter += 1
            if debounce_counter >= debounce_window:
                debounce_counter = 0
    else:
        prob_buffer = []

    cv2.putText(frame, 'Press Q to Quit', (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('VoxSign Live Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Live test ended.')
