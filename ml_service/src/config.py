# config.py - VoxSign ML Service Configuration
# Threshold constants, smoothing window, API keys

# Model configuration
# The model classifies a single frame (static hand pose) - there is no
# temporal input window. SMOOTHING_FRAMES only controls how many recent
# per-frame prediction probabilities get averaged before argmax, to reduce
# flicker between visually similar letters; it is independent of the model.
SMOOTHING_FRAMES = 5  # ~0.15-0.2s at 30fps
NUM_LANDMARKS = 21  # 21 hand landmarks
NUM_AXES = 3  # X, Y, Z coordinates
NUM_HANDS = 2  # left + right, zero-filled when a hand is absent
FEATURES = NUM_LANDMARKS * NUM_AXES * NUM_HANDS  # 126 features per frame

# State machine parameters
# Wall-clock seconds, not frame counts - the frontend detects on an uncapped
# requestAnimationFrame loop, so "frames" arrive at whatever rate the
# viewer's display refreshes; a frame-count threshold would mean a
# different amount of real time for every viewer.
CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence to accept prediction
DEBOUNCE_SECONDS = 0.35  # How long a prediction must hold before it's accepted
IDLE_SECONDS = 30.0  # How long IDLE must hold before the sentence is flushed
LOW_CONFIDENCE_SECONDS = 0.2  # How long confidence must dip to allow re-signing the same letter
GAP_RESET_SECONDS = 0.5  # No frames for this long (hand left frame) resets debounce state,
                          # so re-signing the same letter after a real pause isn't silently dropped

# Motion gate parameters
# The model was trained on held static poses, so a frame captured mid-transition
# between two letters is outside its training distribution and can produce a
# confident wrong prediction. Rather than accept that noise, we track landmark
# velocity and force confidence to 0 (routing through the existing low-confidence
# path above) while the hand is still moving.
MOTION_VELOCITY_THRESHOLD = 0.06  # Unit-scale-normalized landmark distance per frame
MOTION_EMA_ALPHA = 0.5  # Smoothing factor for the velocity estimate

# MediaPipe / Vision settings
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# API keys - load from .env
GOOGLE_API_KEY = "fdajfl;kdsjfal;dsjflkdsjfkl;ads" # Set via environment: export GOOGLE_API_KEY="your_key_here"