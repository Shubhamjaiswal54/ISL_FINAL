# config.py - VoxSign ML Service Configuration
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]

# Model artifacts written by scripts/train_skeleton_cnn.py (relative paths
# resolve against ml_service/, not the working directory).
MODEL_PATH = BASE_DIR / os.getenv('MODEL_PATH', 'models/isl_skeleton_cnn.tflite')
LABELS_PATH = BASE_DIR / os.getenv('LABELS_PATH', 'models/isl_skeleton_labels.json')

# ISL fingerspelling uses one or two hands depending on the letter.
MAX_HANDS = 2

# Average this many recent frames' class probabilities before argmax, to
# reduce flicker between visually similar letters.
SMOOTHING_FRAMES = 4

# State machine parameters (wall-clock seconds, not frame counts - the
# frontend's frame rate depends on the viewer's device).
CONFIDENCE_THRESHOLD = 0.50     # Minimum smoothed confidence to accept a letter into a word
DEBOUNCE_SECONDS = 0.35         # How long a prediction must hold before it's accepted
LOW_CONFIDENCE_SECONDS = 0.2    # How long confidence must dip to allow re-signing the same letter
GAP_RESET_SECONDS = 0.5         # Frames stopped arriving this long -> treat as hands lowered

# Motion gate: mean landmark speed in the renderer's unit square per second.
# A held pose (MediaPipe jitter only) sits well below this; moving between
# letters sits well above. Frames above it are not trusted.
MOTION_SPEED_THRESHOLD = 0.6
MOTION_EMA_ALPHA = 0.5

# Gemini (gloss words -> English sentence). Without a key, words are joined as-is.
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
