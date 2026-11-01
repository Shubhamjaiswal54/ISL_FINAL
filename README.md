# VoxSign: Continuous Indian Sign Language (ISL) Translation Engine

## Project Overview
VoxSign provides real-time, low-latency translation of dynamic Indian Sign Language gestures into grammatically correct spoken and written sentences.
## Architecture
- **Frontend**: React.js (Vite) with MediaPipe for client-side landmark extraction
- **Backend**: Node.js (Express) + MongoDB as API Gateway
- **ML Microservice**: Python FastAPI with TensorFlow/TFLite for gesture classification
## Monorepo Structure
```
voxsign-isl/
├── .gitignore
├── README.md
├── docker-compose.yml
│
├── frontend/       # React.js Client
│   └── ...
│
├── backend/        # Node.js / Express API Gateway
│   └── ...
│
└── ml_service/     # Python ML & NLP Microservice
    └── ...
```

## Quick Start

```bash
# 1. ML Service
cd ml_service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Backend Gateway
cd ../backend
npm install
# Set up .env with PORT=5000, MONGO_URI, and JWT_SECRET
npm run dev

# 3. React Client
cd ../frontend
npm install
# Set up .env with VITE_BACKEND_URL=http://localhost:5000
npm run dev
```

## Train the ISL A-Z Model (two-hand skeleton CNN)

ISL fingerspelling is mostly two-handed, so recognition works on both hands
together: MediaPipe finds up to two hands, both skeletons are drawn onto one
canvas (`src/pipeline/skeleton_render.py`), and a CNN classifies the canvas.
The same renderer is used for training and live inference, and the background
never reaches the model - only the hand skeletons do.

The ML service needs Python 3.10-3.13 (TensorFlow has no newer builds yet).

1. Put your photos in one folder per letter (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`):

```
ml_service/data/original_images/
├── A/
├── B/
├── ...
└── Z/
```

2. From `ml_service/`:

```bash
python scripts/process_original_images.py   # photos -> data/isl_landmarks.npz (+ data/skeleton_previews/)
python scripts/train_skeleton_cnn.py        # -> models/isl_skeleton_cnn.tflite + labels + report
python -m src.main                          # serve on :8000
```

- `process_original_images.py` prints, per letter, how many photos had 0/1/2
  hands detected. Every photo with at least one detected hand is kept, including
  ones whose hand count differs from `src/pipeline/isl_alphabet_reference.py`,
  because live signing produces those frames too. Photos are rotated from their
  EXIF orientation, and ordered by file time. Look at `data/skeleton_previews/`
  to see exactly what the CNN sees.
- `train_skeleton_cnn.py` holds out a later recording session when one exists
  (a gap of 30 seconds), and otherwise the last 15% of each letter, dropping
  the photos right on the cut so a burst doesn't land in both sets. A letter
  with fewer than 8 photos stays in training and is not scored. It prints the
  most-confused letter pairs and writes confusion-derived letter groups to
  `models/isl_skeleton_report.json`.
- Record more photos with `python scripts/1b_collect_skeleton_images.py --gesture A --samples 200`.
  They land in `data/original_images/` and are picked up by the extractor.
- If the MediaPipe model download fails (e.g. behind a TLS-inspecting proxy), fetch it with
  `curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`.

### Using it live

Press **Start** to open the camera, sign a letter, and hold it briefly until
it appears. **Space** ends the word. A misspelling such as `hallo` offers
`hello`, and Space uses the first suggestion. **Backspace** removes the last
letter. **Speak** turns the words into a sentence (Gemini when
`GOOGLE_API_KEY` is set in `ml_service/.env`) and reads it aloud. **Stop**
turns the camera off and leaves the words on screen.

Photograph signs with the front camera. Live landmarks are mirrored to match a
front-camera selfie; a photo taken of you by someone else will be the opposite
left-right.

The older single-hand scripts (`extract_kaggle_dataset.py`, `1_collect_landmarks.py`,
`2_preprocess_and_augment.py`, `3_train_model.py`, `4_export_to_tflite.py`,
`build_skeleton_dataset.py`) belong to the previous ASL-style landmark MLP and
are not used by the ISL pipeline.