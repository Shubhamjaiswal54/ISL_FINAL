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

## Train the A-Z Image Dataset

Arrange the 1,200 images into one folder per class:

```
ml_service/data/alphabet_dataset/
├── A/
├── B/
├── ...
└── Z/
```
Image files can be `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp`. Run the pipeline from `ml_service`:
```bash
python scripts/extract_kaggle_dataset.py --input-dir data/alphabet_dataset
python scripts/2_preprocess_and_augment.py
python scripts/3_train_model.py
python scripts/4_export_to_tflite.py
python -m src.main
```

The extractor detects and normalizes hand landmarks, writes the discovered alphabet labels to `data/actions.json`, and saves the model label mapping to `models/label_encoder.json`. Images without a detectable hand are skipped and reported per class.