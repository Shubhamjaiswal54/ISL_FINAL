# main.py - FastAPI server exposing REST & WebSocket prediction endpoints
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
import asyncio
from pathlib import Path
import tensorflow as tf

try:
    from src.config import (FEATURES, SMOOTHING_FRAMES, CONFIDENCE_THRESHOLD,
                             DEBOUNCE_SECONDS, IDLE_SECONDS, LOW_CONFIDENCE_SECONDS, GAP_RESET_SECONDS,
                             GOOGLE_API_KEY, MOTION_VELOCITY_THRESHOLD, MOTION_EMA_ALPHA)
    from src.pipeline.sliding_window import SlidingWindowBuffer
    from src.pipeline.state_machine import ISLStateMachine
    from src.pipeline.nlp_smoother import smooth_glosses_to_sentence
    from src.pipeline.hand_features import add_derived_features, MODEL_FEATURES
    from src.pipeline.motion_gate import MotionGate
except ModuleNotFoundError:
    from config import (FEATURES, SMOOTHING_FRAMES, CONFIDENCE_THRESHOLD,
                         DEBOUNCE_SECONDS, IDLE_SECONDS, LOW_CONFIDENCE_SECONDS, GAP_RESET_SECONDS,
                         GOOGLE_API_KEY, MOTION_VELOCITY_THRESHOLD, MOTION_EMA_ALPHA)
    from pipeline.sliding_window import SlidingWindowBuffer
    from pipeline.state_machine import ISLStateMachine
    from pipeline.nlp_smoother import smooth_glosses_to_sentence
    from pipeline.hand_features import add_derived_features, MODEL_FEATURES
    from pipeline.motion_gate import MotionGate

# Initialize FastAPI app
app = FastAPI(title="VoxSign ML Microservice")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
# The model classifies a single frame; prob_buffer only smooths flicker by
# averaging the last few frames' output probabilities before argmax.
prob_buffer = SlidingWindowBuffer(max_size=SMOOTHING_FRAMES)
state_machine = ISLStateMachine(
    confidence_threshold=CONFIDENCE_THRESHOLD,
    debounce_seconds=DEBOUNCE_SECONDS,
    idle_seconds=IDLE_SECONDS,
    low_confidence_seconds=LOW_CONFIDENCE_SECONDS,
    gap_reset_seconds=GAP_RESET_SECONDS,
)
motion_gate = MotionGate(velocity_threshold=MOTION_VELOCITY_THRESHOLD, ema_alpha=MOTION_EMA_ALPHA)

# Load TFLite model
BASE_DIR = Path(__file__).resolve().parents[1]
interpreter = None
input_details = None
output_details = None
label_encoder = {}
model_load_error = None


def set_model_input(tensor):
    """Adapt float model input to the dtype and quantization expected by TFLite."""
    detail = input_details[0]
    model_tensor = tensor
    if detail['dtype'] in (np.uint8, np.int8):
        scale, zero_point = detail['quantization']
        if scale:
            model_tensor = np.round(tensor / scale + zero_point).clip(
                np.iinfo(detail['dtype']).min, np.iinfo(detail['dtype']).max
            ).astype(detail['dtype'])
    interpreter.set_tensor(detail['index'], model_tensor)


def get_model_output():
    """Return model probabilities as float32 regardless of TFLite output dtype."""
    detail = output_details[0]
    output = interpreter.get_tensor(detail['index']).astype(np.float32)
    if detail['dtype'] in (np.uint8, np.int8):
        scale, zero_point = detail['quantization']
        if scale:
            output = (output - zero_point) * scale
    return output

try:
    loaded_interpreter = tf.lite.Interpreter(model_path=str(BASE_DIR / 'models' / 'isl_model.tflite'))
    loaded_interpreter.allocate_tensors()
    loaded_input_details = loaded_interpreter.get_input_details()
    loaded_output_details = loaded_interpreter.get_output_details()
    
    with open(BASE_DIR / 'models' / 'label_encoder.json') as f:
        loaded_labels = {int(k): v for k, v in json.load(f).items()}
    
    interpreter = loaded_interpreter
    input_details = loaded_input_details
    output_details = loaded_output_details
    label_encoder = loaded_labels
    print("TFLite model loaded successfully")
except Exception as e:
    model_load_error = str(e)
    interpreter = None
    input_details = None
    output_details = None
    print(f"Warning: Could not load TFLite model: {e}")

# Out-of-distribution rejection (currently disabled - see is_out_of_distribution).
#
# This was a stopgap for rejecting non-sign poses (resting hand, mid-transition)
# before a real IDLE class existed. Its centroids/thresholds are computed purely
# from the Kaggle training photos' own distribution (scripts/2_preprocess_and_augment.py),
# so it has no notion of *your* webcam, hand, lighting, or angle - a genuine
# live sign can legitimately sit farther from those centroids than any Kaggle
# photo did, and gets its confidence wrongly zeroed. Now that IDLE is a real
# trained class (see models/label_encoder.json), it does this job properly and
# this geometric heuristic is redundant. Loaded but not applied below; flip
# APPLY_OOD_GATE on only if IDLE alone proves insufficient, and consider a much
# larger margin than the training-distribution-derived default.
APPLY_OOD_GATE = False
ood_centroids = None
ood_thresholds = None
try:
    with open(BASE_DIR / 'models' / 'ood_centroids.json') as f:
        ood_stats = json.load(f)
    ood_centroids = np.array(ood_stats['centroids'], dtype=np.float32)
    ood_thresholds = np.array(ood_stats['thresholds'], dtype=np.float32) * ood_stats.get('margin', 1.3)
except Exception as e:
    print(f"Warning: OOD rejection stats not loaded: {e}")


def is_out_of_distribution(raw_landmarks, predicted_idx):
    if not APPLY_OOD_GATE or ood_centroids is None or predicted_idx >= len(ood_centroids):
        return False
    distance = float(np.linalg.norm(raw_landmarks - ood_centroids[predicted_idx]))
    return distance > ood_thresholds[predicted_idx]


# Pydantic models
class LandmarkData(BaseModel):
    landmarks: list  # 126-dimensional feature vector


class SentenceResponse(BaseModel):
    sentence: str
    glosses: list


@app.get("/")
async def root():
    return {"message": "VoxSign ML Service is running", "status": "ok"}


@app.post("/predict", response_model=SentenceResponse)
async def predict_gloss(data: LandmarkData):
    """Single-frame prediction endpoint.

    Accepts a single 126-dimensional feature vector and returns
    the predicted gloss word if confidence threshold is met.
    """
    if not interpreter:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_load_error}")

    try:
        # Normalize and reshape
        features = np.array(data.landmarks, dtype=np.float32)
        if features.shape != (FEATURES,):
            raise HTTPException(status_code=400, detail=f"Landmarks must be {FEATURES}-dimensional")

        # Add finger-curl/thumb-palm features, then run inference directly on
        # this one frame - the model classifies a static pose, no temporal
        # window is needed for model input.
        model_features = add_derived_features(features)
        input_data = model_features.reshape(1, MODEL_FEATURES).astype(np.float32)
        set_model_input(input_data)
        interpreter.invoke()

        # Smooth: average the last few frames' probabilities to reduce flicker
        prob_buffer.add(get_model_output()[0])
        output_data = prob_buffer.get_window().mean(axis=0)
        predicted_idx = int(np.argmax(output_data))
        confidence = float(output_data[predicted_idx])

        # Reject if this frame's raw pose doesn't actually resemble the predicted letter
        if is_out_of_distribution(model_features, predicted_idx):
            confidence = 0.0

        # Reject if the hand is still moving between letters - the classifier
        # was only ever trained on held static poses.
        if not motion_gate.update(features):
            confidence = 0.0

        # Decode label
        predicted_label = label_encoder.get(predicted_idx, f'unknown_{predicted_idx}')

        # Update state machine
        accepted = state_machine.update(predicted_label, confidence)
        
        # Get current sentence glosses
        current_glosses = state_machine.get_sentence_glosses()
        
        # If sentence was flushed (idle detected), generate English sentence
        final_sentence = ''
        if state_machine.is_idle:
            final_sentence = smooth_glosses_to_sentence(current_glosses)
            state_machine.reset()
        
        return {
            "sentence": final_sentence,
            "glosses": current_glosses
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """WebSocket endpoint for real-time landmark streaming.
    
    Clients send 126-dimensional feature vectors,
    and the server streams back prediction updates.
    """
    await websocket.accept()

    # Each browser session needs its own smoothing buffer, sentence state, and
    # motion tracker - velocity is meaningless if computed across two different
    # viewers' frames.
    session_probs = SlidingWindowBuffer(max_size=SMOOTHING_FRAMES)
    session_state = ISLStateMachine(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        debounce_seconds=DEBOUNCE_SECONDS,
        idle_seconds=IDLE_SECONDS,
        low_confidence_seconds=LOW_CONFIDENCE_SECONDS,
        gap_reset_seconds=GAP_RESET_SECONDS,
    )
    session_motion = MotionGate(velocity_threshold=MOTION_VELOCITY_THRESHOLD, ema_alpha=MOTION_EMA_ALPHA)

    if not interpreter or not input_details or not output_details:
        await websocket.send_text(json.dumps({
            "error": "Model not loaded",
            "detail": model_load_error,
        }))
        await websocket.close(code=1011, reason="ML model unavailable")
        print(f"[ML] WebSocket rejected: model unavailable: {model_load_error}")
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            landmarks = np.array(data.get('landmarks', []), dtype=np.float32)
            print(f"[Backend -> ML] Landmark vector received: length={landmarks.size}")

            if landmarks.shape != (FEATURES,):
                await websocket.send_text(json.dumps({
                    "error": f"Landmarks must be {FEATURES}-dimensional"
                }))
                continue

            # Add finger-curl/thumb-palm features, then run inference directly
            # on this one frame - no warm-up delay waiting for a window to
            # fill, since the model classifies a single static pose.
            model_features = add_derived_features(landmarks)
            input_data = model_features.reshape(1, MODEL_FEATURES).astype(np.float32)
            set_model_input(input_data)
            interpreter.invoke()

            # Smooth: average the last few frames' probabilities to reduce flicker
            session_probs.add(get_model_output()[0])
            output_data = session_probs.get_window().mean(axis=0)
            predicted_idx = int(np.argmax(output_data))
            confidence = float(output_data[predicted_idx])

            # Reject if this frame's raw pose doesn't actually resemble the predicted letter
            if is_out_of_distribution(model_features, predicted_idx):
                confidence = 0.0

            # Reject if the hand is still moving between letters - the classifier
            # was only ever trained on held static poses, so a mid-transition
            # frame is outside its training distribution.
            settled = session_motion.update(landmarks)
            if not settled:
                confidence = 0.0

            predicted_label = label_encoder.get(predicted_idx, f'unknown_{predicted_idx}')

            # Update state machine
            accepted = session_state.update(predicted_label, confidence)
            current_glosses = session_state.get_sentence_glosses()

            # Build response
            response = {
                "word": predicted_label if accepted else None,
                "predictedLabel": predicted_label,
                "accepted": accepted,
                "confidence": confidence,
                "settled": settled,
                "sentence": "",
                "glosses": current_glosses
            }
            
            # If sentence flushed, generate English sentence
            if session_state.is_idle:
                response["sentence"] = smooth_glosses_to_sentence(current_glosses)
                session_state.reset()
            
            await websocket.send_text(json.dumps(response))
            print(f"[ML -> Backend] Prediction sent: predicted={predicted_label}, accepted={accepted}, word={response['word']}, confidence={confidence:.3f}, glosses={response['glosses']}")
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)