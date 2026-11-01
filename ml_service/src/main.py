# main.py - FastAPI server: two-hand ISL skeleton CNN over REST & WebSocket
#
# Run from ml_service/:  python -m src.main
import asyncio
import json
import math
import time
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import (MODEL_PATH, LABELS_PATH, MAX_HANDS, SMOOTHING_FRAMES, CONFIDENCE_THRESHOLD,
                        DEBOUNCE_SECONDS, LOW_CONFIDENCE_SECONDS, GAP_RESET_SECONDS,
                        MOTION_SPEED_THRESHOLD, MOTION_EMA_ALPHA)
from src.pipeline.control_signs import ControlLatch, detect_control
from src.pipeline.spell_suggest import suggest_words
from src.pipeline.skeleton_render import HandOrderStabilizer, canonical_points, render_hands
from src.pipeline.state_machine import ISLStateMachine
from src.pipeline.motion_gate import MotionGate
from src.pipeline.nlp_smoother import words_to_sentence

app = FastAPI(title="VoxSign ML Microservice")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    return Interpreter


class SkeletonClassifier:
    """Renders hands with the training renderer and runs the TFLite CNN."""

    def __init__(self, model_path, labels_path):
        self.interpreter = _load_interpreter_class()(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.img_size = int(self.input_detail['shape'][1])
        with open(labels_path) as f:
            self.labels = {int(k): v for k, v in json.load(f).items()}

    def predict(self, hands, ordered=False):
        image = render_hands(hands, size=self.img_size, ordered=ordered).astype(np.float32)[np.newaxis]
        self.interpreter.set_tensor(self.input_detail['index'], image)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_detail['index'])[0].astype(np.float32)


# One interpreter shared by every connection. TFLite is not thread-safe, so
# inference is serialized; the lock is held only around the thread that runs it.
_infer_lock = asyncio.Lock()


async def classify(hands, ordered=False):
    async with _infer_lock:
        return await asyncio.to_thread(classifier.predict, hands, ordered)


classifier = None
model_load_error = None
try:
    classifier = SkeletonClassifier(MODEL_PATH, LABELS_PATH)
    print(f"Skeleton CNN loaded: {len(classifier.labels)} classes, {classifier.img_size}px input")
except Exception as error:
    model_load_error = str(error)
    print(f"Warning: could not load model from {MODEL_PATH}: {error}")


def parse_hands(raw_hands):
    """Validate 0-2 hands of 21 [x, y] or [x, y, z] points in pixel units."""
    if not isinstance(raw_hands, list) or len(raw_hands) > MAX_HANDS:
        raise ValueError(f"'hands' must be a list of at most {MAX_HANDS} hands")
    hands = []
    for raw in raw_hands:
        hand = np.asarray(raw, dtype=np.float32)
        if hand.ndim != 2 or hand.shape[0] != 21 or hand.shape[1] not in (2, 3):
            raise ValueError("each hand must be 21 points of [x, y] or [x, y, z]")
        if not np.all(np.isfinite(hand)):
            raise ValueError("hand points must be finite numbers")
        hands.append(hand)
    return hands


class TranslationSession:
    """Per-connection smoothing, motion, and sentence state."""

    def __init__(self):
        self.recent_probs = deque(maxlen=SMOOTHING_FRAMES)
        self.motion = MotionGate(speed_threshold=MOTION_SPEED_THRESHOLD, ema_alpha=MOTION_EMA_ALPHA)
        self.order = HandOrderStabilizer()
        self.seen_hands = None
        self.controls = ControlLatch()
        self.state = ISLStateMachine(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            debounce_seconds=DEBOUNCE_SECONDS,
            low_confidence_seconds=LOW_CONFIDENCE_SECONDS,
            gap_reset_seconds=GAP_RESET_SECONDS,
        )

    def _clear_tracking(self):
        self.recent_probs.clear()
        self.motion.reset()
        self.order.reset()
        self.seen_hands = None
        self.controls.reset()

    def _apply_control(self, name):
        if name == 'space':
            self.state.space()
        elif name == 'backspace':
            self.state.delete_character()

    async def process(self, hands, now):
        """Returns the prediction message for one frame.

        `now` is the client's capture time, so a frame processed late is still
        judged by how long the signer actually held the pose. An empty hand
        list does not end a word or a sentence.
        """
        if not hands:
            self._clear_tracking()
            self.state.hand_absent(now)
            return self._message(None, 0.0, False, False, 0)

        held, fired = self.controls.update(detect_control(hands), now)
        if held:
            if fired:
                self._apply_control(self.controls.name)
            return self._message(None, 1.0, False, True, len(hands), control=self.controls.name, control_fired=fired)

        # A one-hand frame and a two-hand frame of the same letter are
        # different pictures. Averaging them produces a third, wrong letter.
        if len(hands) != self.seen_hands:
            self.recent_probs.clear()
            self.motion.reset()
            self.seen_hands = len(hands)

        ordered = self.order.apply(hands)
        self.recent_probs.append(await classify(ordered, ordered=True))
        probs = np.mean(self.recent_probs, axis=0)
        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])

        settled = self.motion.update(canonical_points(ordered, ordered=True), now)
        label = classifier.labels.get(predicted_idx, f"unknown_{predicted_idx}")
        accepted = self.state.update(label, confidence if settled else 0.0, now)
        return self._message(label, confidence, accepted, settled, len(hands))

    def finish(self):
        """Stop was pressed: end the sentence with whatever was spelled."""
        words = self.state.finish()
        self._clear_tracking()
        return words

    def gloss_payload(self):
        current = self.state.current_word
        return {
            "type": "glosses",
            "glosses": self.state.get_sentence_glosses(),
            "currentWord": current,
            "suggestions": suggest_words(current),
            "threshold": CONFIDENCE_THRESHOLD,
        }

    def _message(self, label, confidence, accepted, settled, hand_count, control=None, control_fired=False):
        current = self.state.current_word
        return {
            "type": "prediction",
            "predictedLabel": label,
            "confidence": confidence,
            "accepted": accepted,
            "word": label if accepted else None,
            "settled": settled,
            "handCount": hand_count,
            "currentWord": current,
            "suggestions": suggest_words(current),
            "glosses": self.state.get_sentence_glosses(),
            "threshold": CONFIDENCE_THRESHOLD,
            "control": control,
            "controlFired": control_fired,
        }


class HandsPayload(BaseModel):
    hands: list


@app.get("/")
async def root():
    return {
        "message": "VoxSign ML Service is running",
        "status": "ok" if classifier else "model_unavailable",
        "detail": model_load_error,
    }


@app.post("/predict")
async def predict(payload: HandsPayload):
    """Stateless single-frame classification (no smoothing or sentence state)."""
    if not classifier:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_load_error}")
    try:
        hands = parse_hands(payload.hands)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if not hands:
        raise HTTPException(status_code=400, detail="at least one hand is required")

    async with _infer_lock:
        probs = await asyncio.to_thread(classifier.predict, hands, False)
    top = np.argsort(probs)[::-1][:3]
    return {
        "label": classifier.labels[int(top[0])],
        "confidence": float(probs[top[0]]),
        "top3": [{"label": classifier.labels[int(i)], "confidence": float(probs[i])} for i in top],
    }


def _capture_time(data):
    """Client capture time in seconds. Frames are judged on this clock, not
    on when the server got around to them."""
    stamp = data.get("t")
    if isinstance(stamp, (int, float)) and math.isfinite(stamp):
        return float(stamp) / 1000.0
    return time.monotonic()


async def _emit_sentence(websocket, words):
    """Runs off the receive loop so signing can continue while Gemini answers."""
    sentence = await asyncio.to_thread(words_to_sentence, words)
    try:
        await websocket.send_text(json.dumps({
            "type": "sentence",
            "sentence": sentence,
            "glosses": words,
        }))
    except Exception:
        pass


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """Clients stream {"hands": [...], "t": ms} frames. An empty hand list
    means nothing is visible. Words end on {"type": "space"} and the
    sentence is built on {"type": "speak"}.

    Only the newest frame is kept when inference falls behind. Control
    messages are never dropped. Capture timestamps make a held letter count
    even if the frames in between were skipped.
    """
    await websocket.accept()
    if not classifier:
        await websocket.send_text(json.dumps({"error": "Model not loaded", "detail": model_load_error}))
        await websocket.close(code=1011, reason="ML model unavailable")
        return

    session = TranslationSession()
    incoming = asyncio.Queue()
    latest_frame = {"data": None}

    async def receive_frames():
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") in ("finish", "reset", "backspace", "space", "speak"):
                    await incoming.put(data)
                else:
                    latest_frame["data"] = data
                    await incoming.put({"type": "frame"})
        except WebSocketDisconnect:
            await incoming.put({"type": "disconnect"})

    async def handle(data):
        if data.get("type") == "reset":
            session.finish()
            return
        if data.get("type") == "finish":
            words = session.finish()
            if words:
                asyncio.create_task(_emit_sentence(websocket, words))
            return
        if data.get("type") == "backspace":
            index = data.get("index")
            if type(index) is int:
                session.state.backspace(index)
            else:
                session.state.delete_character()
            await websocket.send_text(json.dumps(session.gloss_payload()))
            return
        if data.get("type") == "space":
            session.state.space(data.get("word"))
            await websocket.send_text(json.dumps(session.gloss_payload()))
            return
        if data.get("type") == "speak":
            session.state.space()
            words = session.state.words[:]
            await websocket.send_text(json.dumps(session.gloss_payload()))
            if words:
                asyncio.create_task(_emit_sentence(websocket, words))
            return
        if data.get("type") == "frame":
            data = latest_frame["data"]
            latest_frame["data"] = None
            if data is None:
                return

        try:
            hands = parse_hands(data.get("hands", []))
        except ValueError as error:
            await websocket.send_text(json.dumps({"error": str(error)}))
            return

        message = await session.process(hands, _capture_time(data))
        await websocket.send_text(json.dumps(message))

    receiver = asyncio.create_task(receive_frames())
    try:
        while True:
            data = await incoming.get()
            # A burst of frame notifications collapses to the newest one.
            # Control and disconnect messages keep their place in line.
            if data.get("type") == "frame":
                while True:
                    try:
                        nxt = incoming.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if nxt.get("type") != "frame":
                        await handle(data)
                        data = nxt
                        break
            if data.get("type") == "disconnect":
                break
            await handle(data)
    finally:
        receiver.cancel()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
