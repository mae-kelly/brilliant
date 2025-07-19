from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import tensorflow as tf
import hashlib
import time
import json
import os
import threading
from typing import Dict, Any
from collections import defaultdict

app = FastAPI(title="DeFi Sniper Inference Server", version="1.0.0")

# ======================== CONFIGURATION =========================
MODEL_DIR = "./models"
CACHE_TTL_SECONDS = 60
CONFIDENCE_THRESHOLD = 0.94
ENTROPY_THRESHOLD = 0.12
VALID_FEATURES = ["price_delta", "vol_delta", "liq_delta", "velocity", "momentum", "volatility_burst"]
ACTIVE_MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.tflite")

# ======================== GLOBAL STATE =========================
model = None
model_version_hash = ""
inference_cache: Dict[str, Dict[str, Any]] = {}
prediction_entropy_scores: Dict[str, float] = defaultdict(float)
model_metadata: Dict[str, Any] = {}
lock = threading.Lock()

# ======================== UTILS =========================
def hash_model(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_model(filepath):
    interpreter = tf.lite.Interpreter(model_path=filepath)
    interpreter.allocate_tensors()
    return interpreter

def validate_input(features: Dict[str, float]):
    for key in VALID_FEATURES:
        if key not in features:
            raise ValueError(f"Missing feature: {key}")
        if not isinstance(features[key], float):
            raise ValueError(f"Feature {key} must be float")

def normalize_vector(features: Dict[str, float]) -> np.ndarray:
    vec = np.array([features[k] for k in VALID_FEATURES])
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def calculate_entropy(probabilities: np.ndarray) -> float:
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def get_cached_prediction(token_id: str) -> Dict[str, Any]:
    entry = inference_cache.get(token_id)
    if entry and time.time() - entry["timestamp"] < CACHE_TTL_SECONDS:
        return entry
    return {}

def cache_prediction(token_id: str, prediction: float, entropy: float):
    inference_cache[token_id] = {
        "prediction": prediction,
        "entropy": entropy,
        "timestamp": time.time()
    }
    prediction_entropy_scores[token_id] = entropy

# ======================== DATA MODELS =========================
class InferenceRequest(BaseModel):
    token_id: str
    features: Dict[str, float]

class ModelSwapRequest(BaseModel):
    new_model_path: str

# ======================== CORE INFERENCE =========================
def run_inference(features: Dict[str, float]) -> Dict[str, Any]:
    global model

    validate_input(features)
    input_vector = normalize_vector(features).astype(np.float32)

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], [input_vector])
    model.invoke()
    output = model.get_tensor(output_details[0]['index'])[0]

    prediction = float(output[0])
    entropy = calculate_entropy(output)

    return {
        "prediction": prediction,
        "entropy": entropy
    }

# ======================== ROUTES =========================
@app.post("/predict")
async def predict(request: InferenceRequest):
    token_id = request.token_id
    features = request.features

    try:
        cached = get_cached_prediction(token_id)
        if cached:
            return {
                "token_id": token_id,
                "prediction": cached["prediction"],
                "entropy": cached["entropy"],
                "cached": True,
                "model_version": model_version_hash
            }

        with lock:
            result = run_inference(features)
            prediction = result["prediction"]
            entropy = result["entropy"]

        cache_prediction(token_id, prediction, entropy)

        return {
            "token_id": token_id,
            "prediction": prediction,
            "entropy": entropy,
            "model_version": model_version_hash,
            "thresholds": {
                "confidence": CONFIDENCE_THRESHOLD,
                "entropy": ENTROPY_THRESHOLD
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reload-model")
async def reload_model(req: ModelSwapRequest):
    global model, model_version_hash

    try:
        new_path = req.new_model_path
        if not os.path.exists(new_path):
            raise FileNotFoundError("Model path does not exist.")

        with lock:
            model = load_model(new_path)
            model_version_hash = hash_model(new_path)
            model_metadata["path"] = new_path
            model_metadata["hash"] = model_version_hash
            model_metadata["last_loaded"] = time.time()

        return {
            "status": "success",
            "new_model": model_metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    return {
        "model_loaded": model is not None,
        "model_hash": model_version_hash,
        "cached_tokens": len(inference_cache),
        "recent_entropy": dict(list(prediction_entropy_scores.items())[-5:]),
        "uptime": time.time() - model_metadata.get("last_loaded", 0)
    }

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok", "model": bool(model)})

# ======================== SERVER BOOTSTRAP =========================
def init():
    global model, model_version_hash, model_metadata

    if not os.path.exists(ACTIVE_MODEL_PATH):
        raise RuntimeError(f"Model file not found: {ACTIVE_MODEL_PATH}")

    model = load_model(ACTIVE_MODEL_PATH)
    model_version_hash = hash_model(ACTIVE_MODEL_PATH)
    model_metadata["path"] = ACTIVE_MODEL_PATH
    model_metadata["hash"] = model_version_hash
    model_metadata["last_loaded"] = time.time()

if __name__ == "__main__":
    init()
    uvicorn.run(app, host="0.0.0.0", port=8000)
