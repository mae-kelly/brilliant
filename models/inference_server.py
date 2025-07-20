from dynamic_parameters import get_dynamic_config
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
except ImportError:
    def get_dynamic_config(): return {"volatility_threshold": 0.1, "confidence_threshold": 0.75}
    def update_performance(*args): pass
try:
    from dynamic_settings import dynamic_settings
except ImportError:
    class MockSettings:
        def get_trading_params(self): return {"liquidity_threshold": 50000}
        def get_position_size(self, pv, conf): return min(pv * 0.1, 1.0)
    dynamic_settings = MockSettings()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn
import joblib
import os
from typing import List
import json

app = FastAPI(title="Renaissance ML Inference Server")

class PredictionRequest(BaseModel):
    features: List[float]
    token_address: str

class PredictionResponse(BaseModel):
    breakout_probability: float
    confidence: float
    entropy: float
    recommendation: str
    model_version: str

class MLInferenceServer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = "v1.0.0"
        self.model_type = "unknown"
        
    def load_model(self):
        try:
            if os.path.exists("models/model_weights.tflite"):
                try:
                    import tensorflow as tf
                    self.model = tf.lite.Interpreter(model_path="models/model_weights.tflite")
                    self.model.allocate_tensors()
                    self.model_type = "tflite"
                    print("✅ Loaded TensorFlow Lite model")
                except Exception as e:
                    print(f"⚠️ TFLite failed: {e}")
                    raise
            
            if self.model is None and os.path.exists("models/model_weights.pkl"):
                self.model = joblib.load("models/model_weights.pkl")
                self.model_type = "sklearn"
                print("✅ Loaded sklearn model")
            
            if self.model is None:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(n_estimators=50, random_state=42)
                self.model_type = "fallback"
                print("⚠️ Using fallback model")
            
            self.scaler = joblib.load("models/scaler.pkl")
            
            with open("models/feature_names.json", 'r') as f:
                self.feature_names = json.load(f)
                
            return True
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> dict:
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            if self.model_type == "tflite":
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                
                self.model.set_tensor(input_details[0]['index'], features_scaled.astype(np.float32))
                self.model.invoke()
                
                probability = float(self.model.get_tensor(output_details[0]['index'])[0][0])
                
            elif self.model_type == "sklearn":
                prob_array = self.model.predict_proba(features_scaled)[0]
                probability = prob_array[1] if len(prob_array) > 1 else prob_array[0]
                
            else:
                probability = 0.5 + np.random.uniform(-0.2, 0.2)
            
            confidence = abs(probability - 0.5) * 2
            entropy = -(probability * np.log(probability + 1e-10) + 
                      (1 - probability) * np.log(1 - probability + 1e-10))
            
            if probability > 0.8 and confidence > 0.6:
                recommendation = "STRONG_BUY"
            elif probability > 0.6 and confidence > 0.4:
                recommendation = "BUY"
            elif probability < 0.3:
                recommendation = "AVOID"
            else:
                recommendation = "HOLD"
            
            return {
                "breakout_probability": float(probability),
                "confidence": float(confidence),
                "entropy": float(entropy),
                "recommendation": recommendation,
                "model_version": f"{self.model_version}-{self.model_type}"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

server = MLInferenceServer()

@app.on_event("startup")
async def startup_event():
    success = server.load_model()
    if not success:
        raise RuntimeError("Failed to load ML model")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if len(request.features) != len(server.feature_names):
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {len(server.feature_names)} features, got {len(request.features)}"
        )
    
    features = np.array(request.features)
    result = server.predict(features)
    
    return PredictionResponse(**result)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": server.model is not None}

@app.get("/model_info")
async def model_info():
    return {
        "model_version": server.model_version,
        "model_type": server.model_type,
        "feature_count": len(server.feature_names) if server.feature_names else 0,
        "feature_names": server.feature_names
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
