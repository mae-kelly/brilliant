from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
from prometheus_client import Summary, Counter, generate_latest
from fastapi.responses import Response
import json
import asyncio
import time
from typing import Dict, List, Optional
import yaml

app = FastAPI(title="DeFi Momentum Trading API", version="1.0.0")

class PredictionRequest(BaseModel):
    chain: str
    token_address: str
    features: Dict
    timestamp: Optional[int] = None

class PredictionResponse(BaseModel):
    momentum_score: float
    threshold: float
    uncertainty: float
    entropy: float
    recommendation: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    uptime: float
    model_loaded: bool
    predictions_served: int
    avg_response_time: float

class TradingSignal(BaseModel):
    chain: str
    token_address: str
    signal_strength: float
    confidence: float
    recommended_action: str
    position_size: float
    timestamp: int

predict_time = Summary('predict_request_processing_seconds', 'Time spent processing prediction requests')
prediction_counter = Counter('predictions_total', 'Total predictions made', ['chain', 'recommendation'])
error_counter = Counter('api_errors_total', 'Total API errors', ['error_type'])

start_time = time.time()
model_instance = None
settings = None

@app.on_event("startup")
async def startup_event():
    global model_instance, settings
    try:
        with open('settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        
        from inference_model import MomentumEnsemble
        model_instance = MomentumEnsemble()
        
        logging.info("API server started successfully")
        
    except Exception as e:
        logging.error(f"Startup error: {e}")
        model_instance = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        uptime = time.time() - start_time
        
        predictions_served = sum([
            prediction_counter.labels(chain=chain, recommendation=rec)._value._value
            for chain in ['arbitrum', 'polygon', 'optimism']
            for rec in ['BUY', 'HOLD', 'SELL']
        ])
        
        avg_response_time = predict_time._sum._value / max(predict_time._count._value, 1)
        
        return HealthResponse(
            status="healthy" if model_instance else "degraded",
            uptime=uptime,
            model_loaded=model_instance is not None,
            predictions_served=int(predictions_served),
            avg_response_time=avg_response_time
        )
        
    except Exception as e:
        error_counter.labels(error_type="health_check").inc()
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
@predict_time.time()
async def predict(request: PredictionRequest):
    try:
        if not model_instance:
            error_counter.labels(error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_processing = time.time()
        
        features_df = pd.DataFrame([request.features])
        
        if features_df.empty or len(features_df.columns) < 5:
            error_counter.labels(error_type="insufficient_features").inc()
            raise HTTPException(status_code=400, detail="Insufficient features provided")
        
        momentum_score = model_instance.predict(features_df)
        
        uncertainty = 0.0
        entropy = 0.0
        
        if hasattr(model_instance, 'prediction_history') and model_instance.prediction_history:
            recent_predictions = model_instance.prediction_history[-10:]
            if recent_predictions:
                recent_uncertainties = [p.get('uncertainty', 0) for p in recent_predictions]
                recent_entropies = [p.get('entropy', 0) for p in recent_predictions]
                uncertainty = np.mean(recent_uncertainties)
                entropy = np.mean(recent_entropies)
        
        threshold = model_instance.dynamic_threshold
        
        recommendation = "HOLD"
        if momentum_score > threshold * 1.1:
            recommendation = "BUY"
        elif momentum_score < threshold * 0.8:
            recommendation = "SELL"
        
        processing_time = time.time() - start_processing
        
        prediction_counter.labels(chain=request.chain, recommendation=recommendation).inc()
        
        logging.info(json.dumps({
            'event': 'prediction_served',
            'chain': request.chain,
            'token': request.token_address,
            'momentum_score': momentum_score,
            'recommendation': recommendation,
            'processing_time': processing_time
        }))
        
        return PredictionResponse(
            momentum_score=momentum_score,
            threshold=threshold,
            uncertainty=uncertainty,
            entropy=entropy,
            recommendation=recommendation,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="prediction_error").inc()
        logging.error(json.dumps({
            'event': 'api_prediction_error',
            'chain': request.chain,
            'token': request.token_address,
            'error': str(e)
        }))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    try:
        if not model_instance:
            error_counter.labels(error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if len(requests) > 50:
            error_counter.labels(error_type="batch_too_large").inc()
            raise HTTPException(status_code=400, detail="Batch size too large (max 50)")
        
        results = []
        
        for request in requests:
            try:
                prediction_response = await predict(request)
                results.append({
                    'token_address': request.token_address,
                    'chain': request.chain,
                    'success': True,
                    'prediction': prediction_response.dict()
                })
            except Exception as e:
                results.append({
                    'token_address': request.token_address,
                    'chain': request.chain,
                    'success': False,
                    'error': str(e)
                })
        
        return {'results': results, 'total_processed': len(results)}
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="batch_prediction_error").inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/signals/{chain}")
async def get_trading_signals(chain: str, limit: int = 10):
    try:
        if chain not in ['arbitrum', 'polygon', 'optimism']:
            raise HTTPException(status_code=400, detail="Invalid chain")
        
        signals = []
        
        if model_instance and hasattr(model_instance, 'momentum_scores'):
            recent_scores = model_instance.momentum_scores[-limit:] if model_instance.momentum_scores else []
            
            for i, score in enumerate(recent_scores):
                if score > model_instance.dynamic_threshold:
                    signals.append(TradingSignal(
                        chain=chain,
                        token_address=f"0x{'0'*40}",  # Placeholder
                        signal_strength=score,
                        confidence=min(score / model_instance.dynamic_threshold, 1.0),
                        recommended_action="BUY" if score > model_instance.dynamic_threshold * 1.1 else "WATCH",
                        position_size=0.001 * (score / model_instance.dynamic_threshold),
                        timestamp=int(time.time()) - (len(recent_scores) - i) * 60
                    ))
        
        return {'signals': signals[-limit:], 'chain': chain, 'count': len(signals)}
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="signals_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get signals: {str(e)}")

@app.get("/model/status")
async def get_model_status():
    try:
        if not model_instance:
            return {'status': 'not_loaded', 'error': 'Model instance not available'}
        
        model_summary = model_instance.get_model_summary() if hasattr(model_instance, 'get_model_summary') else {}
        
        return {
            'status': 'loaded',
            'model_info': model_summary,
            'dynamic_threshold': getattr(model_instance, 'dynamic_threshold', 0.75),
            'total_predictions': len(getattr(model_instance, 'momentum_scores', [])),
            'last_retrain': 'unknown',
            'device': str(getattr(model_instance, 'device', 'unknown'))
        }
        
    except Exception as e:
        error_counter.labels(error_type="model_status_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@app.post("/model/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not hasattr(model_instance, 'retrain_if_needed'):
            raise HTTPException(status_code=501, detail="Retrain not supported")
        
        async def retrain_task():
            try:
                await model_instance.retrain_if_needed()
                logging.info("Manual retrain completed successfully")
            except Exception as e:
                logging.error(f"Manual retrain failed: {e}")
        
        background_tasks.add_task(retrain_task)
        
        return {'message': 'Retrain started', 'status': 'in_progress'}
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="retrain_trigger_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to trigger retrain: {str(e)}")

@app.get("/model/threshold")
async def get_threshold():
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            'current_threshold': model_instance.dynamic_threshold,
            'threshold_history': getattr(model_instance, 'threshold_history', []),
            'last_updated': int(time.time())
        }
        
    except Exception as e:
        error_counter.labels(error_type="threshold_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get threshold: {str(e)}")

@app.post("/model/threshold")
async def update_threshold(new_threshold: float):
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not 0.1 <= new_threshold <= 0.95:
            raise HTTPException(status_code=400, detail="Threshold must be between 0.1 and 0.95")
        
        old_threshold = model_instance.dynamic_threshold
        model_instance.dynamic_threshold = new_threshold
        
        logging.info(json.dumps({
            'event': 'threshold_updated_manually',
            'old_threshold': old_threshold,
            'new_threshold': new_threshold
        }))
        
        return {
            'message': 'Threshold updated',
            'old_threshold': old_threshold,
            'new_threshold': new_threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="threshold_update_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to update threshold: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    try:
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type="text/plain")
    except Exception as e:
        error_counter.labels(error_type="metrics_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/performance/{chain}")
async def get_performance_metrics(chain: str, days: int = 1):
    try:
        if chain not in ['arbitrum', 'polygon', 'optimism']:
            raise HTTPException(status_code=400, detail="Invalid chain")
        
        cutoff_time = time.time() - (days * 24 * 3600)
        
        performance_data = {
            'chain': chain,
            'period_days': days,
            'predictions_made': prediction_counter.labels(chain=chain, recommendation='BUY')._value._value + 
                             prediction_counter.labels(chain=chain, recommendation='SELL')._value._value + 
                             prediction_counter.labels(chain=chain, recommendation='HOLD')._value._value,
            'buy_signals': prediction_counter.labels(chain=chain, recommendation='BUY')._value._value,
            'sell_signals': prediction_counter.labels(chain=chain, recommendation='SELL')._value._value,
            'hold_signals': prediction_counter.labels(chain=chain, recommendation='HOLD')._value._value,
            'avg_processing_time': predict_time._sum._value / max(predict_time._count._value, 1),
            'error_rate': sum(error_counter._value._value for error_counter in error_counter._metrics.values()) / 
                         max(predict_time._count._value, 1)
        }
        
        return performance_data
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="performance_metrics_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@app.websocket("/ws/predictions")
async def websocket_predictions(websocket):
    await websocket.accept()
    try:
        while True:
            if model_instance and hasattr(model_instance, 'momentum_scores'):
                recent_score = model_instance.momentum_scores[-1] if model_instance.momentum_scores else 0
                
                data = {
                    'timestamp': int(time.time()),
                    'momentum_score': recent_score,
                    'threshold': model_instance.dynamic_threshold,
                    'recommendation': 'BUY' if recent_score > model_instance.dynamic_threshold else 'HOLD'
                }
                
                await websocket.send_json(data)
            
            await asyncio.sleep(5)
            
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)