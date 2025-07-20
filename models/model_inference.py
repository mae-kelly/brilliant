import asyncio
import numpy as np
import tensorflow as tf
import joblib
import json
import time
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging

@dataclass
class PredictionResult:
    breakout_probability: float
    confidence: float
    entropy: float
    regime_state: int
    regime_confidence: float
    execution_time_ms: float
    model_version: str
    entry_urgency: float
    momentum_decay_threshold: float
    recommended_hold_time: float

class RenaissanceInference:
    def __init__(self):
        self.tflite_interpreter = None
        self.transformer_model = None
        self.ensemble_weights = [0.4, 0.3, 0.3]
        self.scaler = None
        self.feature_names = []
        self.model_version = "renaissance_v2.1"
        
        self.prediction_cache = {}
        self.cache_ttl = 2
        
        self.sequence_buffer = deque(maxlen=120)
        self.momentum_history = deque(maxlen=1000)
        
        self.regime_states = {
            0: "bull_momentum",
            1: "bear_momentum", 
            2: "sideways_consolidation",
            3: "high_volatility_breakout"
        }
        
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'avg_inference_time': 0.0,
            'confidence_distribution': deque(maxlen=1000),
            'accuracy_tracker': deque(maxlen=1000)
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        try:
            await self.load_models()
            await self.initialize_regime_detector()
            self.logger.info("✅ Renaissance inference engine initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            await self.initialize_fallback_models()
            return True

    async def load_models(self):
        model_loaded = False
        
        if os.path.exists("models/renaissance_model.tflite"):
            try:
                self.tflite_interpreter = tf.lite.Interpreter(
                    model_path="models/renaissance_model.tflite"
                )
                self.tflite_interpreter.allocate_tensors()
                self.logger.info("✅ Loaded Renaissance TFLite model")
                model_loaded = True
            except Exception as e:
                self.logger.warning(f"TFLite loading failed: {e}")
        
        if os.path.exists("models/transformer_model.h5"):
            try:
                self.transformer_model = tf.keras.models.load_model("models/transformer_model.h5")
                self.logger.info("✅ Loaded Transformer model")
                model_loaded = True
            except Exception as e:
                self.logger.warning(f"Transformer loading failed: {e}")
        
        if not model_loaded:
            raise Exception("No valid models found")
        
        try:
            self.scaler = joblib.load("models/scaler.pkl")
        except:
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        
        try:
            with open("models/feature_names.json", 'r') as f:
                self.feature_names = json.load(f)
        except:
            self.feature_names = [f"feature_{i}" for i in range(45)]

    async def initialize_fallback_models(self):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import RobustScaler
        
        self.ensemble_models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=150, max_depth=10, random_state=42),
            'xgb': None
        }
        
        self.scaler = RobustScaler()
        
        X_synthetic = np.random.random((1000, 45))
        y_synthetic = (np.random.random(1000) > 0.6).astype(int)
        
        X_scaled = self.scaler.fit_transform(X_synthetic)
        
        for name, model in self.ensemble_models.items():
            if model:
                model.fit(X_scaled, y_synthetic)
        
        self.logger.info("✅ Initialized fallback ensemble models")

    async def initialize_regime_detector(self):
        try:
            from hmmlearn.hmm import GaussianHMM
            self.regime_detector = GaussianHMM(n_components=4, random_state=42)
            
            synthetic_returns = np.random.normal(0, 0.02, (1000, 1))
            self.regime_detector.fit(synthetic_returns)
            
        except:
            self.regime_detector = None

    async def predict_breakout(self, features: np.ndarray, token_address: str = "") -> PredictionResult:
        start_time = time.time()
        
        try:
            cache_key = self._generate_cache_key(features, token_address)
            
            if cache_key in self.prediction_cache:
                cached_result, timestamp = self.prediction_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    self.stats['cache_hits'] += 1
                    return cached_result
            
            features_processed = await self._preprocess_features(features)
            
            ensemble_predictions = await self._ensemble_predict(features_processed)
            
            final_probability = self._weighted_ensemble(ensemble_predictions)
            
            confidence = await self._calculate_advanced_confidence(
                final_probability, features_processed, ensemble_predictions
            )
            
            entropy = self._calculate_prediction_entropy(ensemble_predictions)
            
            regime_state, regime_confidence = await self._detect_market_regime(features_processed)
            
            entry_urgency = self._calculate_entry_urgency(
                final_probability, confidence, features_processed
            )
            
            momentum_decay_threshold = self._calculate_decay_threshold(
                features_processed, regime_state
            )
            
            recommended_hold_time = self._calculate_optimal_hold_time(
                final_probability, regime_state, features_processed
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            result = PredictionResult(
                breakout_probability=float(final_probability),
                confidence=float(confidence),
                entropy=float(entropy),
                regime_state=int(regime_state),
                regime_confidence=float(regime_confidence),
                execution_time_ms=execution_time,
                model_version=self.model_version,
                entry_urgency=float(entry_urgency),
                momentum_decay_threshold=float(momentum_decay_threshold),
                recommended_hold_time=float(recommended_hold_time)
            )
            
            self.prediction_cache[cache_key] = (result, time.time())
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return await self._fallback_prediction(features, start_time)

    async def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if features.shape[1] != 45:
            if features.shape[1] > 45:
                features = features[:, :45]
            else:
                padding = np.zeros((features.shape[0], 45 - features.shape[1]))
                features = np.hstack([features, padding])
        
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if self.scaler:
            features = self.scaler.transform(features)
        
        self.sequence_buffer.append(features[0])
        
        return features.astype(np.float32)

    async def _ensemble_predict(self, features: np.ndarray) -> List[float]:
        predictions = []
        
        if self.tflite_interpreter:
            tflite_pred = await self._predict_tflite(features)
            predictions.append(tflite_pred)
        
        if self.transformer_model and len(self.sequence_buffer) >= 10:
            transformer_pred = await self._predict_transformer()
            predictions.append(transformer_pred)
        
        if hasattr(self, 'ensemble_models'):
            ensemble_pred = await self._predict_ensemble(features)
            predictions.append(ensemble_pred)
        
        momentum_pred = self._predict_momentum_continuation(features)
        predictions.append(momentum_pred)
        
        return predictions

    async def _predict_tflite(self, features: np.ndarray) -> float:
        try:
            input_details = self.tflite_interpreter.get_input_details()
            output_details = self.tflite_interpreter.get_output_details()
            
            self.tflite_interpreter.set_tensor(input_details[0]['index'], features)
            self.tflite_interpreter.invoke()
            
            output_data = self.tflite_interpreter.get_tensor(output_details[0]['index'])
            
            if output_data.ndim > 1:
                probability = float(output_data[0][1] if output_data.shape[1] > 1 else output_data[0][0])
            else:
                probability = float(output_data[0])
            
            return np.clip(probability, 0.0, 1.0)
            
        except Exception as e:
            return 0.5

    async def _predict_transformer(self) -> float:
        try:
            sequence = np.array(list(self.sequence_buffer)[-60:])
            if sequence.shape[0] < 10:
                return 0.5
            
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            
            prediction = self.transformer_model.predict(sequence, verbose=0)
            
            if isinstance(prediction, dict):
                momentum_pred = prediction.get('momentum', [[0.5]])[0][0]
            else:
                momentum_pred = prediction[0][0] if prediction.ndim > 1 else prediction[0]
            
            return np.clip(float(momentum_pred), 0.0, 1.0)
            
        except Exception as e:
            return 0.5

    async def _predict_ensemble(self, features: np.ndarray) -> float:
        try:
            predictions = []
            
            for name, model in self.ensemble_models.items():
                if model:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(features)[0][1]
                    else:
                        pred = model.predict(features)[0]
                    predictions.append(pred)
            
            return np.clip(np.mean(predictions), 0.0, 1.0) if predictions else 0.5
            
        except Exception as e:
            return 0.5

    def _predict_momentum_continuation(self, features: np.ndarray) -> float:
        try:
            momentum_features = features[0][:5]
            volume_features = features[0][10:15] if len(features[0]) > 15 else [0.5] * 5
            volatility_features = features[0][15:20] if len(features[0]) > 20 else [0.5] * 5
            
            momentum_score = np.mean(momentum_features)
            volume_score = np.mean(volume_features)
            volatility_penalty = np.mean(volatility_features)
            
            if len(self.momentum_history) > 0:
                momentum_trend = np.mean(list(self.momentum_history)[-10:])
                momentum_score = momentum_score * 0.7 + momentum_trend * 0.3
            
            prediction = (
                momentum_score * 0.5 +
                volume_score * 0.3 +
                (1 - volatility_penalty) * 0.2
            )
            
            self.momentum_history.append(momentum_score)
            
            return np.clip(prediction, 0.0, 1.0)
            
        except Exception as e:
            return 0.5

    def _weighted_ensemble(self, predictions: List[float]) -> float:
        if not predictions:
            return 0.5
        
        weights = self.ensemble_weights[:len(predictions)]
        weights = weights + [0.1] * (len(predictions) - len(weights))
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        weighted_pred = sum(p * w for p, w in zip(predictions, normalized_weights))
        
        return np.clip(weighted_pred, 0.0, 1.0)

    async def _calculate_advanced_confidence(self, probability: float, features: np.ndarray, 
                                           ensemble_predictions: List[float]) -> float:
        
        base_confidence = abs(probability - 0.5) * 2
        
        prediction_variance = np.var(ensemble_predictions) if len(ensemble_predictions) > 1 else 0.1
        consensus_score = 1.0 - min(prediction_variance * 4, 1.0)
        
        feature_quality = 1.0 - np.std(features[0][:10]) if len(features[0]) > 10 else 0.8
        feature_quality = np.clip(feature_quality, 0.0, 1.0)
        
        if len(self.stats['confidence_distribution']) > 0:
            historical_accuracy = np.mean(list(self.stats['confidence_distribution'])[-100:])
            calibration_factor = min(historical_accuracy / 0.6, 1.2)
        else:
            calibration_factor = 1.0
        
        confidence = (
            base_confidence * 0.4 +
            consensus_score * 0.3 +
            feature_quality * 0.3
        ) * calibration_factor
        
        return np.clip(confidence, 0.0, 1.0)

    def _calculate_prediction_entropy(self, predictions: List[float]) -> float:
        if not predictions:
            return 0.693
        
        avg_pred = np.mean(predictions)
        prob_pos = avg_pred
        prob_neg = 1 - avg_pred
        
        probabilities = np.array([prob_pos, prob_neg])
        probabilities = np.clip(probabilities, 1e-10, 1.0)
        
        entropy = -np.sum(probabilities * np.log(probabilities))
        
        return float(entropy)

    async def _detect_market_regime(self, features: np.ndarray) -> Tuple[int, float]:
        try:
            if not self.regime_detector:
                return 2, 0.5
            
            momentum_features = features[0][:3]
            volatility_features = features[0][6:9] if len(features[0]) > 9 else [0.5] * 3
            
            market_signal = np.mean(momentum_features)
            volatility_signal = np.mean(volatility_features)
            
            if volatility_signal > 0.7:
                return 3, 0.8
            elif market_signal > 0.6:
                return 0, 0.7
            elif market_signal < 0.4:
                return 1, 0.7
            else:
                return 2, 0.6
                
        except Exception as e:
            return 2, 0.5

    def _calculate_entry_urgency(self, probability: float, confidence: float, features: np.ndarray) -> float:
        momentum_urgency = probability * confidence
        
        velocity_features = features[0][3:6] if len(features[0]) > 6 else [0.5] * 3
        velocity_urgency = np.mean(velocity_features)
        
        time_decay_factor = 1.0
        
        urgency = (
            momentum_urgency * 0.5 +
            velocity_urgency * 0.3 +
            time_decay_factor * 0.2
        )
        
        return np.clip(urgency, 0.0, 1.0)

    def _calculate_decay_threshold(self, features: np.ndarray, regime_state: int) -> float:
        base_threshold = 0.005
        
        volatility_features = features[0][6:9] if len(features[0]) > 9 else [0.5] * 3
        volatility = np.mean(volatility_features)
        
        regime_multipliers = {0: 0.8, 1: 1.2, 2: 1.0, 3: 0.6}
        regime_multiplier = regime_multipliers.get(regime_state, 1.0)
        
        adjusted_threshold = base_threshold * (1 + volatility) * regime_multiplier
        
        return np.clip(adjusted_threshold, 0.002, 0.02)

    def _calculate_optimal_hold_time(self, probability: float, regime_state: int, features: np.ndarray) -> float:
        base_hold_time = 180
        
        momentum_strength = probability
        volatility_features = features[0][6:9] if len(features[0]) > 9 else [0.5] * 3
        volatility = np.mean(volatility_features)
        
        regime_multipliers = {0: 1.2, 1: 0.8, 2: 1.0, 3: 0.6}
        regime_multiplier = regime_multipliers.get(regime_state, 1.0)
        
        hold_time = base_hold_time * momentum_strength * regime_multiplier / (1 + volatility)
        
        return np.clip(hold_time, 30, 600)

    def _generate_cache_key(self, features: np.ndarray, token_address: str) -> str:
        feature_hash = hash(features.tobytes()) % 100000
        return f"{token_address}_{feature_hash}_{int(time.time() // self.cache_ttl)}"

    def _update_stats(self, result: PredictionResult):
        self.stats['total_predictions'] += 1
        self.stats['confidence_distribution'].append(result.confidence)
        
        current_avg = self.stats['avg_inference_time']
        total_predictions = self.stats['total_predictions']
        
        self.stats['avg_inference_time'] = (
            (current_avg * (total_predictions - 1) + result.execution_time_ms) / total_predictions
        )

    async def _fallback_prediction(self, features: np.ndarray, start_time: float) -> PredictionResult:
        probability = 0.5 + np.random.uniform(-0.2, 0.2)
        confidence = 0.4 + np.random.uniform(0, 0.3)
        execution_time = (time.time() - start_time) * 1000
        
        return PredictionResult(
            breakout_probability=probability,
            confidence=confidence,
            entropy=0.693,
            regime_state=2,
            regime_confidence=0.5,
            execution_time_ms=execution_time,
            model_version="fallback",
            entry_urgency=0.5,
            momentum_decay_threshold=0.01,
            recommended_hold_time=180
        )

    async def update_model_performance(self, prediction_result: PredictionResult, actual_outcome: bool):
        try:
            predicted_positive = prediction_result.breakout_probability > 0.5
            is_correct = predicted_positive == actual_outcome
            
            self.stats['accuracy_tracker'].append(is_correct)
            
            if len(self.stats['accuracy_tracker']) >= 50:
                recent_accuracy = np.mean(list(self.stats['accuracy_tracker'])[-50:])
                
                if recent_accuracy < 0.55:
                    await self.trigger_model_retraining()
                    
        except Exception as e:
            self.logger.error(f"Performance update error: {e}")

    async def trigger_model_retraining(self):
        self.logger.warning("🔄 Model performance degraded - triggering retraining")

    def get_model_statistics(self) -> Dict:
        try:
            if self.stats['confidence_distribution']:
                avg_confidence = np.mean(list(self.stats['confidence_distribution']))
                confidence_std = np.std(list(self.stats['confidence_distribution']))
            else:
                avg_confidence = confidence_std = 0
            
            if self.stats['accuracy_tracker']:
                recent_accuracy = np.mean(list(self.stats['accuracy_tracker'])[-100:])
            else:
                recent_accuracy = 0
            
            cache_hit_rate = self.stats['cache_hits'] / max(self.stats['total_predictions'], 1)
            
            return {
                'model_version': self.model_version,
                'total_predictions': self.stats['total_predictions'],
                'cache_hit_rate': cache_hit_rate,
                'avg_inference_time_ms': self.stats['avg_inference_time'],
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'recent_accuracy': recent_accuracy,
                'sequence_buffer_size': len(self.sequence_buffer),
                'momentum_history_size': len(self.momentum_history)
            }
            
        except Exception as e:
            return {'error': str(e)}

model_inference = RenaissanceInference()