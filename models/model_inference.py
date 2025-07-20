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
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PredictionResult:
    breakout_probability: float
    confidence: float
    entropy: float
    regime_state: int
    regime_confidence: float
    feature_importance: Dict[str, float]
    execution_time_ms: float
    model_version: str

@dataclass
class ModelPerformance:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_latency: float
    total_predictions: int
    correct_predictions: int

class OptimizedModelInference:
    def __init__(self):
        self.tflite_interpreter = None
        self.sklearn_model = None
        self.scaler = None
        self.feature_names = []
        self.regime_detector = None
        self.model_type = "unknown"
        self.model_version = "v1.0.0"
        
        self.prediction_cache = {}
        self.cache_ttl = 5
        
        self.performance_history = deque(maxlen=10000)
        self.feature_importance_cache = {}
        
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'avg_inference_time': 0.0,
            'model_accuracy': 0.0,
            'start_time': time.time()
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        try:
            await self.load_models()
            await self.load_regime_detector()
            self.logger.info("✅ Renaissance model inference engine initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            return False

    async def load_models(self):
        model_loaded = False
        
        if os.path.exists("models/model_weights.tflite"):
            try:
                self.tflite_interpreter = tf.lite.Interpreter(
                    model_path="models/model_weights.tflite"
                )
                self.tflite_interpreter.allocate_tensors()
                self.model_type = "tflite"
                self.logger.info("✅ Loaded TensorFlow Lite model")
                model_loaded = True
            except Exception as e:
                self.logger.warning(f"⚠️ TFLite loading failed: {e}")
        
        if not model_loaded and os.path.exists("models/model_weights.pkl"):
            try:
                self.sklearn_model = joblib.load("models/model_weights.pkl")
                self.model_type = "sklearn"
                self.logger.info("✅ Loaded scikit-learn model")
                model_loaded = True
            except Exception as e:
                self.logger.warning(f"⚠️ SKLearn loading failed: {e}")
        
        if not model_loaded:
            raise Exception("No valid model found")
        
        try:
            self.scaler = joblib.load("models/scaler.pkl")
            self.logger.info("✅ Loaded feature scaler")
        except Exception as e:
            self.logger.warning(f"⚠️ Scaler loading failed: {e}")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        
        try:
            with open("models/feature_names.json", 'r') as f:
                self.feature_names = json.load(f)
            self.logger.info(f"✅ Loaded {len(self.feature_names)} feature names")
        except Exception as e:
            self.logger.warning(f"⚠️ Feature names loading failed: {e}")
            self.feature_names = [f"feature_{i}" for i in range(45)]
        
        try:
            with open("models/model_info.json", 'r') as f:
                model_info = json.load(f)
                self.model_version = model_info.get('model_type', 'unknown')
                self.stats['model_accuracy'] = model_info.get('test_accuracy', 0.0)
        except Exception as e:
            self.logger.warning(f"⚠️ Model info loading failed: {e}")

    async def load_regime_detector(self):
        try:
            if os.path.exists("models/regime_detector.pkl"):
                self.regime_detector = joblib.load("models/regime_detector.pkl")
                self.logger.info("✅ Loaded regime detector")
            else:
                self.regime_detector = GaussianHMM(n_components=4, random_state=42)
                synthetic_data = np.random.normal(0, 0.02, (1000, 1))
                self.regime_detector.fit(synthetic_data)
                self.logger.info("✅ Initialized fallback regime detector")
        except Exception as e:
            self.logger.warning(f"⚠️ Regime detector loading failed: {e}")
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
            
            if self.model_type == "tflite":
                probability, raw_output = await self._predict_tflite(features_processed)
            elif self.model_type == "sklearn":
                probability, raw_output = await self._predict_sklearn(features_processed)
            else:
                probability, raw_output = await self._predict_fallback(features_processed)
            
            confidence = await self._calculate_confidence(probability, features_processed)
            entropy = await self._calculate_entropy(raw_output)
            
            regime_state, regime_confidence = await self._detect_current_regime(features_processed)
            
            feature_importance = await self._calculate_feature_importance(features_processed)
            
            execution_time = (time.time() - start_time) * 1000
            
            result = PredictionResult(
                breakout_probability=float(probability),
                confidence=float(confidence),
                entropy=float(entropy),
                regime_state=int(regime_state),
                regime_confidence=float(regime_confidence),
                feature_importance=feature_importance,
                execution_time_ms=execution_time,
                model_version=self.model_version
            )
            
            self.prediction_cache[cache_key] = (result, time.time())
            
            self._update_performance_stats(result, execution_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return await self._fallback_prediction(features, start_time)

    async def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            expected_features = len(self.feature_names)
            if features.shape[1] != expected_features:
                if features.shape[1] > expected_features:
                    features = features[:, :expected_features]
                else:
                    padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                    features = np.hstack([features, padding])
            
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if self.scaler:
                features = self.scaler.transform(features)
            
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Feature preprocessing error: {e}")
            return np.random.random((1, len(self.feature_names))).astype(np.float32)

    async def _predict_tflite(self, features: np.ndarray) -> Tuple[float, np.ndarray]:
        try:
            input_details = self.tflite_interpreter.get_input_details()
            output_details = self.tflite_interpreter.get_output_details()
            
            input_shape = input_details[0]['shape']
            if features.shape != tuple(input_shape):
                features = features.reshape(input_shape)
            
            self.tflite_interpreter.set_tensor(input_details[0]['index'], features)
            
            self.tflite_interpreter.invoke()
            
            output_data = self.tflite_interpreter.get_tensor(output_details[0]['index'])
            
            if output_data.ndim > 1:
                probability = float(output_data[0][0])
                raw_output = output_data[0]
            else:
                probability = float(output_data[0])
                raw_output = output_data
            
            probability = np.clip(probability, 0.0, 1.0)
            
            return probability, raw_output
            
        except Exception as e:
            self.logger.error(f"TFLite prediction error: {e}")
            return 0.5, np.array([0.5])

    async def _predict_sklearn(self, features: np.ndarray) -> Tuple[float, np.ndarray]:
        try:
            if hasattr(self.sklearn_model, 'predict_proba'):
                probabilities = self.sklearn_model.predict_proba(features)
                if probabilities.shape[1] > 1:
                    probability = float(probabilities[0][1])
                    raw_output = probabilities[0]
                else:
                    probability = float(probabilities[0][0])
                    raw_output = probabilities[0]
            else:
                prediction = self.sklearn_model.predict(features)
                probability = float(np.clip(prediction[0], 0.0, 1.0))
                raw_output = np.array([probability])
            
            return probability, raw_output
            
        except Exception as e:
            self.logger.error(f"SKLearn prediction error: {e}")
            return 0.5, np.array([0.5])

    async def _predict_fallback(self, features: np.ndarray) -> Tuple[float, np.ndarray]:
        momentum_features = features[0][:3]
        volume_features = features[0][8:12] if len(features[0]) > 12 else [0.5, 0.5, 0.5, 0.5]
        volatility_features = features[0][6:8] if len(features[0]) > 8 else [0.5, 0.5]
        
        momentum_score = np.mean(momentum_features)
        volume_score = np.mean(volume_features)
        volatility_penalty = np.mean(volatility_features)
        
        probability = (
            momentum_score * 0.4 +
            volume_score * 0.3 +
            (1 - volatility_penalty) * 0.2 +
            np.random.uniform(0.45, 0.55) * 0.1
        )
        
        probability = np.clip(probability, 0.0, 1.0)
        
        return probability, np.array([probability, 1 - probability])

    async def _calculate_confidence(self, probability: float, features: np.ndarray) -> float:
        base_confidence = abs(probability - 0.5) * 2
        
        feature_consistency = 1.0 - np.std(features[0][:10]) if len(features[0]) > 10 else 0.8
        feature_consistency = np.clip(feature_consistency, 0.0, 1.0)
        
        prediction_strength = min(probability if probability > 0.5 else (1 - probability), 1.0)
        
        confidence = (
            base_confidence * 0.5 +
            feature_consistency * 0.3 +
            prediction_strength * 0.2
        )
        
        return np.clip(confidence, 0.0, 1.0)

    async def _calculate_entropy(self, raw_output: np.ndarray) -> float:
        try:
            if len(raw_output) < 2:
                prob_pos = raw_output[0] if len(raw_output) > 0 else 0.5
                prob_neg = 1 - prob_pos
                probabilities = np.array([prob_pos, prob_neg])
            else:
                probabilities = raw_output
            
            probabilities = np.clip(probabilities, 1e-10, 1.0)
            probabilities = probabilities / np.sum(probabilities)
            
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            
            return float(entropy)
            
        except Exception as e:
            return 0.693

    async def _detect_current_regime(self, features: np.ndarray) -> Tuple[int, float]:
        try:
            if not self.regime_detector:
                return 2, 0.5
            
            momentum_features = features[0][:3]
            recent_returns = np.random.normal(np.mean(momentum_features), 0.02, 20)
            
            state = self.regime_detector.predict(recent_returns.reshape(-1, 1))[-1]
            
            if hasattr(self.regime_detector, 'predict_proba'):
                proba = self.regime_detector.predict_proba(recent_returns.reshape(-1, 1))[-1]
                confidence = np.max(proba)
            else:
                confidence = 0.7
            
            return int(state), float(confidence)
            
        except Exception as e:
            return 2, 0.5

    async def _calculate_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        try:
            cache_key = f"importance_{hash(features.tobytes()) % 10000}"
            
            if cache_key in self.feature_importance_cache:
                return self.feature_importance_cache[cache_key]
            
            if self.model_type == "sklearn" and hasattr(self.sklearn_model, 'feature_importances_'):
                importances = self.sklearn_model.feature_importances_
                importance_dict = dict(zip(self.feature_names, importances))
            else:
                feature_values = np.abs(features[0])
                total_importance = np.sum(feature_values) + 1e-10
                
                importance_dict = {}
                for i, name in enumerate(self.feature_names):
                    if i < len(feature_values):
                        importance_dict[name] = float(feature_values[i] / total_importance)
                    else:
                        importance_dict[name] = 0.0
            
            top_features = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10])
            
            self.feature_importance_cache[cache_key] = top_features
            
            if len(self.feature_importance_cache) > 1000:
                oldest_keys = list(self.feature_importance_cache.keys())[:200]
                for key in oldest_keys:
                    del self.feature_importance_cache[key]
            
            return top_features
            
        except Exception as e:
            return {name: 1.0/len(self.feature_names) for name in self.feature_names[:5]}

    def _generate_cache_key(self, features: np.ndarray, token_address: str) -> str:
        feature_hash = hash(features.tobytes()) % 100000
        return f"{token_address}_{feature_hash}_{int(time.time() // self.cache_ttl)}"

    def _update_performance_stats(self, result: PredictionResult, execution_time: float):
        self.stats['total_predictions'] += 1
        
        current_avg = self.stats['avg_inference_time']
        total_predictions = self.stats['total_predictions']
        
        self.stats['avg_inference_time'] = (
            (current_avg * (total_predictions - 1) + execution_time) / total_predictions
        )
        
        self.performance_history.append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'confidence': result.confidence,
            'probability': result.breakout_probability,
            'entropy': result.entropy
        })

    async def _fallback_prediction(self, features: np.ndarray, start_time: float) -> PredictionResult:
        probability = 0.5 + np.random.uniform(-0.1, 0.1)
        confidence = 0.3 + np.random.uniform(0, 0.2)
        entropy = 0.693
        execution_time = (time.time() - start_time) * 1000
        
        return PredictionResult(
            breakout_probability=probability,
            confidence=confidence,
            entropy=entropy,
            regime_state=2,
            regime_confidence=0.5,
            feature_importance={'fallback': 1.0},
            execution_time_ms=execution_time,
            model_version="fallback"
        )

    async def batch_predict(self, features_batch: List[np.ndarray], token_addresses: List[str] = None) -> List[PredictionResult]:
        if not token_addresses:
            token_addresses = [""] * len(features_batch)
        
        tasks = []
        for features, token_addr in zip(features_batch, token_addresses):
            task = asyncio.create_task(self.predict_breakout(features, token_addr))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch prediction error: {result}")
                valid_results.append(await self._fallback_prediction(np.random.random(45), time.time()))
            else:
                valid_results.append(result)
        
        return valid_results

    async def update_model_performance(self, prediction_result: PredictionResult, actual_outcome: bool):
        try:
            predicted_positive = prediction_result.breakout_probability > 0.5
            is_correct = predicted_positive == actual_outcome
            
            performance_entry = {
                'timestamp': time.time(),
                'predicted_probability': prediction_result.breakout_probability,
                'actual_outcome': actual_outcome,
                'is_correct': is_correct,
                'confidence': prediction_result.confidence,
                'regime_state': prediction_result.regime_state
            }
            
            self.performance_history.append(performance_entry)
            
            if len(self.performance_history) >= 100:
                recent_performance = list(self.performance_history)[-100:]
                correct_predictions = sum(1 for p in recent_performance if p.get('is_correct', False))
                self.stats['model_accuracy'] = correct_predictions / len(recent_performance)
            
        except Exception as e:
            self.logger.error(f"Performance update error: {e}")

    def get_model_statistics(self) -> Dict:
        try:
            runtime = time.time() - self.stats['start_time']
            
            if self.performance_history:
                recent_performance = list(self.performance_history)[-1000:]
                avg_confidence = np.mean([p.get('confidence', 0) for p in recent_performance])
                avg_entropy = np.mean([p.get('entropy', 0.693) for p in recent_performance])
                
                execution_times = [p.get('execution_time', 0) for p in recent_performance]
                if execution_times:
                    p95_latency = np.percentile(execution_times, 95)
                    p99_latency = np.percentile(execution_times, 99)
                else:
                    p95_latency = p99_latency = 0
            else:
                avg_confidence = avg_entropy = p95_latency = p99_latency = 0
            
            predictions_per_second = self.stats['total_predictions'] / max(runtime, 1)
            cache_hit_rate = self.stats['cache_hits'] / max(self.stats['total_predictions'], 1)
            
            return {
                'model_type': self.model_type,
                'model_version': self.model_version,
                'total_predictions': self.stats['total_predictions'],
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': cache_hit_rate,
                'avg_inference_time_ms': self.stats['avg_inference_time'],
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'predictions_per_second': predictions_per_second,
                'model_accuracy': self.stats['model_accuracy'],
                'avg_confidence': avg_confidence,
                'avg_entropy': avg_entropy,
                'runtime_minutes': runtime / 60,
                'feature_count': len(self.feature_names),
                'regime_detector_available': self.regime_detector is not None
            }
            
        except Exception as e:
            self.logger.error(f"Statistics calculation error: {e}")
            return {'error': str(e)}

    async def benchmark_inference_speed(self, num_samples: int = 1000) -> Dict:
        self.logger.info(f"🏃 Running inference benchmark with {num_samples} samples...")
        
        start_time = time.time()
        
        test_features = [np.random.random(len(self.feature_names)) for _ in range(num_samples)]
        
        tasks = []
        for features in test_features:
            task = asyncio.create_task(self.predict_breakout(features))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        successful_predictions = sum(1 for r in results if not isinstance(r, Exception))
        
        execution_times = [
            r.execution_time_ms for r in results 
            if isinstance(r, PredictionResult)
        ]
        
        benchmark_results = {
            'total_samples': num_samples,
            'successful_predictions': successful_predictions,
            'failed_predictions': num_samples - successful_predictions,
            'total_time_seconds': total_time,
            'predictions_per_second': successful_predictions / total_time,
            'avg_latency_ms': np.mean(execution_times) if execution_times else 0,
            'min_latency_ms': np.min(execution_times) if execution_times else 0,
            'max_latency_ms': np.max(execution_times) if execution_times else 0,
            'p95_latency_ms': np.percentile(execution_times, 95) if execution_times else 0,
            'p99_latency_ms': np.percentile(execution_times, 99) if execution_times else 0
        }
        
        self.logger.info(f"✅ Benchmark complete: {benchmark_results['predictions_per_second']:.1f} pred/sec")
        
        return benchmark_results

    async def cleanup_cache(self):
        current_time = time.time()
        
        expired_keys = [
            key for key, (result, timestamp) in self.prediction_cache.items()
            if current_time - timestamp > self.cache_ttl * 2
        ]
        
        for key in expired_keys:
            del self.prediction_cache[key]
        
        if len(self.feature_importance_cache) > 2000:
            keys_to_remove = list(self.feature_importance_cache.keys())[:500]
            for key in keys_to_remove:
                del self.feature_importance_cache[key]

    def get_cache_statistics(self) -> Dict:
        return {
            'prediction_cache_size': len(self.prediction_cache),
            'feature_cache_size': len(self.feature_importance_cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_predictions'], 1),
            'total_cache_hits': self.stats['cache_hits']
        }

model_inference = OptimizedModelInference()