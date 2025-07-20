import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from collections import deque
import asyncio

@dataclass
class MarketRegime:
    regime_id: int
    name: str
    volatility_range: Tuple[float, float]
    momentum_characteristics: Dict[str, float]
    model_weights: Dict[str, float]
    duration_probability: float

class RegimeDetector:
    def __init__(self, n_regimes: int = 4, window_size: int = 100):
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.gmm_model = GaussianMixture(n_components=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self.feature_history = deque(maxlen=window_size)
        self.regime_history = deque(maxlen=window_size)
        
        self.regime_definitions = {
            0: MarketRegime(0, "Low Volatility Bull", (0.0, 0.1), 
                           {"momentum": 0.7, "mean_reversion": 0.3}, 
                           {"transformer": 0.6, "momentum": 0.4}, 0.25),
            1: MarketRegime(1, "High Volatility Bull", (0.1, 0.3), 
                           {"momentum": 0.9, "mean_reversion": 0.1}, 
                           {"transformer": 0.8, "momentum": 0.2}, 0.15),
            2: MarketRegime(2, "Sideways/Choppy", (0.05, 0.15), 
                           {"momentum": 0.3, "mean_reversion": 0.7}, 
                           {"transformer": 0.3, "momentum": 0.7}, 0.40),
            3: MarketRegime(3, "Bear/Declining", (0.1, 0.4), 
                           {"momentum": 0.2, "mean_reversion": 0.8}, 
                           {"transformer": 0.2, "momentum": 0.8}, 0.20)
        }
    
    def extract_regime_features(self, price_data: List[float], 
                               volume_data: List[float]) -> np.ndarray:
        
        if len(price_data) < 20:
            return np.array([0.1, 0.0, 0.0, 0.5, 0.0])
        
        prices = np.array(price_data)
        volumes = np.array(volume_data) if volume_data else np.ones_like(prices)
        
        returns = np.diff(prices) / prices[:-1]
        
        volatility = np.std(returns)
        
        momentum = np.mean(returns[-10:]) if len(returns) >= 10 else 0.0
        
        mean_reversion = -np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 10 else 0.0
        
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0] if len(volumes) > 5 else 0.0
        
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 5 else 0.0
        
        features = np.array([
            volatility,
            momentum,
            mean_reversion,
            volume_trend / np.mean(volumes) if np.mean(volumes) > 0 else 0.0,
            autocorr
        ])
        
        return np.nan_to_num(features)
    
    def fit(self, historical_features: List[np.ndarray]):
        
        if len(historical_features) < 50:
            print("Warning: Insufficient data for regime detection. Using default regimes.")
            return
        
        feature_matrix = np.array(historical_features)
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        self.gmm_model.fit(feature_matrix_scaled)
        self.is_fitted = True
        
        regime_labels = self.gmm_model.predict(feature_matrix_scaled)
        self._update_regime_characteristics(feature_matrix, regime_labels)
    
    def _update_regime_characteristics(self, features: np.ndarray, labels: np.ndarray):
        
        for regime_id in range(self.n_regimes):
            regime_features = features[labels == regime_id]
            
            if len(regime_features) > 5:
                volatility_range = (
                    np.percentile(regime_features[:, 0], 25),
                    np.percentile(regime_features[:, 0], 75)
                )
                
                avg_momentum = np.mean(regime_features[:, 1])
                avg_mean_reversion = np.mean(regime_features[:, 2])
                
                if regime_id in self.regime_definitions:
                    self.regime_definitions[regime_id].volatility_range = volatility_range
                    self.regime_definitions[regime_id].momentum_characteristics = {
                        "momentum": max(0, avg_momentum),
                        "mean_reversion": max(0, avg_mean_reversion)
                    }
    
    def detect_current_regime(self, price_data: List[float], 
                            volume_data: List[float]) -> Tuple[int, float]:
        
        features = self.extract_regime_features(price_data, volume_data)
        self.feature_history.append(features)
        
        if not self.is_fitted:
            regime_id = self._heuristic_regime_detection(features)
            confidence = 0.6
        else:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            regime_probabilities = self.gmm_model.predict_proba(features_scaled)[0]
            regime_id = np.argmax(regime_probabilities)
            confidence = regime_probabilities[regime_id]
        
        self.regime_history.append(regime_id)
        
        return regime_id, confidence
    
    def _heuristic_regime_detection(self, features: np.ndarray) -> int:
        
        volatility, momentum, mean_reversion, volume_trend, autocorr = features
        
        if volatility < 0.05 and momentum > 0:
            return 0  # Low Vol Bull
        elif volatility > 0.15 and momentum > 0:
            return 1  # High Vol Bull
        elif abs(momentum) < 0.01 and mean_reversion > 0:
            return 2  # Sideways
        else:
            return 3  # Bear
    
    def get_regime_transition_probability(self, current_regime: int, 
                                        target_regime: int) -> float:
        
        if len(self.regime_history) < 10:
            return 0.25
        
        recent_regimes = list(self.regime_history)[-20:]
        transitions = 0
        current_to_target = 0
        
        for i in range(len(recent_regimes) - 1):
            if recent_regimes[i] == current_regime:
                transitions += 1
                if recent_regimes[i + 1] == target_regime:
                    current_to_target += 1
        
        return current_to_target / transitions if transitions > 0 else 0.25

class RegimeAwareModelEnsemble:
    def __init__(self):
        self.models = {}
        self.regime_detector = RegimeDetector()
        self.model_weights_cache = {}
        
        self.base_models = [
            'transformer',
            'momentum_neural_net', 
            'mean_reversion_model',
            'volatility_model'
        ]
        
    async def initialize_models(self):
        
        for model_name in self.base_models:
            try:
                if model_name == 'transformer':
                    self.models[model_name] = tf.lite.Interpreter(
                        model_path='models/renaissance_transformer.tflite'
                    )
                    self.models[model_name].allocate_tensors()
                else:
                    self.models[model_name] = joblib.load(f'models/{model_name}.pkl')
            except:
                self.models[model_name] = self._create_fallback_model(model_name)
    
    def _create_fallback_model(self, model_name: str):
        
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, random_state=42)
    
    async def predict_with_regime_awareness(self, features: np.ndarray,
                                          price_history: List[float],
                                          volume_history: List[float]) -> Tuple[float, Dict]:
        
        current_regime, regime_confidence = self.regime_detector.detect_current_regime(
            price_history, volume_history
        )
        
        regime_info = self.regime_detector.regime_definitions[current_regime]
        model_weights = regime_info.model_weights
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'transformer' and hasattr(model, 'get_input_details'):
                    pred = await self._predict_transformer(model, features)
                else:
                    pred = await self._predict_sklearn_model(model, features)
                predictions[model_name] = pred
            except:
                predictions[model_name] = 0.5
        
        regime_weights = self._calculate_regime_specific_weights(
            current_regime, regime_confidence, predictions
        )
        
        ensemble_prediction = sum(
            predictions.get(model, 0.5) * regime_weights.get(model, 0.25)
            for model in self.base_models
        )
        
        ensemble_prediction = np.clip(ensemble_prediction, 0.0, 1.0)
        
        meta_info = {
            'regime': regime_info.name,
            'regime_confidence': regime_confidence,
            'model_weights': regime_weights,
            'individual_predictions': predictions,
            'regime_id': current_regime
        }
        
        return ensemble_prediction, meta_info
    
    async def _predict_transformer(self, model, features: np.ndarray) -> float:
        
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        if len(features.shape) == 1:
            features = features.reshape(1, 1, -1)
        elif len(features.shape) == 2:
            features = features.reshape(1, features.shape[0], features.shape[1])
        
        input_shape = input_details[0]['shape']
        if features.shape != tuple(input_shape):
            features = np.random.random((1, 120, 45)).astype(np.float32)
        
        model.set_tensor(input_details[0]['index'], features.astype(np.float32))
        model.invoke()
        
        prediction = model.get_tensor(output_details[0]['index'])[0][0]
        return float(prediction)
    
    async def _predict_sklearn_model(self, model, features: np.ndarray) -> float:
        
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(features.reshape(1, -1))[0]
            return prob[1] if len(prob) > 1 else prob[0]
        else:
            return float(model.predict(features.reshape(1, -1))[0])
    
    def _calculate_regime_specific_weights(self, regime_id: int, confidence: float,
                                         predictions: Dict[str, float]) -> Dict[str, float]:
        
        base_weights = self.regime_detector.regime_definitions[regime_id].model_weights
        
        performance_adjustment = self._calculate_performance_weights(predictions)
        
        final_weights = {}
        total_weight = 0
        
        for model in self.base_models:
            base_weight = base_weights.get(model, 0.25)
            perf_weight = performance_adjustment.get(model, 1.0)
            confidence_weight = confidence if model == 'transformer' else (1 - confidence) * 0.5 + 0.5
            
            final_weight = base_weight * perf_weight * confidence_weight
            final_weights[model] = final_weight
            total_weight += final_weight
        
        if total_weight > 0:
            final_weights = {k: v / total_weight for k, v in final_weights.items()}
        else:
            final_weights = {model: 0.25 for model in self.base_models}
        
        return final_weights
    
    def _calculate_performance_weights(self, predictions: Dict[str, float]) -> Dict[str, float]:
        
        pred_values = list(predictions.values())
        if len(pred_values) == 0:
            return {model: 1.0 for model in self.base_models}
        
        mean_pred = np.mean(pred_values)
        
        weights = {}
        for model, pred in predictions.items():
            confidence_from_consensus = 1.0 - abs(pred - mean_pred)
            weights[model] = confidence_from_consensus
        
        return weights
    
    async def update_regime_model(self, historical_data: List[Dict]):
        
        features_list = []
        for data_point in historical_data:
            features = self.regime_detector.extract_regime_features(
                data_point.get('price_history', []),
                data_point.get('volume_history', [])
            )
            features_list.append(features)
        
        if len(features_list) >= 50:
            self.regime_detector.fit(features_list)
    
    def save_ensemble_state(self, filepath: str = 'models/regime_ensemble_state.pkl'):
        
        state = {
            'regime_detector': self.regime_detector,
            'model_weights_cache': self.model_weights_cache
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_ensemble_state(self, filepath: str = 'models/regime_ensemble_state.pkl'):
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.regime_detector = state['regime_detector']
            self.model_weights_cache = state.get('model_weights_cache', {})
            
            return True
        except:
            return False

regime_ensemble = RegimeAwareModelEnsemble()