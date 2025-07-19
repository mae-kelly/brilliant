import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import numpy as np
import tensorflow as tf
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import threading

@dataclass
class TradeResult:
    features: np.ndarray
    prediction: float
    actual_outcome: int
    roi: float
    timestamp: float
    confidence: float

class OnlineLearner:
    def __init__(self):
        self.models = {
            'sgd': SGDClassifier(
                loss='log_loss',
                learning_rate='adaptive',
                eta0=0.01,
                alpha=0.0001,
                max_iter=1
            ),
            'passive_aggressive': PassiveAggressiveClassifier(
                C=0.01,
                max_iter=1,
                random_state=42
            ),
            'ensemble': None
        }
        
        self.scalers = {
            'sgd': StandardScaler(),
            'passive_aggressive': StandardScaler()
        }
        
        self.trade_results = deque(maxlen=10000)
        self.model_weights = {'sgd': 0.4, 'passive_aggressive': 0.3, 'ensemble': 0.3}
        self.performance_metrics = {
            'sgd': deque(maxlen=100),
            'passive_aggressive': deque(maxlen=100),
            'ensemble': deque(maxlen=100)
        }
        
        self.lock = threading.Lock()
        self.update_count = 0
        self.last_ensemble_retrain = 0
        
        self.feature_names = [
            'price_delta', 'volume_delta', 'liquidity_delta',
            'volatility', 'velocity', 'momentum',
            'order_flow_imbalance', 'microstructure_noise',
            'jump_intensity', 'volume_profile_anomaly'
        ]

    async def update_on_trade_result(self, features: np.ndarray, prediction: float, 
                                   actual_outcome: int, roi: float, confidence: float):
        with self.lock:
            trade_result = TradeResult(
                features=features,
                prediction=prediction,
                actual_outcome=actual_outcome,
                roi=roi,
                timestamp=time.time(),
                confidence=confidence
            )
            
            self.trade_results.append(trade_result)
            
            await self.incremental_update(features, actual_outcome)
            
            self.update_count += 1
            
            if self.update_count % 50 == 0:
                await self.evaluate_model_performance()
            
            if self.update_count % 200 == 0:
                await self.retrain_ensemble()

    async def incremental_update(self, features: np.ndarray, outcome: int):
        try:
            for model_name in ['sgd', 'passive_aggressive']:
                model = self.models[model_name]
                scaler = self.scalers[model_name]
                
                if not hasattr(model, 'classes_'):
                    if len(self.trade_results) >= 10:
                        initial_data = self.get_initial_training_data()
                        X_init, y_init = initial_data
                        X_scaled = scaler.fit_transform(X_init)
                        model.fit(X_scaled, y_init)
                else:
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    model.partial_fit(features_scaled, [outcome])
                    
        except Exception as e:
            pass

    def get_initial_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.trade_results) < 10:
            return np.array([]), np.array([])
        
        features_list = []
        outcomes_list = []
        
        for result in list(self.trade_results)[-50:]:
            features_list.append(result.features)
            outcomes_list.append(result.actual_outcome)
        
        return np.array(features_list), np.array(outcomes_list)

    async def evaluate_model_performance(self):
        if len(self.trade_results) < 20:
            return
        
        recent_results = list(self.trade_results)[-50:]
        
        for model_name in ['sgd', 'passive_aggressive']:
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            if not hasattr(model, 'classes_'):
                continue
            
            correct_predictions = 0
            total_predictions = 0
            
            for result in recent_results:
                try:
                    features_scaled = scaler.transform(result.features.reshape(1, -1))
                    prediction = model.predict(features_scaled)[0]
                    
                    if prediction == result.actual_outcome:
                        correct_predictions += 1
                    total_predictions += 1
                    
                except Exception as e:
                    continue
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            self.performance_metrics[model_name].append(accuracy)
        
        await self.update_model_weights()

    async def update_model_weights(self):
        total_weight = 0.0
        new_weights = {}
        
        for model_name in ['sgd', 'passive_aggressive']:
            if self.performance_metrics[model_name]:
                avg_performance = np.mean(list(self.performance_metrics[model_name])[-10:])
                new_weights[model_name] = max(avg_performance, 0.1)
                total_weight += new_weights[model_name]
        
        if total_weight > 0:
            for model_name in new_weights:
                self.model_weights[model_name] = new_weights[model_name] / total_weight * 0.7
            
            self.model_weights['ensemble'] = 0.3

    async def retrain_ensemble(self):
        if time.time() - self.last_ensemble_retrain < 300:
            return
        
        if len(self.trade_results) < 100:
            return
        
        try:
            features_list = []
            outcomes_list = []
            
            for result in list(self.trade_results)[-500:]:
                features_list.append(result.features)
                outcomes_list.append(result.actual_outcome)
            
            X = np.array(features_list)
            y = np.array(outcomes_list)
            
            self.models['ensemble'] = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=2
            )
            
            self.models['ensemble'].fit(X, y)
            
            ensemble_accuracy = self.models['ensemble'].score(X, y)
            self.performance_metrics['ensemble'].append(ensemble_accuracy)
            
            self.last_ensemble_retrain = time.time()
            
            print(f"🌳 Ensemble retrained: {ensemble_accuracy:.3f} accuracy")
            
        except Exception as e:
            pass

    async def predict(self, features: np.ndarray) -> Tuple[float, float]:
        predictions = {}
        confidences = {}
        
        for model_name, model in self.models.items():
            if model is None or not hasattr(model, 'predict'):
                continue
            
            try:
                if model_name in ['sgd', 'passive_aggressive']:
                    if not hasattr(model, 'classes_'):
                        continue
                    
                    scaler = self.scalers[model_name]
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_scaled)[0]
                        predictions[model_name] = proba[1] if len(proba) > 1 else 0.5
                        confidences[model_name] = max(proba) - min(proba)
                    else:
                        prediction = model.predict(features_scaled)[0]
                        predictions[model_name] = float(prediction)
                        confidences[model_name] = 0.6
                
                elif model_name == 'ensemble':
                    proba = model.predict_proba(features.reshape(1, -1))[0]
                    predictions[model_name] = proba[1] if len(proba) > 1 else 0.5
                    confidences[model_name] = max(proba) - min(proba)
                    
            except Exception as e:
                continue
        
        if not predictions:
            return 0.5, 0.5
        
        weighted_prediction = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            weight = self.model_weights.get(model_name, 0.0)
            weighted_prediction += prediction * weight
            weighted_confidence += confidences.get(model_name, 0.5) * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_prediction /= total_weight
            weighted_confidence /= total_weight
        
        return float(weighted_prediction), float(weighted_confidence)

    async def adapt_to_market_regime(self, market_volatility: float, volume_surge: float):
        if market_volatility > 0.15:
            self.model_weights['sgd'] *= 1.2
            self.model_weights['passive_aggressive'] *= 0.8
        elif market_volatility < 0.05:
            self.model_weights['sgd'] *= 0.8
            self.model_weights['passive_aggressive'] *= 1.2
        
        if volume_surge > 2.0:
            self.model_weights['ensemble'] *= 1.1
        
        total_weight = sum(self.model_weights.values())
        for model_name in self.model_weights:
            self.model_weights[model_name] /= total_weight

    def get_performance_stats(self) -> Dict:
        stats = {
            'total_trades': len(self.trade_results),
            'update_count': self.update_count,
            'model_weights': self.model_weights.copy()
        }
        
        for model_name, metrics in self.performance_metrics.items():
            if metrics:
                stats[f'{model_name}_accuracy'] = np.mean(list(metrics)[-10:])
        
        if len(self.trade_results) > 0:
            recent_results = list(self.trade_results)[-100:]
            roi_values = [r.roi for r in recent_results]
            stats['recent_avg_roi'] = np.mean(roi_values)
            stats['recent_win_rate'] = sum(1 for r in recent_results if r.roi > 0) / len(recent_results)
        
        return stats

    async def save_models(self, path_prefix: str = 'models/online_'):
        try:
            for model_name, model in self.models.items():
                if model is not None and hasattr(model, 'predict'):
                    joblib.dump(model, f"{path_prefix}{model_name}.pkl")
            
            for scaler_name, scaler in self.scalers.items():
                joblib.dump(scaler, f"{path_prefix}scaler_{scaler_name}.pkl")
            
            metadata = {
                'model_weights': self.model_weights,
                'update_count': self.update_count,
                'last_ensemble_retrain': self.last_ensemble_retrain,
                'performance_metrics': {
                    name: list(metrics) for name, metrics in self.performance_metrics.items()
                }
            }
            
            with open(f"{path_prefix}metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            pass

    async def load_models(self, path_prefix: str = 'models/online_'):
        try:
            for model_name in ['sgd', 'passive_aggressive', 'ensemble']:
                try:
                    self.models[model_name] = joblib.load(f"{path_prefix}{model_name}.pkl")
                except:
                    continue
            
            for scaler_name in ['sgd', 'passive_aggressive']:
                try:
                    self.scalers[scaler_name] = joblib.load(f"{path_prefix}scaler_{scaler_name}.pkl")
                except:
                    continue
            
            try:
                with open(f"{path_prefix}metadata.json", 'r') as f:
                    metadata = json.load(f)
                    self.model_weights = metadata.get('model_weights', self.model_weights)
                    self.update_count = metadata.get('update_count', 0)
                    self.last_ensemble_retrain = metadata.get('last_ensemble_retrain', 0)
            except:
                pass
                
        except Exception as e:
            pass

online_learner = OnlineLearner()
