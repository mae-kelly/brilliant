import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from typing import List, Tuple, Dict

class EnsembleModel:
    def __init__(self):
        self.models = {
            'neural_net': None,
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10),
            'gradient_boost': GradientBoostingClassifier(n_estimators=50, max_depth=5),
            'logistic': LogisticRegression(max_iter=1000)
        }
        self.weights = {'neural_net': 0.4, 'random_forest': 0.3, 'gradient_boost': 0.2, 'logistic': 0.1}
        self.trained = False
        
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        self.models['neural_net'] = model
        
        for name, model in self.models.items():
            if name != 'neural_net':
                model.fit(X_train, y_train)
        
        self.trained = True
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[float, float]:
        if not self.trained:
            raise ValueError("Model not trained")
        
        predictions = {}
        
        if self.models['neural_net']:
            predictions['neural_net'] = float(self.models['neural_net'].predict(X)[0][0])
        
        for name, model in self.models.items():
            if name != 'neural_net':
                pred_proba = model.predict_proba(X)[0]
                predictions[name] = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
        
        weighted_pred = sum(pred * self.weights[name] for name, pred in predictions.items())
        
        variance = sum(self.weights[name] * (pred - weighted_pred)**2 for name, pred in predictions.items())
        confidence = 1 / (1 + variance)
        
        return weighted_pred, confidence
    
    def update_weights(self, performance_scores: Dict[str, float]) -> None:
        total_performance = sum(performance_scores.values())
        if total_performance > 0:
            for name in self.weights:
                if name in performance_scores:
                    self.weights[name] = performance_scores[name] / total_performance

ensemble_model = EnsembleModel()
