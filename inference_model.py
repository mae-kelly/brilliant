import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import logging
from prometheus_client import Gauge, Summary
import json
import yaml
import asyncio
import time

# Handle optuna import gracefully
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available, using basic hyperparameter optimization")

# Handle TensorFlow import gracefully  
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available, using PyTorch only")

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attention_pooling = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        x = self.input_linear(x)
        
        pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_encoding
        
        transformer_out = self.transformer(x)
        
        query = transformer_out.mean(dim=1, keepdim=True)
        attended, _ = self.attention_pooling(query, transformer_out, transformer_out)
        attended = self.layer_norm(attended.squeeze(1))
        
        prediction = self.fc(attended)
        uncertainty = self.uncertainty_head(attended)
        
        return prediction, uncertainty

class MomentumEnsemble:
    def __init__(self, input_dim=11, hidden_dim=256, num_layers=3, num_heads=8):
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = TransformerEncoder(input_dim, hidden_dim, num_layers, num_heads).to(self.device)
        self.transformer_optimizer = torch.optim.AdamW(
            self.transformer.parameters(), 
            lr=0.0001, 
            weight_decay=0.01
        )
        self.transformer_criterion = nn.BCELoss()
        self.uncertainty_criterion = nn.MSELoss()
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200, 
            max_depth=6, 
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.meta_learner = xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        self.dynamic_threshold = self.settings['ml']['prediction_confidence']
        self.momentum_scores = []
        self.training_data = []
        self.training_labels = []
        self.entropy_scores = []
        self.prediction_history = []
        self.weights = self.settings['ml']['ensemble_weights']
        self.feature_importance = {}
        
        self.loss_gauge = Gauge('model_loss', 'Training loss of the ensemble model')
        self.training_time = Summary('model_training_seconds', 'Time spent training the model')
        self.entropy_gauge = Gauge('prediction_entropy', 'Shannon entropy of predictions')
        self.uncertainty_gauge = Gauge('prediction_uncertainty', 'Model uncertainty')
        self.threshold_gauge = Gauge('dynamic_threshold', 'Current dynamic threshold')

    def predict(self, features):
        try:
            if isinstance(features, pd.DataFrame):
                feature_array = features.values
            else:
                feature_array = np.array(features)
            
            if feature_array.ndim == 1:
                feature_array = feature_array.reshape(1, -1)
            
            batch_size = feature_array.shape[0]
            
            transformer_input = torch.tensor(feature_array, dtype=torch.float32).to(self.device)
            if transformer_input.ndim == 2:
                transformer_input = transformer_input.unsqueeze(1)
            
            with torch.no_grad():
                self.transformer.eval()
                transformer_pred, uncertainty = self.transformer(transformer_input)
                transformer_score = transformer_pred.squeeze().cpu().numpy()
                uncertainty_score = uncertainty.squeeze().cpu().numpy()
            
            if batch_size == 1:
                transformer_score = float(transformer_score)
                uncertainty_score = float(uncertainty_score)
            
            flat_features = feature_array.reshape(batch_size, -1)
            
            if hasattr(self.xgb_model, 'predict_proba'):
                try:
                    xgb_proba = self.xgb_model.predict_proba(flat_features)
                    if xgb_proba.shape[1] > 1:
                        xgb_score = xgb_proba[:, 1]
                    else:
                        xgb_score = xgb_proba[:, 0]
                except:
                    xgb_score = np.full(batch_size, 0.5)
            else:
                xgb_score = np.full(batch_size, 0.5)
            
            if batch_size == 1:
                xgb_score = float(xgb_score[0]) if isinstance(xgb_score, np.ndarray) else float(xgb_score)
            
            if batch_size == 1:
                scores = np.array([transformer_score, xgb_score])
                ensemble_variance = np.var(scores)
                
                meta_input = np.array([[transformer_score, xgb_score, ensemble_variance, uncertainty_score]])
            else:
                transformer_scores = transformer_score if isinstance(transformer_score, np.ndarray) else np.full(batch_size, transformer_score)
                xgb_scores = xgb_score if isinstance(xgb_score, np.ndarray) else np.full(batch_size, xgb_score)
                uncertainty_scores = uncertainty_score if isinstance(uncertainty_score, np.ndarray) else np.full(batch_size, uncertainty_score)
                
                ensemble_variances = np.var(np.column_stack([transformer_scores, xgb_scores]), axis=1)
                meta_input = np.column_stack([transformer_scores, xgb_scores, ensemble_variances, uncertainty_scores])
            
            if hasattr(self.meta_learner, 'predict'):
                try:
                    ensemble_score = self.meta_learner.predict(meta_input)
                except:
                    ensemble_score = (transformer_score * self.weights[0] + xgb_score * self.weights[1])
            else:
                if batch_size == 1:
                    ensemble_score = transformer_score * self.weights[0] + xgb_score * self.weights[1]
                else:
                    ensemble_score = transformer_scores * self.weights[0] + xgb_scores * self.weights[1]
            
            if batch_size == 1:
                ensemble_score = float(ensemble_score[0]) if isinstance(ensemble_score, np.ndarray) else float(ensemble_score)
                
                entropy = self.calculate_prediction_entropy([transformer_score, xgb_score])
                self.entropy_gauge.set(entropy)
                self.uncertainty_gauge.set(uncertainty_score)
                
                self.momentum_scores.append(ensemble_score)
                self.training_data.append(feature_array[0])
                self.training_labels.append(1 if ensemble_score > self.dynamic_threshold else 0)
                self.entropy_scores.append(entropy)
                self.prediction_history.append({
                    'timestamp': time.time(),
                    'transformer_score': transformer_score,
                    'xgb_score': xgb_score,
                    'ensemble_score': ensemble_score,
                    'uncertainty': uncertainty_score,
                    'entropy': entropy
                })
                
                return ensemble_score
            else:
                return ensemble_score
                
        except Exception as e:
            logging.error(json.dumps({
                'event': 'prediction_error',
                'error': str(e)
            }))
            return 0.0 if batch_size == 1 else np.zeros(batch_size)

    def calculate_prediction_entropy(self, scores):
        try:
            probs = np.array(scores)
            probs = probs / np.sum(probs)
            probs = np.clip(probs, 1e-10, 1.0)
            entropy = -np.sum(probs * np.log2(probs))
            return entropy
        except:
            return 1.0

    def rebalance_thresholds(self):
        if len(self.momentum_scores) > 200:
            recent_scores = self.momentum_scores[-200:]
            
            score_volatility = np.std(recent_scores)
            score_mean = np.mean(recent_scores)
            
            if hasattr(self, 'prediction_history') and len(self.prediction_history) > 50:
                recent_uncertainty = np.mean([p['uncertainty'] for p in self.prediction_history[-50:]])
                uncertainty_factor = min(recent_uncertainty * 2, 0.2)
            else:
                uncertainty_factor = 0.1
            
            percentile_base = 85 if score_volatility > 0.2 else 90
            new_threshold = np.percentile(recent_scores, percentile_base)
            
            new_threshold = max(0.5, min(0.95, new_threshold + uncertainty_factor))
            
            if abs(new_threshold - self.dynamic_threshold) > 0.02:
                self.dynamic_threshold = new_threshold
                self.threshold_gauge.set(self.dynamic_threshold)
                
                logging.info(json.dumps({
                    'event': 'threshold_rebalanced',
                    'new_threshold': self.dynamic_threshold,
                    'score_volatility': score_volatility,
                    'uncertainty_factor': uncertainty_factor
                }))

    async def retrain_if_needed(self):
        retrain_threshold = self.settings['ml']['retrain_threshold']
        
        if len(self.training_data) < retrain_threshold:
            return
            
        try:
            with self.training_time.time():
                X = np.array(self.training_data[-retrain_threshold:])
                y = np.array(self.training_labels[-retrain_threshold:])
                
                if len(np.unique(y)) < 2:
                    logging.warning("Insufficient class diversity for retraining")
                    return
                
                await self.retrain_transformer(X, y)
                await self.retrain_xgboost(X, y)
                await self.retrain_meta_learner(X, y)
                
                self.calculate_feature_importance(X, y)
                
                performance_metrics = self.evaluate_ensemble_performance(X, y)
                self.loss_gauge.set(performance_metrics['loss'])
                
                self.save_tflite_model()
                
                self.training_data = self.training_data[-retrain_threshold//2:]
                self.training_labels = self.training_labels[-retrain_threshold//2:]
                self.entropy_scores = self.entropy_scores[-retrain_threshold//2:]
                self.prediction_history = self.prediction_history[-1000:]
                
                logging.info(json.dumps({
                    'event': 'model_retrained',
                    'performance_metrics': performance_metrics,
                    'data_size': len(X)
                }))
                
        except Exception as e:
            logging.error(json.dumps({
                'event': 'retrain_error',
                'error': str(e)
            }))

    async def retrain_transformer(self, X, y):
        try:
            if not OPTUNA_AVAILABLE:
                # Simple grid search if optuna not available
                best_lr = 0.0001
                best_loss = float('inf')
                
                for lr in [0.0001, 0.0005, 0.001]:
                    optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=lr, weight_decay=0.01)
                    
                    X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
                    y_torch = torch.tensor(y, dtype=torch.float32).to(self.device)
                    
                    if X_torch.ndim == 2:
                        X_torch = X_torch.unsqueeze(1)
                    
                    self.transformer.train()
                    total_loss = 0
                    
                    for epoch in range(5):
                        optimizer.zero_grad()
                        predictions, uncertainty = self.transformer(X_torch)
                        
                        pred_loss = self.transformer_criterion(predictions.squeeze(), y_torch)
                        uncertainty_target = torch.abs(predictions.squeeze() - y_torch).detach()
                        uncertainty_loss = self.uncertainty_criterion(uncertainty.squeeze(), uncertainty_target)
                        
                        loss = pred_loss + 0.1 * uncertainty_loss
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / 5
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_lr = lr
                
                self.transformer_optimizer = torch.optim.AdamW(
                    self.transformer.parameters(), 
                    lr=best_lr, 
                    weight_decay=0.01
                )
            else:
                # Use optuna for hyperparameter optimization
                def objective(trial):
                    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
                    dropout = trial.suggest_float('dropout', 0.1, 0.5)
                    
                    optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=lr, weight_decay=0.01)
                    
                    X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
                    y_torch = torch.tensor(y, dtype=torch.float32).to(self.device)
                    
                    if X_torch.ndim == 2:
                        X_torch = X_torch.unsqueeze(1)
                    
                    self.transformer.train()
                    total_loss = 0
                    
                    for epoch in range(10):
                        optimizer.zero_grad()
                        predictions, uncertainty = self.transformer(X_torch)
                        
                        pred_loss = self.transformer_criterion(predictions.squeeze(), y_torch)
                        
                        uncertainty_target = torch.abs(predictions.squeeze() - y_torch).detach()
                        uncertainty_loss = self.uncertainty_criterion(uncertainty.squeeze(), uncertainty_target)
                        
                        loss = pred_loss + 0.1 * uncertainty_loss
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        total_loss += loss.item()
                    
                    return total_loss / 10
                
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=15)
                
                best_params = study.best_params
                self.transformer_optimizer = torch.optim.AdamW(
                    self.transformer.parameters(), 
                    lr=best_params['lr'], 
                    weight_decay=0.01
                )
            
            # Final training with best parameters
            X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_torch = torch.tensor(y, dtype=torch.float32).to(self.device)
            
            if X_torch.ndim == 2:
                X_torch = X_torch.unsqueeze(1)
            
            self.transformer.train()
            for epoch in range(20):
                self.transformer_optimizer.zero_grad()
                predictions, uncertainty = self.transformer(X_torch)
                
                pred_loss = self.transformer_criterion(predictions.squeeze(), y_torch)
                uncertainty_target = torch.abs(predictions.squeeze() - y_torch).detach()
                uncertainty_loss = self.uncertainty_criterion(uncertainty.squeeze(), uncertainty_target)
                
                loss = pred_loss + 0.1 * uncertainty_loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
                self.transformer_optimizer.step()
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'transformer_retrain_error',
                'error': str(e)
            }))

    async def retrain_xgboost(self, X, y):
        try:
            X_flat = X.reshape(X.shape[0], -1)
            
            param_space = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            best_score = -1
            best_params = {}
            
            for n_est in param_space['n_estimators']:
                for depth in param_space['max_depth']:
                    for lr in param_space['learning_rate']:
                        for subsample in param_space['subsample']:
                            model = xgb.XGBClassifier(
                                n_estimators=n_est,
                                max_depth=depth,
                                learning_rate=lr,
                                subsample=subsample,
                                random_state=42
                            )
                            
                            try:
                                scores = cross_val_score(model, X_flat, y, cv=3, scoring='accuracy')
                                avg_score = np.mean(scores)
                                
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_params = {
                                        'n_estimators': n_est,
                                        'max_depth': depth,
                                        'learning_rate': lr,
                                        'subsample': subsample
                                    }
                            except:
                                continue
            
            if best_params:
                self.xgb_model = xgb.XGBClassifier(**best_params, random_state=42)
            
            self.xgb_model.fit(X_flat, y)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'xgboost_retrain_error',
                'error': str(e)
            }))

    async def retrain_meta_learner(self, X, y):
        try:
            X_torch = torch.tensor(X, dtype=torch.float32).to(self.device)
            if X_torch.ndim == 2:
                X_torch = X_torch.unsqueeze(1)
            
            with torch.no_grad():
                self.transformer.eval()
                transformer_preds, uncertainties = self.transformer(X_torch)
                transformer_scores = transformer_preds.squeeze().cpu().numpy()
                uncertainty_scores = uncertainties.squeeze().cpu().numpy()
            
            X_flat = X.reshape(X.shape[0], -1)
            xgb_scores = self.xgb_model.predict_proba(X_flat)[:, 1]
            
            ensemble_variances = np.var(np.column_stack([transformer_scores, xgb_scores]), axis=1)
            
            meta_X = np.column_stack([transformer_scores, xgb_scores, ensemble_variances, uncertainty_scores])
            
            self.meta_learner.fit(meta_X, y)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'meta_learner_retrain_error',
                'error': str(e)
            }))

    def calculate_feature_importance(self, X, y):
        try:
            if hasattr(self.xgb_model, 'feature_importances_'):
                importances = self.xgb_model.feature_importances_
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                self.feature_importance = dict(zip(feature_names, importances))
                
                top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                logging.info(json.dumps({
                    'event': 'feature_importance_calculated',
                    'top_features': top_features
                }))
        except Exception as e:
            logging.error(json.dumps({
                'event': 'feature_importance_error',
                'error': str(e)
            }))

    def evaluate_ensemble_performance(self, X, y):
        try:
            predictions = []
            for i in range(len(X)):
                pred = self.predict(X[i])
                predictions.append(1 if pred > self.dynamic_threshold else 0)
            
            accuracy = np.mean(np.array(predictions) == y)
            precision = np.sum((np.array(predictions) == 1) & (y == 1)) / max(np.sum(predictions), 1)
            recall = np.sum((np.array(predictions) == 1) & (y == 1)) / max(np.sum(y), 1)
            
            loss = np.mean((np.array([self.predict(X[i]) for i in range(len(X))]) - y) ** 2)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'loss': loss
            }
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'performance_evaluation_error',
                'error': str(e)
            }))
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'loss': 1}

    def save_tflite_model(self):
        try:
            dummy_input = torch.randn(1, 1, 11).to(self.device)
            
            self.transformer.eval()
            with torch.no_grad():
                traced_model = torch.jit.trace(self.transformer, dummy_input)
            
            traced_model.save('transformer_model.pt')
            
            logging.info(json.dumps({'event': 'model_saved_successfully'}))
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'model_save_error',
                'error': str(e)
            }))

    def get_model_summary(self):
        try:
            return {
                'dynamic_threshold': self.dynamic_threshold,
                'ensemble_weights': self.weights,
                'total_predictions': len(self.momentum_scores),
                'recent_entropy': np.mean(self.entropy_scores[-50:]) if len(self.entropy_scores) >= 50 else 0,
                'feature_importance': self.feature_importance,
                'model_device': str(self.device),
                'training_data_size': len(self.training_data)
            }
        except Exception as e:
            logging.error(json.dumps({
                'event': 'model_summary_error',
                'error': str(e)
            }))
            return {}