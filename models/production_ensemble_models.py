import numpy as np
import pandas as pd
import asyncio
import pickle
import joblib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

@dataclass
class ModelPrediction:
    model_name: str
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    model_version: str

@dataclass
class EnsemblePrediction:
    final_prediction: float
    confidence: float
    individual_predictions: List[ModelPrediction]
    ensemble_weights: Dict[str, float]
    model_agreement: float

class TransformerModel:
    def __init__(self, sequence_length: int = 120, features: int = 45):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        inputs = Input(shape=(self.sequence_length, self.features))
        
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm2 = LSTM(64, return_sequences=True, dropout=0.2)(lstm1)
        
        attention = Attention()([lstm2, lstm2])
        
        dense1 = Dense(64, activation='relu')(attention)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        
        outputs = Dense(1, activation='sigmoid')(dropout2)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('models/transformer_best.h5', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_scaled, y,
            validation_split=validation_split,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = self.model.predict(X_scaled)
        
        confidence = np.abs(predictions - 0.5) * 2
        
        return predictions.flatten(), confidence.flatten()

class LSTMModel:
    def __init__(self, sequence_length: int = 60, features: int = 10):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        self.model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, self.features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        history = self.model.fit(
            X_scaled, y,
            validation_split=validation_split,
            epochs=50,
            batch_size=64,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = self.model.predict(X_scaled)
        confidence = np.abs(predictions - 0.5) * 2
        
        return predictions.flatten(), confidence.flatten()

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(
            X_scaled, y,
            eval_set=[(X_scaled, y)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        confidence = np.abs(predictions - 0.5) * 2
        
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        return predictions, confidence, feature_importance

class GradientBoostingModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.15,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def train(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict_proba(X_scaled)[:, 1]
        confidence = np.abs(predictions - 0.5) * 2
        
        return predictions, confidence

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        
        return x

class GNNModel:
    def __init__(self, input_dim: int):
        self.model = GraphNeuralNetwork(input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
    def create_graph_data(self, features: np.ndarray, correlation_matrix: np.ndarray) -> List[Data]:
        threshold = 0.7
        edge_indices = np.where(np.abs(correlation_matrix) > threshold)
        edge_index = torch.tensor(np.stack(edge_indices), dtype=torch.long)
        
        graph_data = []
        for i in range(len(features)):
            x = torch.tensor(features[i:i+1], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            graph_data.append(data)
        
        return graph_data
    
    def train(self, graph_data: List[Data], labels: np.ndarray, epochs: int = 100):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for data, label in zip(graph_data, labels):
                self.optimizer.zero_grad()
                
                batch = torch.zeros(data.x.size(0), dtype=torch.long)
                out = self.model(data.x, data.edge_index, batch)
                
                loss = self.criterion(out.squeeze(), torch.tensor([label], dtype=torch.float))
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss/len(graph_data):.4f}')
    
    def predict(self, graph_data: List[Data]) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data in graph_data:
                batch = torch.zeros(data.x.size(0), dtype=torch.long)
                out = self.model(data.x, data.edge_index, batch)
                predictions.append(out.item())
        
        predictions = np.array(predictions)
        confidence = np.abs(predictions - 0.5) * 2
        
        return predictions, confidence

class ProductionEnsemble:
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        self.logger = logging.getLogger(__name__)
        
        self.model_configs = {
            'transformer': {'weight': 0.3, 'min_confidence': 0.7},
            'xgboost': {'weight': 0.25, 'min_confidence': 0.6},
            'lstm': {'weight': 0.2, 'min_confidence': 0.6},
            'gradient_boosting': {'weight': 0.15, 'min_confidence': 0.5},
            'gnn': {'weight': 0.1, 'min_confidence': 0.5}
        }
        
    async def initialize_models(self):
        self.models['transformer'] = TransformerModel()
        self.models['xgboost'] = XGBoostModel()
        self.models['lstm'] = LSTMModel()
        self.models['gradient_boosting'] = GradientBoostingModel()
        self.models['gnn'] = GNNModel(input_dim=45)
        
        self.models['transformer'].build_model()
        self.models['lstm'].build_model()
        
        self.logger.info("All ensemble models initialized")
    
    async def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                           X_seq: np.ndarray, correlation_matrix: np.ndarray):
        
        self.models['xgboost'].train(X, y)
        self.logger.info("XGBoost training completed")
        
        self.models['gradient_boosting'].train(X, y)
        self.logger.info("Gradient Boosting training completed")
        
        if X_seq is not None:
            self.models['transformer'].train(X_seq, y)
            self.logger.info("Transformer training completed")
            
            self.models['lstm'].train(X_seq[:, :60, :10], y)
            self.logger.info("LSTM training completed")
        
        if correlation_matrix is not None:
            graph_data = self.models['gnn'].create_graph_data(X, correlation_matrix)
            self.models['gnn'].train(graph_data, y)
            self.logger.info("GNN training completed")
        
        await self.save_models()
    
    async def predict_ensemble(self, X: np.ndarray, X_seq: Optional[np.ndarray] = None,
                             correlation_matrix: Optional[np.ndarray] = None) -> EnsemblePrediction:
        
        individual_predictions = []
        
        if 'xgboost' in self.models:
            pred, conf, feat_imp = self.models['xgboost'].predict(X)
            individual_predictions.append(ModelPrediction(
                model_name='xgboost',
                prediction=pred[0],
                confidence=conf[0],
                feature_importance=feat_imp,
                model_version='1.0'
            ))
        
        if 'gradient_boosting' in self.models:
            pred, conf = self.models['gradient_boosting'].predict(X)
            individual_predictions.append(ModelPrediction(
                model_name='gradient_boosting',
                prediction=pred[0],
                confidence=conf[0],
                feature_importance={},
                model_version='1.0'
            ))
        
        if X_seq is not None:
            if 'transformer' in self.models:
                pred, conf = self.models['transformer'].predict(X_seq)
                individual_predictions.append(ModelPrediction(
                    model_name='transformer',
                    prediction=pred[0],
                    confidence=conf[0],
                    feature_importance={},
                    model_version='1.0'
                ))
            
            if 'lstm' in self.models:
                pred, conf = self.models['lstm'].predict(X_seq[:, :60, :10])
                individual_predictions.append(ModelPrediction(
                    model_name='lstm',
                    prediction=pred[0],
                    confidence=conf[0],
                    feature_importance={},
                    model_version='1.0'
                ))
        
        if correlation_matrix is not None and 'gnn' in self.models:
            graph_data = self.models['gnn'].create_graph_data(X, correlation_matrix)
            pred, conf = self.models['gnn'].predict(graph_data)
            individual_predictions.append(ModelPrediction(
                model_name='gnn',
                prediction=pred[0],
                confidence=conf[0],
                feature_importance={},
                model_version='1.0'
            ))
        
        ensemble_prediction = self.combine_predictions(individual_predictions)
        
        return ensemble_prediction
    
    def combine_predictions(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
        if not predictions:
            return EnsemblePrediction(0.5, 0.0, [], {}, 0.0)
        
        weighted_sum = 0.0
        total_weight = 0.0
        valid_predictions = []
        ensemble_weights = {}
        
        for pred in predictions:
            config = self.model_configs.get(pred.model_name, {'weight': 0.1, 'min_confidence': 0.5})
            
            if pred.confidence >= config['min_confidence']:
                weight = config['weight'] * pred.confidence
                weighted_sum += pred.prediction * weight
                total_weight += weight
                valid_predictions.append(pred)
                ensemble_weights[pred.model_name] = weight
        
        if total_weight == 0:
            final_prediction = np.mean([p.prediction for p in predictions])
            confidence = np.mean([p.confidence for p in predictions])
        else:
            final_prediction = weighted_sum / total_weight
            confidence = total_weight / len(predictions)
        
        model_agreement = self.calculate_agreement([p.prediction for p in valid_predictions])
        
        return EnsemblePrediction(
            final_prediction=final_prediction,
            confidence=confidence,
            individual_predictions=predictions,
            ensemble_weights=ensemble_weights,
            model_agreement=model_agreement
        )
    
    def calculate_agreement(self, predictions: List[float]) -> float:
        if len(predictions) < 2:
            return 1.0
        
        mean_pred = np.mean(predictions)
        variance = np.var(predictions)
        
        agreement = 1.0 / (1.0 + variance)
        return agreement
    
    async def update_model_weights(self, model_name: str, performance_score: float):
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append(performance_score)
        
        recent_performance = np.mean(self.performance_history[model_name][-10:])
        
        base_weight = self.model_configs[model_name]['weight']
        performance_multiplier = min(recent_performance * 2, 2.0)
        
        self.model_configs[model_name]['weight'] = base_weight * performance_multiplier
        
        total_weight = sum(config['weight'] for config in self.model_configs.values())
        for config in self.model_configs.values():
            config['weight'] /= total_weight
    
    async def save_models(self):
        os.makedirs('models/ensemble', exist_ok=True)
        
        if 'transformer' in self.models:
            self.models['transformer'].model.save('models/ensemble/transformer.h5')
            joblib.dump(self.models['transformer'].scaler, 'models/ensemble/transformer_scaler.pkl')
        
        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], 'models/ensemble/xgboost.pkl')
        
        if 'lstm' in self.models:
            self.models['lstm'].model.save('models/ensemble/lstm.h5')
            joblib.dump(self.models['lstm'].scaler, 'models/ensemble/lstm_scaler.pkl')
        
        if 'gradient_boosting' in self.models:
            joblib.dump(self.models['gradient_boosting'], 'models/ensemble/gradient_boosting.pkl')
        
        if 'gnn' in self.models:
            torch.save(self.models['gnn'].model.state_dict(), 'models/ensemble/gnn.pth')
        
        with open('models/ensemble/model_configs.pkl', 'wb') as f:
            pickle.dump(self.model_configs, f)
        
        self.logger.info("All models saved successfully")
    
    async def load_models(self):
        try:
            if os.path.exists('models/ensemble/transformer.h5'):
                self.models['transformer'] = TransformerModel()
                self.models['transformer'].model = tf.keras.models.load_model('models/ensemble/transformer.h5')
                self.models['transformer'].scaler = joblib.load('models/ensemble/transformer_scaler.pkl')
            
            if os.path.exists('models/ensemble/xgboost.pkl'):
                self.models['xgboost'] = joblib.load('models/ensemble/xgboost.pkl')
            
            if os.path.exists('models/ensemble/lstm.h5'):
                self.models['lstm'] = LSTMModel()
                self.models['lstm'].model = tf.keras.models.load_model('models/ensemble/lstm.h5')
                self.models['lstm'].scaler = joblib.load('models/ensemble/lstm_scaler.pkl')
            
            if os.path.exists('models/ensemble/gradient_boosting.pkl'):
                self.models['gradient_boosting'] = joblib.load('models/ensemble/gradient_boosting.pkl')
            
            if os.path.exists('models/ensemble/gnn.pth'):
                self.models['gnn'] = GNNModel(input_dim=45)
                self.models['gnn'].model.load_state_dict(torch.load('models/ensemble/gnn.pth'))
            
            if os.path.exists('models/ensemble/model_configs.pkl'):
                with open('models/ensemble/model_configs.pkl', 'rb') as f:
                    self.model_configs = pickle.load(f)
            
            self.logger.info("All models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            await self.initialize_models()

production_ensemble = ProductionEnsemble()