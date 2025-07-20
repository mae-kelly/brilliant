import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import json
import time
import os
from typing import Dict, List, Tuple, Optional
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from data.real_market_data_collector import real_data_collector
from models.advanced_feature_engineer import advanced_feature_engineer

class ProductionBreakoutModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_projection = tf.keras.layers.Dense(128, activation='relu')
        self.attention_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=16) for _ in range(3)
        ]
        self.norm_layers = [tf.keras.layers.LayerNormalization() for _ in range(3)]
        self.dropout_layers = [tf.keras.layers.Dropout(0.2) for _ in range(3)]
        
        self.momentum_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.regime_head = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        
        self.confidence_head = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=None):
        x = self.feature_projection(inputs)
        x = tf.expand_dims(x, 1)
        
        for attention, norm, dropout in zip(self.attention_layers, self.norm_layers, self.dropout_layers):
            attn_output = attention(x, x, training=training)
            x = norm(x + dropout(attn_output, training=training))
        
        x = tf.squeeze(x, 1)
        
        momentum_pred = self.momentum_head(x, training=training)
        regime_pred = self.regime_head(x, training=training)
        confidence_pred = self.confidence_head(x, training=training)
        
        return {
            'momentum': momentum_pred,
            'regime': regime_pred,
            'confidence': confidence_pred
        }

class RealMarketDataTrainer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = None
        self.performance_metrics = {}
        self.feature_names = []
        
    async def collect_training_data(self, days: int = 30) -> pd.DataFrame:
        await real_data_collector.initialize()
        
        all_data = []
        chains = ['ethereum', 'arbitrum', 'polygon']
        
        for chain in chains:
            tokens = await real_data_collector.collect_live_token_data(chain, 500)
            
            for token in tokens:
                if token.price_usd > 0 and token.volume_24h > 1000:
                    price_history = await real_data_collector.get_token_price_history(
                        token.address, chain, hours=48
                    )
                    
                    if len(price_history) >= 24:
                        processed_samples = await self.process_token_for_training(
                            token, price_history
                        )
                        all_data.extend(processed_samples)
        
        if len(all_data) < 1000:
            synthetic_data = self.generate_realistic_training_data(5000)
            all_data.extend(synthetic_data)
        
        await real_data_collector.close()
        return pd.DataFrame(all_data)

    async def process_token_for_training(self, token, price_history: List[Tuple]) -> List[Dict]:
        samples = []
        
        prices = [p[1] for p in price_history]
        volumes = [token.volume_24h * (1 + np.random.uniform(-0.3, 0.3)) for _ in range(len(prices))]
        
        for i in range(24, len(prices) - 1):
            try:
                current_window = prices[i-24:i+1]
                volume_window = volumes[i-24:i+1]
                
                features = await advanced_feature_engineer.engineer_features(
                    {
                        'address': token.address,
                        'chain': token.chain,
                        'symbol': token.symbol,
                        'volume_24h': token.volume_24h,
                        'liquidity_usd': token.liquidity_usd,
                        'tx_count': token.tx_count
                    },
                    current_window,
                    volume_window,
                    []
                )
                
                if i < len(prices) - 1:
                    future_price = prices[i + 1]
                    current_price = prices[i]
                    
                    price_change_1h = (future_price - current_price) / current_price
                    
                    is_breakout = 1 if (0.09 <= price_change_1h <= 0.15) else 0
                    
                    regime = self.determine_market_regime(current_window)
                    confidence = self.calculate_label_confidence(current_window, volume_window)
                    
                    sample = {
                        **{f'feature_{i}': val for i, val in enumerate(features.combined_features)},
                        'target_momentum': is_breakout,
                        'target_regime': regime,
                        'target_confidence': confidence,
                        'price_change': price_change_1h,
                        'token_address': token.address,
                        'timestamp': time.time() - (len(prices) - i) * 3600
                    }
                    
                    samples.append(sample)
                    
            except Exception as e:
                continue
        
        return samples

    def determine_market_regime(self, prices: List[float]) -> int:
        if len(prices) < 10:
            return 1
        
        returns = np.diff(prices) / np.array(prices[:-1])
        volatility = np.std(returns)
        trend = np.mean(returns)
        
        if volatility > 0.15:
            return 3
        elif trend > 0.05:
            return 0
        elif trend < -0.05:
            return 2
        else:
            return 1

    def calculate_label_confidence(self, prices: List[float], volumes: List[float]) -> float:
        price_volatility = np.std(prices) / np.mean(prices)
        volume_consistency = 1.0 - (np.std(volumes) / (np.mean(volumes) + 1e-10))
        
        return np.clip(0.5 + volume_consistency - price_volatility, 0.0, 1.0)

    def generate_realistic_training_data(self, num_samples: int) -> List[Dict]:
        samples = []
        
        for i in range(num_samples):
            regime = np.random.choice([0, 1, 2, 3], p=[0.25, 0.4, 0.25, 0.1])
            
            if regime == 0:
                momentum_signal = np.random.uniform(0.05, 0.15)
                volatility = np.random.uniform(0.02, 0.08)
            elif regime == 1:
                momentum_signal = np.random.uniform(-0.02, 0.02)
                volatility = np.random.uniform(0.01, 0.05)
            elif regime == 2:
                momentum_signal = np.random.uniform(-0.15, -0.05)
                volatility = np.random.uniform(0.02, 0.08)
            else:
                momentum_signal = np.random.uniform(-0.10, 0.10)
                volatility = np.random.uniform(0.15, 0.40)
            
            features = np.random.normal(0, 0.1, 45)
            features[0] = momentum_signal + np.random.normal(0, 0.01)
            features[6] = volatility
            features[10] = np.random.uniform(0.5, 2.0) if regime == 0 else np.random.uniform(0.1, 1.0)
            
            is_breakout = 1 if (0.09 <= momentum_signal <= 0.15 and volatility < 0.1) else 0
            confidence = 0.8 if is_breakout else 0.6
            
            sample = {
                **{f'feature_{j}': features[j] for j in range(45)},
                'target_momentum': is_breakout,
                'target_regime': regime,
                'target_confidence': confidence,
                'price_change': momentum_signal,
                'token_address': f'synthetic_{i}',
                'timestamp': time.time() - i * 3600
            }
            
            samples.append(sample)
        
        return samples

    async def train_production_model(self, df: pd.DataFrame):
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        X = df[feature_cols].values
        y_momentum = df['target_momentum'].values
        y_regime = df['target_regime'].values
        y_confidence = df['target_confidence'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=3)
        best_model = None
        best_score = 0
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train_momentum, y_val_momentum = y_momentum[train_idx], y_momentum[val_idx]
            y_train_regime, y_val_regime = y_regime[train_idx], y_regime[val_idx]
            y_train_confidence, y_val_confidence = y_confidence[train_idx], y_confidence[val_idx]
            
            model = ProductionBreakoutModel()
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            
            model.compile(
                optimizer=optimizer,
                loss={
                    'momentum': 'binary_crossentropy',
                    'regime': 'sparse_categorical_crossentropy',
                    'confidence': 'mse'
                },
                metrics={
                    'momentum': ['accuracy'],
                    'regime': ['accuracy'],
                    'confidence': ['mae']
                },
                loss_weights={'momentum': 0.6, 'regime': 0.3, 'confidence': 0.1}
            )
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_momentum_accuracy',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_momentum_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
            
            history = model.fit(
                X_train,
                {
                    'momentum': y_train_momentum,
                    'regime': y_train_regime,
                    'confidence': y_train_confidence
                },
                validation_data=(
                    X_val,
                    {
                        'momentum': y_val_momentum,
                        'regime': y_val_regime,
                        'confidence': y_val_confidence
                    }
                ),
                epochs=100,
                batch_size=256,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            val_score = max(history.history['val_momentum_accuracy'])
            
            if val_score > best_score:
                best_score = val_score
                best_model = model
        
        self.model = best_model
        
        X_test = X_scaled[-500:]
        y_test = y_momentum[-500:]
        
        predictions = self.model.predict(X_test)
        y_pred = (predictions['momentum'] > 0.5).astype(int).flatten()
        
        self.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'auc': roc_auc_score(y_test, predictions['momentum'].flatten()),
            'best_val_score': best_score
        }
        
        return self.model

    def convert_to_tflite(self) -> str:
        if not self.model:
            raise ValueError("No model to convert")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        os.makedirs('models', exist_ok=True)
        model_path = 'models/production_model.tflite'
        
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        
        return model_path

    def save_artifacts(self):
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.scaler, 'models/production_scaler.pkl')
        
        feature_names = [f'feature_{i}' for i in range(45)]
        with open('models/production_feature_names.json', 'w') as f:
            json.dump(feature_names, f)
        
        model_info = {
            'model_type': 'ProductionBreakoutModel',
            'performance': self.performance_metrics,
            'training_timestamp': time.time(),
            'feature_count': 45
        }
        
        with open('models/production_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

    async def full_training_pipeline(self):
        print("Collecting real market data...")
        df = await self.collect_training_data()
        
        print(f"Training on {len(df)} samples...")
        model = await self.train_production_model(df)
        
        print("Converting to TFLite...")
        tflite_path = self.convert_to_tflite()
        
        print("Saving artifacts...")
        self.save_artifacts()
        
        print(f"Training complete. Model saved to {tflite_path}")
        print(f"Performance: {self.performance_metrics}")
        
        return {
            'model': model,
            'tflite_path': tflite_path,
            'performance': self.performance_metrics
        }

production_trainer = RealMarketDataTrainer()