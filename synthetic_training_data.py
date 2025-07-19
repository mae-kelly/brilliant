
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import numpy as np
import pandas as pd
import json
import time
import random
from typing import List, Dict, Tuple
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class SyntheticDataGenerator:
    def __init__(self):
        self.patterns = {
            'organic_breakout': {'price_multiplier': 1.15, 'volume_multiplier': 3.0, 'success_rate': 0.75},
            'whale_pump': {'price_multiplier': 1.25, 'volume_multiplier': 8.0, 'success_rate': 0.60},
            'bot_arbitrage': {'price_multiplier': 1.08, 'volume_multiplier': 1.5, 'success_rate': 0.85},
            'rug_pull': {'price_multiplier': 1.30, 'volume_multiplier': 12.0, 'success_rate': 0.05},
            'false_breakout': {'price_multiplier': 1.12, 'volume_multiplier': 2.0, 'success_rate': 0.25}
        }

    def generate_price_series(self, pattern: str, length: int = 60) -> List[float]:
        base_price = random.uniform(0.001, 10.0)
        prices = [base_price]
        
        config = self.patterns[pattern]
        target_multiplier = config['price_multiplier']
        
        if pattern == 'organic_breakout':
            for i in range(1, length):
                if i < 20:
                    change = random.gauss(0.001, 0.005)
                elif i < 40:
                    change = random.gauss(0.02, 0.01)
                else:
                    change = random.gauss(-0.005, 0.008)
                prices.append(prices[-1] * (1 + change))
                
        elif pattern == 'whale_pump':
            for i in range(1, length):
                if i < 30:
                    change = random.gauss(0.001, 0.003)
                elif i < 35:
                    change = random.gauss(0.05, 0.02)
                else:
                    change = random.gauss(-0.02, 0.015)
                prices.append(prices[-1] * (1 + change))
                
        elif pattern == 'rug_pull':
            for i in range(1, length):
                if i < 25:
                    change = random.gauss(0.003, 0.005)
                elif i < 30:
                    change = random.gauss(0.08, 0.03)
                else:
                    change = random.gauss(-0.15, 0.05)
                prices.append(max(0.001, prices[-1] * (1 + change)))
                
        else:
            for i in range(1, length):
                volatility = 0.01 if i < 30 else 0.02
                change = random.gauss(0.002, volatility)
                prices.append(prices[-1] * (1 + change))
                
        return prices

    def generate_volume_series(self, pattern: str, price_series: List[float]) -> List[float]:
        base_volume = random.uniform(1000, 100000)
        volumes = []
        
        config = self.patterns[pattern]
        multiplier = config['volume_multiplier']
        
        for i, price in enumerate(price_series):
            if i == 0:
                volumes.append(base_volume)
                continue
                
            price_change = abs(price - price_series[i-1]) / price_series[i-1]
            
            if pattern in ['whale_pump', 'rug_pull']:
                if 25 <= i <= 35:
                    volume = base_volume * multiplier * (1 + price_change * 10)
                else:
                    volume = base_volume * (1 + price_change * 2)
            else:
                volume = base_volume * (1 + price_change * 5) * random.uniform(0.8, 1.2)
                
            volumes.append(volume)
            
        return volumes

    def calculate_features(self, prices: List[float], volumes: List[float]) -> Dict:
        prices_arr = np.array(prices)
        volumes_arr = np.array(volumes)
        
        price_delta = (prices[-1] - prices[0]) / prices[0]
        volume_delta = (np.mean(volumes[-10:]) - np.mean(volumes[:10])) / np.mean(volumes[:10])
        liquidity_delta = random.uniform(-0.1, 0.3)
        
        volatility = np.std(np.diff(prices) / prices[:-1])
        velocity = price_delta / len(prices)
        momentum = np.sum(np.diff(prices[-10:]))
        
        age_seconds = random.randint(30, 1800)
        dex_id = random.randint(0, 4)
        base_volatility = random.uniform(0.01, 0.05)
        base_velocity = random.uniform(0.001, 0.01)
        
        return {
            'price_delta': price_delta,
            'volume_delta': volume_delta,
            'liquidity_delta': liquidity_delta,
            'volatility': volatility,
            'velocity': velocity,
            'momentum': momentum,
            'age_seconds': age_seconds,
            'dex_id': dex_id,
            'base_volatility': base_volatility,
            'base_velocity': base_velocity
        }

    def generate_dataset(self, num_samples: int = 10000) -> pd.DataFrame:
        data = []
        
        for _ in range(num_samples):
            pattern = random.choice(list(self.patterns.keys()))
            config = self.patterns[pattern]
            
            prices = self.generate_price_series(pattern)
            volumes = self.generate_volume_series(pattern, prices)
            features = self.calculate_features(prices, volumes)
            
            is_successful = random.random() < config['success_rate']
            roi = random.uniform(1.05, 1.20) if is_successful else random.uniform(0.85, 1.05)
            
            features.update({
                'pattern': pattern,
                'roi': roi,
                'successful': is_successful,
                'price_series': json.dumps(prices[-20:]),
                'volume_series': json.dumps(volumes[-20:])
            })
            
            data.append(features)
            
        return pd.DataFrame(data)

    def save_training_data(self, df: pd.DataFrame, path: str = 'data/training_data.csv'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Generated {len(df)} training samples -> {path}")

class ProductionModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [
            'price_delta', 'volume_delta', 'liquidity_delta',
            'volatility', 'velocity', 'momentum',
            'age_seconds', 'dex_id', 'base_volatility', 'base_velocity'
        ]

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df[self.feature_names].values
        y = (df['roi'] > 1.05).astype(int).values
        
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def build_model(self, input_dim: int) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model

    def train_model(self, X: np.ndarray, y: np.ndarray) -> tf.keras.Model:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = self.build_model(X.shape[1])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.model

    def convert_to_tflite(self, model_path: str = 'models/best_model.h5', 
                         output_path: str = 'models/model.tflite'):
        model = tf.keras.models.load_model(model_path)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Model converted to TFLite: {output_path}")

    def save_artifacts(self):
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        with open('models/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)

def main():
    generator = SyntheticDataGenerator()
    trainer = ProductionModelTrainer()
    
    print("Generating synthetic training data...")
    df = generator.generate_dataset(50000)
    generator.save_training_data(df)
    
    print("Training production model...")
    X, y = trainer.prepare_data(df)
    model = trainer.train_model(X, y)
    
    print("Converting to TFLite...")
    trainer.convert_to_tflite()
    trainer.save_artifacts()
    
    print("Training pipeline complete!")

if __name__ == "__main__":
    main()
