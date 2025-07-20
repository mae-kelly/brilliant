import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
except ImportError:
    def get_dynamic_config(): return {"volatility_threshold": 0.1, "confidence_threshold": 0.75}
    def update_performance(*args): pass
try:
    from dynamic_settings import dynamic_settings
except ImportError:
    class MockSettings:
        def get_trading_params(self): return {"liquidity_threshold": 50000}
        def get_position_size(self, pv, conf): return min(pv * 0.1, 1.0)
    dynamic_settings = MockSettings()
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os

os.makedirs('models', exist_ok=True)

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

def generate_features(n_samples=100000):
    data = []
    for i in range(n_samples):
        features = {
            'price_delta': np.random.uniform(-0.5, 0.5),
            'liquidity_delta': np.random.uniform(-0.3, 0.8),
            'volume_delta': np.random.uniform(-0.2, 1.5),
            'volatility': np.random.uniform(0.01, 0.4),
            'velocity': np.random.uniform(-0.1, 0.2),
            'age_seconds': np.random.uniform(10, 3600),
            'dex_id': np.random.randint(0, 10),
            'base_volatility': np.random.uniform(0.005, 0.1),
            'base_velocity': np.random.uniform(0.001, 0.05),
            'momentum_score': np.random.uniform(0, 1),
            'order_flow_imbalance': np.random.uniform(-1, 1),
            'microstructure_noise': np.random.uniform(0, 0.1),
            'jump_intensity': np.random.uniform(0, 0.5),
            'volume_profile_anomaly': np.random.uniform(0, 2),
            'liquidity_fragmentation': np.random.uniform(0, 1)
        }
        
        momentum = features['momentum_score']
        volatility = features['volatility']
        volume_spike = features['volume_delta']
        
        breakout_prob = (momentum * 0.4 + 
                        volume_spike * 0.3 + 
                        (1 - volatility) * 0.2 + 
                        features['order_flow_imbalance'] * 0.1)
        
        breakout_prob = max(0, min(1, breakout_prob))
        
        if np.random.random() < 0.1:
            breakout_prob = 1 - breakout_prob
        
        features['target'] = 1 if breakout_prob > 0.7 else 0
        data.append(features)
    
    return pd.DataFrame(data)

print("Generating training data...")
df = generate_features()

feature_cols = [col for col in df.columns if col != 'target']
X = df[feature_cols].values
y = df['target'].values

print(f"Dataset shape: {X.shape}")
print(f"Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

print("Training model...")
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=256,
                    verbose=1)

test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")
print(f"Test precision: {test_prec:.3f}")
print(f"Test recall: {test_rec:.3f}")

model.save('models/breakout_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/model_weights.tflite', 'wb') as f:
    f.write(tflite_model)

joblib.dump(scaler, 'models/scaler.pkl')

with open('models/feature_names.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

print("Model saved:")
print("- models/model_weights.tflite")
print("- models/scaler.pkl") 
print("- models/feature_names.json")
