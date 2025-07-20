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
import tf.keras as keras
from tf.keras import layers
import pickle
import json

np.random.seed(42)
X = np.random.randn(10000, 45)
y = (X[:, 0] + X[:, 1] * 0.3 + X[:, 2] * 0.2 + np.random.randn(10000) * 0.1 > 0.5).astype(int)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(45,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

converter = tf.lite.TFLiteConverter.from_keras_model
tflite_model = converter(model)

with open('models/model_weights.tflite', 'wb') as f:
    f.write(tflite_model)

scaler_data = {'mean_': np.mean(X, axis=0), 'scale_': np.std(X, axis=0)}
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_data, f)

features = [f'feature_{i}' for i in range(45)]
with open('models/feature_names.json', 'w') as f:
    json.dump(features, f)
