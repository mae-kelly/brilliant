import tensorflow as tf
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
import numpy as np
import os

def create_and_convert_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(45,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    X_dummy = np.random.random((1000, 45))
    y_dummy = np.random.randint(0, 2, (1000, 1))
    model.fit(X_dummy, y_dummy, epochs=5, verbose=0)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('models/model_weights.tflite', 'wb') as f:
        f.write(tflite_model)
    
    import joblib
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    import json
    feature_names = [f'feature_{i}' for i in range(45)]
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    print("✅ Model created and converted")

if __name__ == "__main__":
    create_and_convert_model()
