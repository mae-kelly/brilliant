import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import time

def generate_training_data(n_samples=50000):
    np.random.seed(42)
    features = np.random.randn(n_samples, 45)
    momentum_signal = features[:, 0] * 0.3 + features[:, 1] * 0.2 + features[:, 2] * 0.15
    volatility_signal = np.abs(features[:, 3]) * 0.2 + np.abs(features[:, 4]) * 0.1
    volume_signal = features[:, 5] * 0.1 + features[:, 6] * 0.05
    
    combined_signal = momentum_signal + volatility_signal + volume_signal
    noise = np.random.randn(n_samples) * 0.1
    target_continuous = combined_signal + noise
    
    threshold = np.percentile(target_continuous, 85)
    targets = (target_continuous > threshold).astype(int)
    
    return features, targets

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(45,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
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

def train_and_convert():
    X, y = generate_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = create_model()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=256,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open('models/model_weights.tflite', 'wb') as f:
        f.write(tflite_model)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    
    feature_names = [f"feature_{i}" for i in range(45)]
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    model_info = {
        'created_at': time.time(),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'model_size_bytes': len(tflite_model),
        'feature_count': 45
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("Model training and conversion completed successfully!")

if __name__ == "__main__":
    train_and_convert()
