
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os

def generate_training_data():
    np.random.seed(42)
    n_samples = 50000
    features = []
    labels = []
    
    for i in range(n_samples):
        if np.random.random() < 0.25:
            velocity = np.random.uniform(0.08, 0.20)
            volume_surge = np.random.uniform(3.0, 15.0)
            momentum = np.random.uniform(0.75, 1.0)
            volatility = np.random.uniform(0.02, 0.06)
            breakout_strength = momentum * velocity * 2
            liquidity_delta = np.random.uniform(0.3, 1.5)
            age = np.random.uniform(10, 180)
            dex_concentration = np.random.uniform(0.4, 0.9)
            label = 1
        else:
            velocity = np.random.uniform(-0.05, 0.04)
            volume_surge = np.random.uniform(0.3, 2.5)
            momentum = np.random.uniform(0.0, 0.65)
            volatility = np.random.uniform(0.0, 0.25)
            breakout_strength = momentum * abs(velocity) * 0.5
            liquidity_delta = np.random.uniform(0.05, 0.8)
            age = np.random.uniform(5, 600)
            dex_concentration = np.random.uniform(0.1, 0.7)
            label = 0
        
        features.append([velocity, volume_surge, momentum, volatility, 
                        breakout_strength, liquidity_delta, age, dex_concentration])
        labels.append(label)
    
    return np.array(features, dtype=np.float32), np.array(labels)

def create_model():
    inputs = tf.keras.Input(shape=(8,))
    
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

X, y = generate_training_data()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split_idx = int(0.85 * len(X))
X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

model = create_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.3),
    tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

model.save('models/breakout_classifier.h5')
joblib.dump(scaler, 'models/scaler.pkl')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/latest_model.tflite', 'wb') as f:
    f.write(tflite_model)

final_acc = history.history['val_accuracy'][-1]
final_precision = history.history['val_precision'][-1]
final_recall = history.history['val_recall'][-1]

print(f"Training complete. Accuracy: {final_acc:.4f}, Precision: {final_precision:.4f}, Recall: {final_recall:.4f}")

with open('models/model_stats.txt', 'w') as f:
    f.write(f"accuracy={final_acc:.4f}\n")
    f.write(f"precision={final_precision:.4f}\n")
    f.write(f"recall={final_recall:.4f}\n")
    f.write(f"features=velocity,volume_surge,momentum,volatility,breakout_strength,liquidity_delta,age,dex_concentration\n")

