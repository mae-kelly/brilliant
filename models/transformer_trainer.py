import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from datetime import datetime

class TransformerBreakoutModel:
    def __init__(self, sequence_length=60, d_model=256, num_heads=8, num_layers=6):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.scaler = StandardScaler()
        
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def build_transformer(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        
        # Input embedding
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(self.sequence_length, self.d_model)
        x += pos_encoding[:, :tf.shape(x)[1], :]
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.d_model//self.num_heads
            )(x, x)
            
            # Residual connection and layer norm
            x = tf.keras.layers.LayerNormalization()(x + attn_output)
            
            # Feed forward network
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(self.d_model * 4, activation='relu'),
                tf.keras.layers.Dense(self.d_model)
            ])
            
            ffn_output = ffn(x)
            x = tf.keras.layers.LayerNormalization()(x + ffn_output)
        
        # Global pooling and classification
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Multi-task outputs
        breakout_prob = tf.keras.layers.Dense(1, activation='sigmoid', name='breakout')(x)
        regime_class = tf.keras.layers.Dense(5, activation='softmax', name='regime')(x)
        volatility_pred = tf.keras.layers.Dense(1, activation='sigmoid', name='volatility')(x)
        momentum_pred = tf.keras.layers.Dense(1, activation='tanh', name='momentum')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs={
            'breakout': breakout_prob,
            'regime': regime_class,
            'volatility': volatility_pred,
            'momentum': momentum_pred
        })
        
        return model
    
    def compile_model(self, model):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={
                'breakout': 'binary_crossentropy',
                'regime': 'categorical_crossentropy',
                'volatility': 'mse',
                'momentum': 'mse'
            },
            loss_weights={
                'breakout': 1.0,
                'regime': 0.5,
                'volatility': 0.3,
                'momentum': 0.2
            },
            metrics={
                'breakout': ['accuracy', 'precision', 'recall'],
                'regime': ['accuracy'],
                'volatility': ['mae'],
                'momentum': ['mae']
            }
        )
        return model

def train_production_model():
    print("🔥 Training Transformer Model...")
    
    # Generate enhanced training data
    np.random.seed(42)
    n_samples = 100000
    sequence_length = 60
    n_features = 25
    
    X = []
    y_breakout = []
    y_regime = []
    y_volatility = []
    y_momentum = []
    
    for i in range(n_samples):
        # Generate realistic sequences
        sequence = np.random.randn(sequence_length, n_features)
        
        # Add patterns for different regimes
        regime = np.random.choice(5)
        if regime == 0:  # Breakout pattern
            sequence[-10:, :5] *= 2.0  # Amplify price features
            label = 1
        else:
            label = 0
        
        # Add noise and trends
        for j in range(n_features):
            trend = np.linspace(0, np.random.normal(0, 0.1), sequence_length)
            sequence[:, j] += trend
        
        X.append(sequence)
        y_breakout.append(label)
        y_regime.append(tf.keras.utils.to_categorical(regime, 5))
        y_volatility.append(np.random.beta(2, 5))  # Skewed toward low volatility
        y_momentum.append(np.random.normal(0, 0.1))
    
    X = np.array(X)
    y_breakout = np.array(y_breakout)
    y_regime = np.array(y_regime)
    y_volatility = np.array(y_volatility)
    y_momentum = np.array(y_momentum)
    
    print(f"Training data shape: {X.shape}")
    
    # Create and train model
    trainer = TransformerBreakoutModel()
    model = trainer.build_transformer((sequence_length, n_features))
    model = trainer.compile_model(model)
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint('models/transformer_best.h5', save_best_only=True)
    ]
    
    # Train
    history = model.fit(
        X, {
            'breakout': y_breakout,
            'regime': y_regime,
            'volatility': y_volatility,
            'momentum': y_momentum
        },
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save scaler and model
    joblib.dump(trainer.scaler, 'models/transformer_scaler.pkl')
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('models/transformer_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # Save metadata
    metadata = {
        'model_type': 'transformer',
        'version': '2.0',
        'sequence_length': sequence_length,
        'n_features': n_features,
        'd_model': trainer.d_model,
        'num_heads': trainer.num_heads,
        'num_layers': trainer.num_layers,
        'trained_at': datetime.now().isoformat(),
        'training_samples': n_samples,
        'final_accuracy': float(max(history.history['val_breakout_accuracy'])),
        'feature_names': [f'feature_{i}' for i in range(n_features)]
    }
    
    with open('models/transformer_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Transformer model training complete!")
    return model, history

if __name__ == "__main__":
    train_production_model()
