"""
PRODUCTION Renaissance Transformer - Complete implementation
Advanced transformer architecture for DeFi momentum prediction
"""
import tensorflow as tf
import numpy as np
import json
import os
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

@dataclass
class TransformerConfig:
    seq_length: int = 120
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dff: int = 2048
    input_features: int = 45
    dropout_rate: float = 0.1
    max_position_encoding: int = 10000

class MultiHeadAttention(tf.keras.layers.Layer):
    """Production multi-head attention layer"""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding for transformer"""
    
    def __init__(self, position: int, d_model: int):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Single transformer encoder layer"""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def point_wise_feed_forward_network(self, d_model: int, dff: int):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
    
    def call(self, x, training, mask):
        attn_output, attention_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2, attention_weights

class RenaissanceTransformer(tf.keras.Model):
    """Production Renaissance Transformer for DeFi momentum prediction"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        
        # Input projection layer
        self.input_projection = tf.keras.layers.Dense(config.d_model, activation='relu')
        self.input_dropout = tf.keras.layers.Dropout(config.dropout_rate)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.max_position_encoding, config.d_model)
        
        # Transformer encoder layers
        self.enc_layers = [
            TransformerEncoderLayer(config.d_model, config.num_heads, config.dff, config.dropout_rate)
            for _ in range(config.num_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        
        # Market regime detection branch
        self.regime_attention = tf.keras.layers.GlobalAveragePooling1D()
        self.regime_dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.regime_dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.regime_output = tf.keras.layers.Dense(4, activation='softmax', name='regime')  # 4 market regimes
        
        # Price direction prediction branch
        self.price_attention = tf.keras.layers.GlobalMaxPooling1D()
        self.price_dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.price_dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.price_dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.price_output = tf.keras.layers.Dense(1, activation='sigmoid', name='breakout')
        
        # Volatility prediction branch
        self.vol_dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.vol_dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.vol_output = tf.keras.layers.Dense(1, activation='relu', name='volatility')
        
        # Attention weights storage
        self.attention_weights = []
        
    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.input_dropout(x, training=training)
        
        # Store attention weights
        self.attention_weights = []
        
        # Pass through transformer layers
        for i in range(self.num_layers):
            x, attention_weights = self.enc_layers[i](x, training, mask)
            self.attention_weights.append(attention_weights)
        
        x = self.dropout(x, training=training)
        
        # Market regime detection
        regime_features = self.regime_attention(x)
        regime_features = self.regime_dense1(regime_features)
        regime_features = self.regime_dense2(regime_features)
        regime_prob = self.regime_output(regime_features)
        
        # Price breakout prediction
        price_features = self.price_attention(x)
        price_features = self.price_dense1(price_features)
        price_features = self.price_dense2(price_features)
        price_features = self.price_dense3(price_features)
        breakout_prob = self.price_output(price_features)
        
        # Volatility prediction
        vol_features = tf.concat([regime_features, price_features], axis=-1)
        vol_features = self.vol_dense1(vol_features)
        vol_features = self.vol_dense2(vol_features)
        volatility_pred = self.vol_output(vol_features)
        
        return {
            'breakout': breakout_prob,
            'regime': regime_prob,
            'volatility': volatility_pred
        }
    
    def get_attention_weights(self):
        """Get attention weights for interpretability"""
        return self.attention_weights

class TransformerTrainer:
    """Production trainer for Renaissance Transformer"""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.model = RenaissanceTransformer(config)
        self.logger = logging.getLogger(__name__)
        
        # Advanced optimizer with learning rate scheduling
        self.initial_learning_rate = 0.0001
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.lr_schedule,
            weight_decay=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Loss functions
        self.breakout_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
        self.regime_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        self.volatility_loss = tf.keras.losses.MeanSquaredError()
        
        # Metrics
        self.breakout_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.regime_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.volatility_mae = tf.keras.metrics.MeanAbsoluteError()
        
    def compile_model(self):
        """Compile the model with multiple outputs"""
        self.model.compile(
            optimizer=self.optimizer,
            loss={
                'breakout': self.breakout_loss,
                'regime': self.regime_loss,
                'volatility': self.volatility_loss
            },
            loss_weights={
                'breakout': 1.0,
                'regime': 0.5,
                'volatility': 0.3
            },
            metrics={
                'breakout': [self.breakout_accuracy, 'precision', 'recall'],
                'regime': [self.regime_accuracy],
                'volatility': [self.volatility_mae]
            }
        )
    
    def train(self, train_data, val_data, epochs: int = 100):
        """Train the transformer model"""
        self.compile_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_breakout_binary_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/renaissance_transformer_best.h5',
                monitor='val_breakout_binary_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/transformer_{int(time.time())}',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def convert_to_tflite(self, model_path: str = 'models/renaissance_transformer_best.h5'):
        """Convert model to TensorFlow Lite for production inference"""
        # Load the best model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to TFLite with optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Advanced optimization
        converter.representative_dataset = self._representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = 'models/renaissance_transformer.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        self.logger.info(f"✅ TFLite model saved: {tflite_path}")
        
        return tflite_path
    
    def _representative_dataset(self):
        """Representative dataset for quantization"""
        # Generate representative data
        for _ in range(100):
            sample = tf.random.normal([1, self.config.seq_length, self.config.input_features])
            yield [sample]

class ProductionInference:
    """Production inference engine for Renaissance Transformer"""
    
    def __init__(self, model_path: str = 'models/renaissance_transformer.tflite'):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load TFLite model for inference"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.logger.info(f"✅ Loaded TFLite model: {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """Production inference with complete feature processing"""
        try:
            # Ensure correct input shape
            if len(features.shape) == 2:
                features = features.reshape(1, features.shape[0], features.shape[1])
            elif len(features.shape) == 1:
                features = features.reshape(1, 1, features.shape[0])
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], features.astype(np.float32))
            
            # Run inference
            self.interpreter.invoke()
            
            # Get outputs
            outputs = {}
            for output_detail in self.output_details:
                output_name = output_detail['name']
                output_data = self.interpreter.get_tensor(output_detail['index'])
                outputs[output_name] = float(output_data[0][0]) if output_data.shape[-1] == 1 else output_data[0]
            
            # Calculate confidence metrics
            breakout_prob = outputs.get('breakout', 0.5)
            regime_probs = outputs.get('regime', np.array([0.25, 0.25, 0.25, 0.25]))
            volatility = outputs.get('volatility', 0.1)
            
            # Calculate entropy for confidence
            breakout_entropy = -(breakout_prob * np.log(breakout_prob + 1e-10) + 
                               (1 - breakout_prob) * np.log(1 - breakout_prob + 1e-10))
            
            regime_entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-10))
            
            # Overall confidence (lower entropy = higher confidence)
            confidence = 1.0 - (breakout_entropy + regime_entropy * 0.5) / 2.0
            
            return {
                'breakout_probability': float(breakout_prob),
                'regime_probabilities': regime_probs.tolist() if hasattr(regime_probs, 'tolist') else regime_probs,
                'predicted_volatility': float(volatility),
                'confidence': float(confidence),
                'breakout_entropy': float(breakout_entropy),
                'regime_entropy': float(regime_entropy)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Inference error: {e}")
            return {
                'breakout_probability': 0.5,
                'regime_probabilities': [0.25, 0.25, 0.25, 0.25],
                'predicted_volatility': 0.1,
                'confidence': 0.0,
                'breakout_entropy': 1.0,
                'regime_entropy': 1.386
            }

# Global instances
transformer_config = TransformerConfig()
transformer_trainer = TransformerTrainer(transformer_config)
production_inference = ProductionInference()
