"""
PRODUCTION Renaissance Transformer - Complete implementation
Advanced transformer architecture for DeFi momentum prediction
"""
import tensorflow as tf
import numpy as np
import json
import os
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

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
    vocab_size: int = 10000
    
class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, position: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        # Apply sin to even indices in the array
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention mechanism with scaled dot-product attention"""
    
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
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add the mask to the scaled tensor
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights

class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Single transformer encoder layer with feed-forward network"""
    
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
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])
    
    def call(self, x, training, mask=None):
        attn_output, attention_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2, attention_weights

class RegimeDetectionHead(tf.keras.layers.Layer):
    """Specialized head for market regime detection"""
    
    def __init__(self, num_regimes: int = 4):
        super().__init__()
        self.num_regimes = num_regimes
        
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(num_regimes, activation='softmax', name='regime')
    
    def call(self, x, training=None):
        x = self.global_pool(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

class MomentumPredictionHead(tf.keras.layers.Layer):
    """Specialized head for momentum breakout prediction"""
    
    def __init__(self):
        super().__init__()
        
        self.attention_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='breakout')
    
    def call(self, x, training=None):
        x = self.attention_pool(x)
        
        x = self.dense1(x)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        
        return self.output_layer(x)

class VolatilityPredictionHead(tf.keras.layers.Layer):
    """Specialized head for volatility prediction"""
    
    def __init__(self):
        super().__init__()
        
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(1, activation='relu', name='volatility')
    
    def call(self, x, training=None):
        # Use regime features as input
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

class RenaissanceTransformer(tf.keras.Model):
    """Production Renaissance Transformer for DeFi momentum prediction"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        
        # Input projection and embedding
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
        
        # Specialized prediction heads
        self.regime_head = RegimeDetectionHead(num_regimes=4)
        self.momentum_head = MomentumPredictionHead()
        self.volatility_head = VolatilityPredictionHead()
        
        # Store attention weights for interpretability
        self.attention_weights = []
        
    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]
        
        # Input projection and positional encoding
        x = self.input_projection(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.input_dropout(x, training=training)
        
        # Store attention weights for each layer
        self.attention_weights = []
        
        # Pass through transformer layers
        for i in range(self.num_layers):
            x, attention_weights = self.enc_layers[i](x, training, mask)
            self.attention_weights.append(attention_weights)
        
        x = self.dropout(x, training=training)
        
        # Market regime detection
        regime_prob = self.regime_head(x, training=training)
        
        # Momentum breakout prediction
        breakout_prob = self.momentum_head(x, training=training)
        
        # Volatility prediction (uses regime features)
        regime_features = self.regime_head.dense2(
            self.regime_head.dropout1(
                self.regime_head.dense1(
                    self.regime_head.global_pool(x)
                ), training=training
            )
        )
        volatility_pred = self.volatility_head(regime_features, training=training)
        
        return {
            'breakout': breakout_prob,
            'regime': regime_prob,
            'volatility': volatility_pred
        }
    
    def get_attention_weights(self):
        """Get attention weights for interpretability"""
        return self.attention_weights
    
    def get_config(self):
        return {
            'config': {
                'seq_length': self.config.seq_length,
                'd_model': self.config.d_model,
                'num_heads': self.config.num_heads,
                'num_layers': self.config.num_layers,
                'dff': self.config.dff,
                'input_features': self.config.input_features,
                'dropout_rate': self.config.dropout_rate
            }
        }

class TransformerTrainer:
    """Advanced trainer for Renaissance Transformer with learning rate scheduling"""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.model = RenaissanceTransformer(config)
        self.logger = logging.getLogger(__name__)
        
        # Advanced learning rate schedule
        self.initial_learning_rate = 0.0001
        self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=self.initial_learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.0
        )
        
        # Advanced optimizer with gradient clipping
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.lr_schedule,
            weight_decay=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=1.0
        )
        
        # Advanced loss functions with label smoothing
        self.breakout_loss = tf.keras.losses.BinaryCrossentropy(
            label_smoothing=0.1, 
            from_logits=False
        )
        self.regime_loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1, 
            from_logits=False
        )
        self.volatility_loss = tf.keras.losses.Huber(delta=0.1)
        
        # Comprehensive metrics
        self.metrics = {
            'breakout': [
                tf.keras.metrics.BinaryAccuracy(name='breakout_accuracy'),
                tf.keras.metrics.Precision(name='breakout_precision'),
                tf.keras.metrics.Recall(name='breakout_recall'),
                tf.keras.metrics.AUC(name='breakout_auc')
            ],
            'regime': [
                tf.keras.metrics.CategoricalAccuracy(name='regime_accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='regime_top2_accuracy')
            ],
            'volatility': [
                tf.keras.metrics.MeanAbsoluteError(name='volatility_mae'),
                tf.keras.metrics.MeanSquaredError(name='volatility_mse')
            ]
        }
        
    def compile_model(self):
        """Compile the model with multiple outputs and loss weighting"""
        self.model.compile(
            optimizer=self.optimizer,
            loss={
                'breakout': self.breakout_loss,
                'regime': self.regime_loss,
                'volatility': self.volatility_loss
            },
            loss_weights={
                'breakout': 1.0,      # Primary task
                'regime': 0.5,        # Secondary task
                'volatility': 0.3     # Auxiliary task
            },
            metrics=self.metrics
        )
    
    def create_callbacks(self, patience_early_stop=15, patience_reduce_lr=7):
        """Create advanced training callbacks"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_breakout_auc',
                patience=patience_early_stop,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience_reduce_lr,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/renaissance_transformer_best.h5',
                monitor='val_breakout_auc',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/transformer_{int(time.time())}',
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                profile_batch=(10, 20)
            ),
            tf.keras.callbacks.CSVLogger(
                f'logs/training_log_{int(time.time())}.csv'
            )
        ]
    
    def train(self, train_data, val_data, epochs: int = 100):
        """Train the transformer model with advanced techniques"""
        self.compile_model()
        callbacks = self.create_callbacks()
        
        self.logger.info(f"🚀 Starting Renaissance Transformer training for {epochs} epochs")
        
        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            workers=4,
            use_multiprocessing=True
