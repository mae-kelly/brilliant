import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import joblib
from collections import defaultdict
import time

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
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
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, x, training, mask):
        attn_output, _ = self.att(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class TransformerBreakoutModel:
    def __init__(self):
        self.sequence_length = 120
        self.d_model = 256
        self.num_heads = 8
        self.num_layers = 6
        self.dff = 1024
        self.input_vocab_size = 150
        self.dropout_rate = 0.1
        self.model = None
        self.ensemble_models = []
        
    def build_model(self):
        inputs = layers.Input(shape=(self.sequence_length, self.input_vocab_size))
        
        x = layers.Dense(self.d_model)(inputs)
        x = layers.Dropout(self.dropout_rate)(x)
        
        for i in range(self.num_layers):
            x = TransformerBlock(self.d_model, self.num_heads, self.dff, self.dropout_rate)(x, training=True, mask=None)
            
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=x)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
        
    def build_ensemble(self, num_models=5):
        for i in range(num_models):
            model = self.build_model()
            self.ensemble_models.append(model)
            
    def predict_ensemble(self, token_data):
        features = self.extract_features(token_data)
        sequence_features = np.tile(features, (self.sequence_length, 1))
        sequence_features = sequence_features.reshape(1, self.sequence_length, -1)
        
        if sequence_features.shape[-1] < self.input_vocab_size:
            padding = np.zeros((1, self.sequence_length, self.input_vocab_size - sequence_features.shape[-1]))
            sequence_features = np.concatenate([sequence_features, padding], axis=-1)
        elif sequence_features.shape[-1] > self.input_vocab_size:
            sequence_features = sequence_features[:, :, :self.input_vocab_size]
            
        predictions = []
        for model in self.ensemble_models:
            pred = model.predict(sequence_features, verbose=0)
            predictions.append(pred[0][0])
            
        avg_prediction = np.mean(predictions)
        uncertainty = np.std(predictions)
        
        return {
            'breakout_probability': float(avg_prediction),
            'confidence': 1.0 - uncertainty,
            'ensemble_uncertainty': float(uncertainty),
            'prediction_timestamp': time.time()
        }
        
    def extract_features(self, token_data):
        price_series = token_data.get('price_series', [1.0]*10)
        volume_series = token_data.get('volume_series', [1000.0]*10)
        
        features = [
            np.mean(price_series),
            np.std(price_series),
            np.mean(volume_series),
            np.std(volume_series),
            token_data.get('liquidity', 10000),
            token_data.get('momentum', 0.05),
            token_data.get('velocity', 1.0),
            token_data.get('volatility', 0.02)
        ]
        
        return np.array(features, dtype=np.float32)

model = TransformerBreakoutModel()
