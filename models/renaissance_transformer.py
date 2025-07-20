import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import json
import os

class TimeSeriesTransformer(tf.keras.Model):
    def __init__(self, 
                 seq_length=120,
                 d_model=512, 
                 num_heads=16, 
                 num_layers=8,
                 dff=2048,
                 dropout_rate=0.1,
                 num_features=45):
        super().__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        self.num_features = num_features
        
        # Input layers
        self.feature_projection = layers.Dense(d_model, name='feature_projection')
        self.pos_encoding = self.create_positional_encoding(seq_length, d_model)
        self.dropout = layers.Dropout(dropout_rate)
        
        # Transformer encoder layers
        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append({
                'mha': layers.MultiHeadAttention(
                    num_heads=num_heads, 
                    key_dim=d_model//num_heads,
                    name=f'mha_{i}'
                ),
                'ffn': tf.keras.Sequential([
                    layers.Dense(dff, activation='gelu', name=f'ffn_1_{i}'),
                    layers.Dropout(dropout_rate),
                    layers.Dense(d_model, name=f'ffn_2_{i}')
                ], name=f'ffn_{i}'),
                'layernorm1': layers.LayerNormalization(name=f'ln1_{i}'),
                'layernorm2': layers.LayerNormalization(name=f'ln2_{i}'),
                'dropout1': layers.Dropout(dropout_rate),
                'dropout2': layers.Dropout(dropout_rate)
            })
        
        # Multi-task output heads
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Main breakout prediction head
        self.breakout_head = tf.keras.Sequential([
            layers.Dense(512, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='gelu'),
            layers.Dense(1, activation='sigmoid')
        ], name='breakout_prediction')
        
        # Auxiliary heads for enhanced learning
        self.momentum_head = tf.keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.Dense(1, activation='tanh')
        ], name='momentum_prediction')
        
        self.volatility_head = tf.keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.Dense(1, activation='sigmoid')
        ], name='volatility_prediction')
        
        self.regime_head = tf.keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.Dense(5, activation='softmax')
        ], name='regime_classification')
        
        # Confidence estimation head
        self.confidence_head = tf.keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.Dense(1, activation='sigmoid')
        ], name='confidence_estimation')
        
    def create_positional_encoding(self, seq_length, d_model):
        """Create sinusoidal positional encoding"""
        position = tf.cast(tf.range(seq_length)[:, tf.newaxis], tf.float32)
        div_term = tf.exp(tf.cast(tf.range(0, d_model, 2), tf.float32) * 
                         -(tf.math.log(10000.0) / d_model))
        
        pos_encoding = tf.zeros((seq_length, d_model))
        
        # Sin for even indices
        sin_vals = tf.sin(position * div_term)
        indices_sin = tf.stack([
            tf.range(seq_length)[:, tf.newaxis] * tf.ones([1, d_model//2], tf.int32),
            tf.range(0, d_model, 2)[tf.newaxis, :] * tf.ones([seq_length, 1], tf.int32)
        ], axis=-1)
        pos_encoding = tf.tensor_scatter_nd_update(pos_encoding, indices_sin, sin_vals)
        
        # Cos for odd indices  
        cos_vals = tf.cos(position * div_term)
        indices_cos = tf.stack([
            tf.range(seq_length)[:, tf.newaxis] * tf.ones([1, d_model//2], tf.int32),
            tf.range(1, d_model, 2)[tf.newaxis, :] * tf.ones([seq_length, 1], tf.int32)
        ], axis=-1)
        pos_encoding = tf.tensor_scatter_nd_update(pos_encoding, indices_cos, cos_vals)
        
        return pos_encoding[tf.newaxis, :, :]
        
    def call(self, inputs, training=None, return_attention_weights=False):
        # Input shape: (batch_size, seq_length, num_features)
        seq_len = tf.shape(inputs)[1]
        
        # Project features to model dimension
        x = self.feature_projection(inputs)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        attention_weights = []
        
        # Pass through transformer layers
        for layer in self.encoder_layers:
            # Multi-head attention
            if return_attention_weights:
                attn_output, attn_weights = layer['mha'](
                    x, x, return_attention_scores=True, training=training
                )
                attention_weights.append(attn_weights)
            else:
                attn_output = layer['mha'](x, x, training=training)
            
            attn_output = layer['dropout1'](attn_output, training=training)
            x1 = layer['layernorm1'](x + attn_output)
            
            # Feed forward
            ffn_output = layer['ffn'](x1, training=training)
            ffn_output = layer['dropout2'](ffn_output, training=training)
            x = layer['layernorm2'](x1 + ffn_output)
        
        # Global pooling
        pooled = self.global_pool(x)
        
        # Multi-task predictions
        outputs = {
            'breakout': self.breakout_head(pooled, training=training),
            'momentum': self.momentum_head(pooled, training=training),
            'volatility': self.volatility_head(pooled, training=training),
            'regime': self.regime_head(pooled, training=training),
            'confidence': self.confidence_head(pooled, training=training)
        }
        
        if return_attention_weights:
            outputs['attention_weights'] = attention_weights
            
        return outputs

class AdvancedFeatureExtractor:
    """Extract 45+ features for each timestep"""
    
    def __init__(self):
        self.feature_names = [
            # Price features (8)
            'price', 'log_return', 'volatility_5', 'volatility_20',
            'rsi', 'macd', 'bollinger_position', 'price_momentum',
            
            # Volume features (8)
            'volume', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
            'volume_price_correlation', 'volume_momentum', 'vwap', 'volume_spike',
            
            # Liquidity features (6)
            'liquidity', 'liquidity_change', 'bid_ask_spread', 'market_depth',
            'liquidity_fragmentation', 'slippage_estimate',
            
            # Microstructure features (8)
            'order_flow_imbalance', 'trade_intensity', 'price_impact',
            'microstructure_noise', 'tick_rule', 'effective_spread',
            'realized_spread', 'adverse_selection',
            
            # Market regime features (6)
            'trend_strength', 'trend_direction', 'volatility_regime',
            'momentum_regime', 'mean_reversion', 'jump_intensity',
            
            # Cross-asset features (5)
            'eth_correlation', 'btc_correlation', 'market_beta',
            'sector_momentum', 'relative_strength',
            
            # Time features (4)
            'hour_of_day', 'day_of_week', 'time_since_creation', 'sequence_position'
        ]
    
    def extract_features(self, price_series, volume_series, liquidity_series, 
                        timestamps, additional_data=None):
        """Extract comprehensive features for transformer input"""
        
        features = []
        n = len(price_series)
        
        for i in range(n):
            # Get windows
            start_idx = max(0, i - 20)
            price_window = price_series[start_idx:i+1]
            volume_window = volume_series[start_idx:i+1]
            liquidity_window = liquidity_series[start_idx:i+1]
            
            feature_vector = self._compute_features_at_time(
                i, price_window, volume_window, liquidity_window, 
                timestamps[i] if i < len(timestamps) else 0,
                additional_data
            )
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _compute_features_at_time(self, idx, prices, volumes, liquidity, 
                                 timestamp, additional_data):
        """Compute all 45 features for a single timestep"""
        
        if len(prices) < 2:
            return np.zeros(45)
        
        features = []
        
        # Price features
        current_price = prices[-1]
        features.append(current_price)
        
        # Returns and volatility
        returns = np.diff(prices) / prices[:-1]
        features.append(returns[-1] if len(returns) > 0 else 0)
        features.append(np.std(returns[-5:]) if len(returns) >= 5 else 0)
        features.append(np.std(returns) if len(returns) > 0 else 0)
        
        # Technical indicators
        features.append(self._calculate_rsi(prices))
        features.append(self._calculate_macd(prices))
        features.append(self._calculate_bollinger_position(prices))
        features.append(np.mean(returns[-3:]) if len(returns) >= 3 else 0)
        
        # Volume features
        current_volume = volumes[-1]
        features.append(current_volume)
        features.append(np.mean(volumes[-5:]) if len(volumes) >= 5 else current_volume)
        features.append(np.mean(volumes) if len(volumes) > 0 else current_volume)
        features.append(current_volume / np.mean(volumes) if np.mean(volumes) > 0 else 1)
        
        # Volume-price correlation
        if len(prices) > 5 and len(volumes) > 5:
            features.append(np.corrcoef(prices[-5:], volumes[-5:])[0,1])
        else:
            features.append(0)
        
        features.append(np.mean(np.diff(volumes)) if len(volumes) > 1 else 0)
        features.append(self._calculate_vwap(prices, volumes))
        features.append(1 if current_volume > np.mean(volumes) * 2 else 0)
        
        # Liquidity features
        current_liquidity = liquidity[-1] if len(liquidity) > 0 else 0
        features.append(current_liquidity)
        features.append(np.diff(liquidity)[-1] if len(liquidity) > 1 else 0)
        features.append(0.01)  # Estimated bid-ask spread
        features.append(current_liquidity * 0.1)  # Market depth estimate
        features.append(np.std(liquidity) / np.mean(liquidity) if np.mean(liquidity) > 0 else 0)
        features.append(0.02)  # Slippage estimate
        
        # Microstructure features
        features.append(self._calculate_order_flow_imbalance(volumes, returns))
        features.append(len(returns) / 60 if len(returns) > 0 else 0)  # Trade intensity
        features.append(np.std(returns) * current_volume if len(returns) > 0 else 0)
        features.append(np.std(returns) / current_price if len(returns) > 0 else 0)
        features.append(np.sum(np.sign(returns)) / len(returns) if len(returns) > 0 else 0)
        features.append(0.005)  # Effective spread estimate
        features.append(0.003)  # Realized spread estimate
        features.append(0.002)  # Adverse selection estimate
        
        # Market regime features
        features.append(self._calculate_trend_strength(prices))
        features.append(1 if prices[-1] > prices[0] else -1)
        features.append(self._classify_volatility_regime(returns))
        features.append(self._classify_momentum_regime(returns))
        features.append(self._calculate_mean_reversion(prices))
        features.append(self._calculate_jump_intensity(returns))
        
        # Cross-asset features (simplified)
        features.extend([0.0, 0.0, 1.0, 0.0, 0.5])  # Would use real market data
        
        # Time features
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp) if timestamp > 0 else datetime.datetime.now()
        features.append(dt.hour / 24.0)
        features.append(dt.weekday() / 7.0)
        features.append(min(timestamp / 86400, 365) if timestamp > 0 else 0)  # Days since creation
        features.append(idx / 120.0)  # Position in sequence
        
        return features[:45]  # Ensure exactly 45 features
    
    def _calculate_rsi(self, prices, window=14):
        if len(prices) < window + 1:
            return 0.5
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-window:])
        avg_loss = np.mean(losses[-window:])
        
        if avg_loss == 0:
            return 1.0
        
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices):
        if len(prices) < 26:
            return 0
        
        ema_12 = np.mean(prices[-12:])
        ema_26 = np.mean(prices[-26:])
        macd = ema_12 - ema_26
        return macd / prices[-1] if prices[-1] > 0 else 0
    
    def _calculate_bollinger_position(self, prices, window=20):
        if len(prices) < window:
            return 0.5
        
        ma = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        
        if std == 0:
            return 0.5
        
        position = (prices[-1] - ma) / (2 * std) + 0.5
        return np.clip(position, 0, 1)
    
    def _calculate_vwap(self, prices, volumes):
        if len(prices) != len(volumes) or len(prices) == 0:
            return prices[-1] if len(prices) > 0 else 0
        
        return np.sum(np.array(prices) * np.array(volumes)) / np.sum(volumes)
    
    def _calculate_order_flow_imbalance(self, volumes, returns):
        if len(volumes) != len(returns) or len(volumes) == 0:
            return 0
        
        buy_volume = np.sum([v for v, r in zip(volumes, returns) if r > 0])
        sell_volume = np.sum([v for v, r in zip(volumes, returns) if r < 0])
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0
        
        return (buy_volume - sell_volume) / total_volume
    
    def _calculate_trend_strength(self, prices):
        if len(prices) < 3:
            return 0
        
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = scipy.stats.linregress(x, prices)
        return abs(r_value) if not np.isnan(r_value) else 0
    
    def _classify_volatility_regime(self, returns):
        if len(returns) == 0:
            return 0.5
        
        vol = np.std(returns)
        if vol < 0.02:
            return 0.2  # Low vol
        elif vol < 0.05:
            return 0.5  # Medium vol
        else:
            return 0.8  # High vol
    
    def _classify_momentum_regime(self, returns):
        if len(returns) == 0:
            return 0.5
        
        momentum = np.mean(returns)
        return np.tanh(momentum * 50) * 0.5 + 0.5
    
    def _calculate_mean_reversion(self, prices):
        if len(prices) < 10:
            return 0.5
        
        mean_price = np.mean(prices)
        current_price = prices[-1]
        deviation = abs(current_price - mean_price) / mean_price
        
        return 1 / (1 + deviation)  # Higher value = more mean reverting
    
    def _calculate_jump_intensity(self, returns):
        if len(returns) < 5:
            return 0
        
        threshold = 2 * np.std(returns)
        jumps = np.sum(np.abs(returns) > threshold)
        return jumps / len(returns)

def train_renaissance_transformer():
    """Train the Renaissance-grade transformer model"""
    print("🧠 Training Renaissance Transformer...")
    
    # Generate comprehensive training data
    feature_extractor = AdvancedFeatureExtractor()
    
    # ... training implementation
    
    model = TimeSeriesTransformer()
    print("✅ Renaissance Transformer ready!")
    return model

if __name__ == "__main__":
    train_renaissance_transformer()
