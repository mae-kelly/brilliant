import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time
import json

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
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
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
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

class BreakoutTransformer(tf.keras.Model):
    def __init__(self, num_layers=6, d_model=256, num_heads=8, dff=1024, 
                 input_vocab_size=50, maximum_position_encoding=200, rate=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)
        
        self.enc_layers = [TransformerBlock(d_model, num_heads, dff, rate) 
                          for _ in range(num_layers)]
        
        self.dropout = layers.Dropout(rate)
        
        self.global_pool = layers.GlobalAveragePooling1D()
        
        self.regime_dense = layers.Dense(128, activation='relu')
        self.momentum_dense = layers.Dense(128, activation='relu')
        self.volatility_dense = layers.Dense(64, activation='relu')
        
        self.breakout_output = layers.Dense(1, activation='sigmoid', name='breakout')
        self.regime_output = layers.Dense(5, activation='softmax', name='regime')
        self.momentum_output = layers.Dense(1, activation='tanh', name='momentum')
        self.volatility_output = layers.Dense(1, activation='sigmoid', name='volatility')
        self.confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')
    
    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        pooled = self.global_pool(x)
        
        regime_features = self.regime_dense(pooled)
        momentum_features = self.momentum_dense(pooled)
        volatility_features = self.volatility_dense(pooled)
        
        breakout_prob = self.breakout_output(pooled)
        regime_pred = self.regime_output(regime_features)
        momentum_pred = self.momentum_output(momentum_features)
        volatility_pred = self.volatility_output(volatility_features)
        confidence_pred = self.confidence_output(pooled)
        
        return {
            'breakout': breakout_prob,
            'regime': regime_pred,
            'momentum': momentum_pred,
            'volatility': volatility_pred,
            'confidence': confidence_pred
        }

class AdvancedFeatureExtractor:
    def __init__(self):
        self.lookback_window = 60
        
    def extract_price_features(self, prices):
        if len(prices) < 5:
            return np.zeros(15)
        
        prices = np.array(prices)
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        
        features = [
            prices[-1],
            np.mean(returns),
            np.std(returns),
            np.sum(returns > 0) / len(returns),
            np.percentile(returns, 75) - np.percentile(returns, 25),
            (prices[-1] - prices[0]) / prices[0],
            np.max(prices) - np.min(prices),
            np.sum(returns[-5:]) if len(returns) >= 5 else 0,
            np.mean(returns[-10:]) if len(returns) >= 10 else 0,
            np.std(returns[-10:]) if len(returns) >= 10 else 0,
            self.calculate_momentum(prices),
            self.calculate_acceleration(prices),
            self.calculate_trend_strength(prices),
            self.calculate_mean_reversion(prices),
            self.calculate_breakout_strength(prices)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def extract_volume_features(self, volumes, prices):
        if len(volumes) < 5:
            return np.zeros(10)
        
        volumes = np.array(volumes)
        prices = np.array(prices)
        
        features = [
            volumes[-1],
            np.mean(volumes),
            np.std(volumes),
            volumes[-1] / (np.mean(volumes) + 1e-6),
            np.sum(volumes[-5:]) / np.sum(volumes[:-5]) if len(volumes) > 5 else 1,
            self.calculate_volume_price_correlation(volumes, prices),
            self.calculate_volume_momentum(volumes),
            self.calculate_volume_volatility(volumes),
            self.calculate_unusual_volume(volumes),
            self.calculate_volume_trend(volumes)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def extract_microstructure_features(self, trades_data):
        if not trades_data or len(trades_data) < 5:
            return np.zeros(8)
        
        features = [
            self.calculate_bid_ask_spread(trades_data),
            self.calculate_order_flow_imbalance(trades_data),
            self.calculate_trade_intensity(trades_data),
            self.calculate_price_impact(trades_data),
            self.calculate_market_depth(trades_data),
            self.calculate_tick_size_analysis(trades_data),
            self.calculate_volatility_clustering(trades_data),
            self.calculate_jump_detection(trades_data)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def calculate_momentum(self, prices):
        if len(prices) < 10:
            return 0.0
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-10:])
        return (short_ma - long_ma) / (long_ma + 1e-10)
    
    def calculate_acceleration(self, prices):
        if len(prices) < 6:
            return 0.0
        velocity = np.diff(prices)
        acceleration = np.diff(velocity)
        return np.mean(acceleration[-3:]) if len(acceleration) >= 3 else 0.0
    
    def calculate_trend_strength(self, prices):
        if len(prices) < 5:
            return 0.0
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        r_squared = np.corrcoef(x, prices)[0, 1] ** 2
        return abs(slope) * r_squared
    
    def calculate_mean_reversion(self, prices):
        if len(prices) < 10:
            return 0.0
        mean_price = np.mean(prices)
        current_deviation = (prices[-1] - mean_price) / (mean_price + 1e-10)
        return abs(current_deviation)
    
    def calculate_breakout_strength(self, prices):
        if len(prices) < 20:
            return 0.0
        recent_max = np.max(prices[-10:])
        historical_max = np.max(prices[-20:-10])
        if historical_max > 0:
            return (recent_max - historical_max) / historical_max
        return 0.0
    
    def calculate_volume_price_correlation(self, volumes, prices):
        if len(volumes) != len(prices) or len(volumes) < 5:
            return 0.0
        return np.corrcoef(volumes, prices)[0, 1] if len(volumes) > 2 else 0.0
    
    def calculate_volume_momentum(self, volumes):
        if len(volumes) < 5:
            return 0.0
        recent_vol = np.mean(volumes[-3:])
        historical_vol = np.mean(volumes[:-3])
        return (recent_vol - historical_vol) / (historical_vol + 1e-6)
    
    def calculate_volume_volatility(self, volumes):
        if len(volumes) < 5:
            return 0.0
        return np.std(volumes) / (np.mean(volumes) + 1e-6)
    
    def calculate_unusual_volume(self, volumes):
        if len(volumes) < 10:
            return 0.0
        threshold = np.mean(volumes) + 2 * np.std(volumes)
        return 1.0 if volumes[-1] > threshold else 0.0
    
    def calculate_volume_trend(self, volumes):
        if len(volumes) < 5:
            return 0.0
        x = np.arange(len(volumes))
        slope, _ = np.polyfit(x, volumes, 1)
        return slope / (np.mean(volumes) + 1e-6)
    
    def calculate_bid_ask_spread(self, trades_data):
        if len(trades_data) < 2:
            return 0.0
        prices = [trade.get('price', 0) for trade in trades_data[-10:]]
        return (max(prices) - min(prices)) / (np.mean(prices) + 1e-10)
    
    def calculate_order_flow_imbalance(self, trades_data):
        if len(trades_data) < 2:
            return 0.0
        buy_volume = sum(trade.get('size', 0) for trade in trades_data[-10:] 
                        if trade.get('side') == 'buy')
        sell_volume = sum(trade.get('size', 0) for trade in trades_data[-10:] 
                         if trade.get('side') == 'sell')
        total_volume = buy_volume + sell_volume
        return (buy_volume - sell_volume) / (total_volume + 1e-6)
    
    def calculate_trade_intensity(self, trades_data):
        if len(trades_data) < 2:
            return 0.0
        timestamps = [trade.get('timestamp', 0) for trade in trades_data[-10:]]
        time_diffs = np.diff(sorted(timestamps))
        return 1.0 / (np.mean(time_diffs) + 1e-6) if len(time_diffs) > 0 else 0.0
    
    def calculate_price_impact(self, trades_data):
        if len(trades_data) < 3:
            return 0.0
        price_changes = []
        for i in range(1, min(len(trades_data), 6)):
            prev_price = trades_data[i-1].get('price', 0)
            curr_price = trades_data[i].get('price', 0)
            if prev_price > 0:
                price_changes.append(abs(curr_price - prev_price) / prev_price)
        return np.mean(price_changes) if price_changes else 0.0
    
    def calculate_market_depth(self, trades_data):
        if len(trades_data) < 5:
            return 0.0
        sizes = [trade.get('size', 0) for trade in trades_data[-10:]]
        return np.sum(sizes)
    
    def calculate_tick_size_analysis(self, trades_data):
        if len(trades_data) < 3:
            return 0.0
        prices = [trade.get('price', 0) for trade in trades_data[-10:]]
        tick_sizes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
        return np.std(tick_sizes) if len(tick_sizes) > 1 else 0.0
    
    def calculate_volatility_clustering(self, trades_data):
        if len(trades_data) < 5:
            return 0.0
        prices = [trade.get('price', 0) for trade in trades_data[-10:]]
        returns = [abs((prices[i] - prices[i-1]) / prices[i-1]) 
                  for i in range(1, len(prices)) if prices[i-1] > 0]
        if len(returns) < 3:
            return 0.0
        volatilities = [np.std(returns[max(0, i-2):i+1]) for i in range(2, len(returns))]
        return np.std(volatilities) if len(volatilities) > 1 else 0.0
    
    def calculate_jump_detection(self, trades_data):
        if len(trades_data) < 5:
            return 0.0
        prices = [trade.get('price', 0) for trade in trades_data[-10:]]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                  for i in range(1, len(prices)) if prices[i-1] > 0]
        if len(returns) < 3:
            return 0.0
        threshold = 3 * np.std(returns)
        jumps = [1 for r in returns if abs(r) > threshold]
        return len(jumps) / len(returns)
    
    def extract_comprehensive_features(self, price_data, volume_data, trades_data):
        price_features = self.extract_price_features(price_data)
        volume_features = self.extract_volume_features(volume_data, price_data)
        microstructure_features = self.extract_microstructure_features(trades_data)
        
        time_features = np.array([
            time.time() % 86400 / 86400,
            time.time() % 3600 / 3600,
            len(price_data) / self.lookback_window
        ], dtype=np.float32)
        
        all_features = np.concatenate([
            price_features,
            volume_features, 
            microstructure_features,
            time_features
        ])
        
        return all_features.reshape(1, -1)

class MultiObjectiveLoss:
    def __init__(self):
        self.alpha_breakout = 1.0
        self.alpha_regime = 0.3
        self.alpha_momentum = 0.2
        self.alpha_volatility = 0.15
        self.alpha_confidence = 0.25
    
    def __call__(self, y_true, y_pred):
        breakout_loss = tf.keras.losses.binary_crossentropy(
            y_true['breakout'], y_pred['breakout']
        )
        
        regime_loss = tf.keras.losses.categorical_crossentropy(
            y_true['regime'], y_pred['regime']
        )
        
        momentum_loss = tf.keras.losses.mean_squared_error(
            y_true['momentum'], y_pred['momentum']
        )
        
        volatility_loss = tf.keras.losses.mean_squared_error(
            y_true['volatility'], y_pred['volatility']
        )
        
        confidence_loss = tf.keras.losses.binary_crossentropy(
            y_true['confidence'], y_pred['confidence']
        )
        
        total_loss = (
            self.alpha_breakout * breakout_loss +
            self.alpha_regime * regime_loss +
            self.alpha_momentum * momentum_loss +
            self.alpha_volatility * volatility_loss +
            self.alpha_confidence * confidence_loss
        )
        
        return total_loss

def create_optimized_transformer(input_shape=(60, 36)):
    model = BreakoutTransformer(
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=512,
        maximum_position_encoding=100,
        rate=0.1
    )
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=10000
        ),
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss=MultiObjectiveLoss(),
        metrics={
            'breakout': ['accuracy', 'precision', 'recall'],
            'regime': ['accuracy'],
            'momentum': ['mae'],
            'volatility': ['mae'],
            'confidence': ['accuracy']
        }
    )
    
    return model