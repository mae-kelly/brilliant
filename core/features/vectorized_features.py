import numpy as np
import pandas as pd
import numba
from numba import jit, prange
from typing import List, Dict, Tuple
import time
import logging
import json

class VectorizedFeatureEngine:
    
    def __init__(self, feature_cache_size: int = 50000):
        self.feature_cache = {}
        self.cache_timestamps = {}
        self.feature_cache_size = feature_cache_size
        self.cache_ttl = 300
        
    def engineer_batch_features(self, price_batches: np.ndarray, 
                               metadata_batches: List[Dict] = None) -> np.ndarray:
        
        if price_batches.ndim == 2:
            batch_size, sequence_length = price_batches.shape
            feature_count = 11
        else:
            batch_size, sequence_length, feature_count = price_batches.shape
        
        if metadata_batches is None:
            metadata_batches = [{}] * batch_size
        
        features_batch = np.zeros((batch_size, feature_count), dtype=np.float32)
        
        for i in range(batch_size):
            if price_batches.ndim == 2:
                price_series = price_batches[i]
            else:
                price_series = price_batches[i, :, 0]
            
            metadata = metadata_batches[i] if i < len(metadata_batches) else {}
            features = self._engineer_single_token_vectorized(price_series, metadata)
            features_batch[i] = features
        
        return features_batch
    
    @jit(nopython=True, cache=True)
    def _calculate_returns_vectorized(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) <= 1:
            return np.array([0.0])
        
        returns = np.zeros(len(prices) - 1)
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
            else:
                returns[i-1] = 0.0
        
        return returns
    
    @jit(nopython=True, cache=True)
    def _calculate_rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        n = len(data)
        result = np.zeros(n)
        
        for i in range(n):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i+1]
            result[i] = np.std(window_data)
        
        return result
    
    @jit(nopython=True, cache=True)
    def _calculate_rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        n = len(data)
        result = np.zeros(n)
        
        for i in range(n):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i+1]
            result[i] = np.mean(window_data)
        
        return result
    
    @jit(nopython=True, cache=True)
    def _calculate_rsi_vectorized(self, prices: np.ndarray, window: int = 14) -> float:
        if len(prices) <= window:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gain = np.mean(gains[-window:])
        avg_loss = np.mean(losses[-window:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @jit(nopython=True, cache=True)
    def _calculate_bollinger_position(self, prices: np.ndarray, window: int = 20) -> float:
        if len(prices) < window:
            return 0.5
        
        recent_prices = prices[-window:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        if std_price == 0:
            return 0.5
        
        current_price = prices[-1]
        upper_band = mean_price + (2 * std_price)
        lower_band = mean_price - (2 * std_price)
        
        bb_position = (current_price - lower_band) / (upper_band - lower_band)
        
        return max(0.0, min(1.0, bb_position))
    
    def _engineer_single_token_vectorized(self, prices: np.ndarray, metadata: Dict) -> np.ndarray:
        if len(prices) == 0:
            return np.zeros(11, dtype=np.float32)
        
        if len(prices) == 1:
            return np.array([
                0.0,  # returns
                0.1,  # volatility
                0.0,  # momentum
                50.0, # rsi
                0.5,  # bb_position
                metadata.get('volume_ma', 1000.0),
                metadata.get('whale_activity', 0.1),
                0.0,  # price_acceleration
                1.0,  # volatility_ratio
                0.0,  # momentum_strength
                metadata.get('swap_volume', 1000.0)
            ], dtype=np.float32)
        
        returns = self._calculate_returns_vectorized(prices)
        
        if len(returns) == 0:
            current_return = 0.0
        else:
            current_return = returns[-1]
        
        volatility = self._calculate_rolling_std(returns, min(10, len(returns)))[-1] if len(returns) > 0 else 0.1
        
        if len(prices) >= 25:
            short_ma = self._calculate_rolling_mean(prices, 5)[-1]
            long_ma = self._calculate_rolling_mean(prices, 20)[-1]
            momentum = (short_ma - long_ma) / prices[-1] if prices[-1] != 0 else 0.0
        else:
            momentum = 0.0
        
        rsi = self._calculate_rsi_vectorized(prices)
        bb_position = self._calculate_bollinger_position(prices)
        
        volume_ma = float(metadata.get('volume_ma', 1000.0))
        whale_activity = float(metadata.get('whale_activity', 0.1))
        
        if len(returns) >= 3:
            price_acceleration = self._calculate_rolling_mean(np.diff(returns), 3)[-1] if len(returns) > 3 else 0.0
        else:
            price_acceleration = 0.0
        
        if len(prices) >= 20:
            current_vol = volatility
            historical_vol = self._calculate_rolling_std(returns, min(20, len(returns)))
            avg_historical_vol = np.mean(historical_vol) if len(historical_vol) > 0 else volatility
            volatility_ratio = current_vol / avg_historical_vol if avg_historical_vol != 0 else 1.0
        else:
            volatility_ratio = 1.0
        
        momentum_strength = abs(momentum) * volatility_ratio
        swap_volume = float(metadata.get('swap_volume', volume_ma))
        
        features = np.array([
            current_return,
            volatility,
            momentum,
            rsi,
            bb_position,
            volume_ma,
            whale_activity,
            price_acceleration,
            volatility_ratio,
            momentum_strength,
            swap_volume
        ], dtype=np.float32)
        
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features
    
    def engineer_features_with_cache(self, token_address: str, prices: np.ndarray, 
                                   metadata: Dict) -> np.ndarray:
        
        current_time = time.time()
        cache_key = f"{token_address}_{len(prices)}"
        
        if (cache_key in self.feature_cache and 
            current_time - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl):
            return self.feature_cache[cache_key]
        
        features = self._engineer_single_token_vectorized(prices, metadata)
        
        if len(self.feature_cache) >= self.feature_cache_size:
            self._evict_old_cache_entries()
        
        self.feature_cache[cache_key] = features
        self.cache_timestamps[cache_key] = current_time
        
        return features
    
    def _evict_old_cache_entries(self):
        
        current_time = time.time()
        keys_to_remove = []
        
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self.feature_cache:
                del self.feature_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
        
        if len(self.feature_cache) >= self.feature_cache_size:
            oldest_keys = sorted(self.cache_timestamps.items(), key=lambda x: x[1])[:100]
            for key, _ in oldest_keys:
                if key in self.feature_cache:
                    del self.feature_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
    
    def precompute_features_batch(self, token_data_list: List[Tuple[str, np.ndarray, Dict]]):
        
        start_time = time.time()
        
        for token_address, prices, metadata in token_data_list:
            self.engineer_features_with_cache(token_address, prices, metadata)
        
        computation_time = time.time() - start_time
        
        logging.info(json.dumps({
            'event': 'batch_feature_computation',
            'tokens_processed': len(token_data_list),
            'computation_time': computation_time,
            'tokens_per_second': len(token_data_list) / computation_time if computation_time > 0 else 0
        }))
    
    def get_cache_stats(self) -> Dict:
        
        current_time = time.time()
        valid_entries = sum(1 for timestamp in self.cache_timestamps.values() 
                          if current_time - timestamp < self.cache_ttl)
        
        return {
            'total_entries': len(self.feature_cache),
            'valid_entries': valid_entries,
            'cache_utilization': len(self.feature_cache) / self.feature_cache_size,
            'hit_rate_estimate': valid_entries / max(len(self.feature_cache), 1)
        }
    
    def engineer_advanced_features(self, price_batches: np.ndarray, 
                                 volume_batches: np.ndarray = None,
                                 metadata_batches: List[Dict] = None) -> np.ndarray:
        
        batch_size = price_batches.shape[0]
        advanced_feature_count = 15
        
        features_batch = np.zeros((batch_size, advanced_feature_count), dtype=np.float32)
        
        for i in range(batch_size):
            prices = price_batches[i]
            volumes = volume_batches[i] if volume_batches is not None else np.ones_like(prices) * 1000
            metadata = metadata_batches[i] if metadata_batches is not None else {}
            
            basic_features = self._engineer_single_token_vectorized(prices, metadata)
            
            advanced_features = self._calculate_advanced_features(prices, volumes, metadata)
            
            combined_features = np.concatenate([basic_features, advanced_features])[:advanced_feature_count]
            features_batch[i] = combined_features
        
        return features_batch
    
    @jit(nopython=True, cache=True)
    def _calculate_advanced_features(self, prices: np.ndarray, volumes: np.ndarray, 
                                   metadata: Dict) -> np.ndarray:
        
        if len(prices) < 5:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        returns = self._calculate_returns_vectorized(prices)
        
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        
        if len(volumes) == len(prices):
            vwap = np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else prices[-1]
            volume_price_correlation = np.corrcoef(prices[-min(20, len(prices)):], 
                                                 volumes[-min(20, len(volumes)):])[0, 1] if len(prices) >= 2 else 0.0
        else:
            vwap = prices[-1]
            volume_price_correlation = 0.0
        
        if np.isnan(volume_price_correlation):
            volume_price_correlation = 0.0
        
        return np.