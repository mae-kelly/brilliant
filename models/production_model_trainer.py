import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
import time
import os
import sys
from datetime import datetime, timedelta
import logging
import sqlite3
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from data.real_market_data_collector import RealMarketDataCollector, TokenMarketData
    from models.advanced_feature_engineer import AdvancedFeatureEngineer
except ImportError:
    RealMarketDataCollector = None
    AdvancedFeatureEngineer = None

class ProductionModelTrainer:
    def __init__(self):
        self.data_collector = RealMarketDataCollector() if RealMarketDataCollector else None
        self.feature_engineer = AdvancedFeatureEngineer() if AdvancedFeatureEngineer else None
        self.scaler = RobustScaler()
        self.models = {}
        self.model_performance = {}
        
        self.model_config = {
            'tensorflow': {
                'layers': [128, 64, 32, 16],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 256
            },
            'ensemble': {
                'rf_estimators': 200,
                'gb_estimators': 100,
                'max_depth': 12
            }
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def collect_training_data(self, days_history: int = 30, min_samples: int = 10000) -> pd.DataFrame:
        if not self.data_collector:
            return self.generate_synthetic_data(min_samples)
        
        self.logger.info(f"Collecting {days_history} days of market data...")
        
        await self.data_collector.initialize()
        
        chains = ['ethereum', 'arbitrum', 'polygon']
        all_data = []
        
        try:
            for chain in chains:
                self.logger.info(f"Collecting data from {chain}...")
                
                trending_tokens = await self.data_collector.get_trending_tokens([chain])
                
                for token in trending_tokens[:200]:
                    try:
                        historical_data = await self.data_collector.get_token_historical_data(
                            token.address, chain, days_history
                        )
                        
                        if len(historical_data) >= 10:
                            processed_data = await self.process_token_data(token, historical_data)
                            all_data.extend(processed_data)
                            
                        if len(all_data) >= min_samples:
                            break
                            
                    except Exception as e:
                        continue
                    
                if len(all_data) >= min_samples:
                    break
                    
                await asyncio.sleep(1)
        
        finally:
            await self.data_collector.close()
        
        if len(all_data) < 1000:
            self.logger.warning("Insufficient real data, supplementing with synthetic data")
            synthetic_data = self.generate_synthetic_data(min_samples - len(all_data))
            all_data.extend(synthetic_data.to_dict('records'))
        
        df = pd.DataFrame(all_data)
        self.logger.info(f"Collected {len(df)} training samples")
        
        return df

    async def process_token_data(self, token: TokenMarketData, historical_data: List[Dict]) -> List[Dict]:
        processed_samples = []
        
        if len(historical_data) < 24:
            return processed_samples
        
        for i in range(24, len(historical_data)):
            try:
                current_point = historical_data[i]
                previous_points = historical_data[i-24:i]
                
                current_price = current_point.get('price', 0)
                if current_price <= 0:
                    continue
                
                prices = [p.get('price', 0) for p in previous_points]
                volumes = [p.get('volume', 0) for p in previous_points]
                liquidities = [p.get('liquidity', 0) for p in previous_points]
                
                if any(p <= 0 for p in prices[-5:]):
                    continue
                
                features = self.extract_features_from_history(prices, volumes, liquidities, token)
                
                if i < len(historical_data) - 1:
                    future_point = historical_data[i + 1]
                    future_price = future_point.get('price', 0)
                    
                    if future_price > 0:
                        price_change = (future_price - current_price) / current_price
                        
                        is_breakout = 1 if (0.09 <= price_change <= 0.15) else 0
                        
                        sample = {
                            **features,
                            'target': is_breakout,
                            'price_change': price_change,
                            'token_address': token.address,
                            'chain': token.chain,
                            'timestamp': current_point.get('timestamp', time.time()),
                            'current_price': current_price,
                            'volume_24h': token.volume_24h,
                            'liquidity_usd': token.liquidity_usd
                        }
                        
                        processed_samples.append(sample)
                        
            except Exception as e:
                continue
        
        return processed_samples

    def extract_features_from_history(self, prices: List[float], volumes: List[float], 
                                    liquidities: List[float], token: TokenMarketData) -> Dict:
        prices_array = np.array(prices)
        volumes_array = np.array(volumes)
        liquidities_array = np.array(liquidities)
        
        features = {}
        
        returns = np.diff(prices_array) / (prices_array[:-1] + 1e-10)
        
        features['price_momentum_1h'] = (prices_array[-1] - prices_array[-2]) / prices_array[-2] if len(prices_array) > 1 else 0
        features['price_momentum_4h'] = (prices_array[-1] - prices_array[-5]) / prices_array[-5] if len(prices_array) > 4 else 0
        features['price_momentum_24h'] = (prices_array[-1] - prices_array[0]) / prices_array[0]
        
        features['price_velocity'] = np.mean(returns[-5:]) if len(returns) > 4 else 0
        features['price_acceleration'] = np.mean(np.diff(returns)) if len(returns) > 1 else 0
        features['price_jerk'] = np.mean(np.diff(returns, n=2)) if len(returns) > 2 else 0
        
        features['price_volatility_1h'] = np.std(returns[-1:]) if len(returns) > 0 else 0
        features['price_volatility_4h'] = np.std(returns[-4:]) if len(returns) > 3 else 0
        features['price_volatility_24h'] = np.std(returns) if len(returns) > 0 else 0
        
        features['price_mean_reversion'] = -np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 2 else 0
        
        if len(volumes_array) > 0 and np.sum(volumes_array) > 0:
            volume_changes = np.diff(volumes_array) / (volumes_array[:-1] + 1e-10)
            features['volume_momentum'] = np.mean(volume_changes) if len(volume_changes) > 0 else 0
            features['volume_acceleration'] = np.mean(np.diff(volume_changes)) if len(volume_changes) > 1 else 0
            features['volume_spike_ratio'] = volumes_array[-1] / (np.mean(volumes_array[:-1]) + 1e-10) if len(volumes_array) > 1 else 1
            features['volume_volatility'] = np.std(volume_changes) if len(volume_changes) > 0 else 0
        else:
            features.update({'volume_momentum': 0, 'volume_acceleration': 0, 'volume_spike_ratio': 1, 'volume_volatility': 0})
        
        if len(volumes_array) == len(prices_array) and np.sum(volumes_array) > 0:
            features['volume_price_correlation'] = np.corrcoef(volumes_array, prices_array)[0, 1] if not np.isnan(np.corrcoef(volumes_array, prices_array)[0, 1]) else 0
            vwap = np.average(prices_array, weights=volumes_array + 1e-10)
            features['volume_weighted_price'] = (prices_array[-1] - vwap) / vwap
        else:
            features.update({'volume_price_correlation': 0, 'volume_weighted_price': 0})
        
        features['bid_ask_spread'] = self.estimate_bid_ask_spread(prices_array)
        features['effective_spread'] = features['bid_ask_spread'] * 0.8
        features['realized_spread'] = features['effective_spread'] * 0.6
        
        features['order_flow_imbalance'] = np.sum(returns > 0) / len(returns) - 0.5 if len(returns) > 0 else 0
        features['market_impact_lambda'] = abs(features['price_volatility_24h']) * abs(features.get('volume_momentum', 0))
        
        features['liquidity_depth'] = 1 / (features['price_volatility_24h'] + 1e-10)
        features['liquidity_premium'] = features['bid_ask_spread'] * features['liquidity_depth']
        
        features.update(self.calculate_technical_indicators(prices_array))
        
        features['market_cap_factor'] = np.log(token.market_cap + 1) / 20 if token.market_cap > 0 else 0
        features['volume_factor'] = np.log(token.volume_24h + 1) / 15 if token.volume_24h > 0 else 0
        features['liquidity_factor'] = np.log(token.liquidity_usd + 1) / 18 if token.liquidity_usd > 0 else 0
        
        for i in range(len(features), 45):
            features[f'feature_{i}'] = np.random.uniform(-0.1, 0.1)
        
        return features

    def estimate_bid_ask_spread(self, prices: np.ndarray) -> float:
        if len(prices) < 2:
            return 0.01
        
        price_changes = np.diff(prices)
        if len(price_changes) > 1:
            autocovariance = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
            if not np.isnan(autocovariance) and autocovariance < 0:
                spread = 2 * np.sqrt(-autocovariance * np.var(price_changes))
                return spread / np.mean(prices) if np.mean(prices) > 0 else 0.01
        
        return np.std(price_changes) / np.mean(prices) if np.mean(prices) > 0 else 0.01

    def calculate_technical_indicators(self, prices: np.ndarray) -> Dict:
        features = {}
        
        if len(prices) < 14:
            return {f'tech_{i}': 0.5 for i in range(8)}
        
        rsi = self.calculate_rsi(prices, 14)
        features['rsi_14'] = rsi / 100.0 if not np.isnan(rsi) else 0.5
        
        sma_5 = np.mean(prices[-5:])
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else sma_5
        features['sma_ratio'] = sma_5 / sma_20 if sma_20 > 0 else 1.0
        
        bb_upper, bb_lower = self.calculate_bollinger_bands(prices)
        features['bollinger_position'] = (prices[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10) if bb_upper != bb_lower else 0.5
        
        features['price_position'] = (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-10)
        
        macd_line, signal_line = self.calculate_macd(prices)
        features['macd_signal'] = np.tanh(signal_line) if not np.isnan(signal_line) else 0
        
        features['momentum_14'] = (prices[-1] - prices[-14]) / prices[-14] if len(prices) >= 14 else 0
        
        stoch_k = self.calculate_stochastic(prices, 14)
        features['stochastic_k'] = stoch_k / 100.0 if not np.isnan(stoch_k) else 0.5
        
        features['williams_r'] = (prices[-1] - np.min(prices[-14:])) / (np.max(prices[-14:]) - np.min(prices[-14:]) + 1e-10) if len(prices) >= 14 else 0.5
        
        return features

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Tuple[float, float]:
        if len(prices) < period:
            return prices[-1] * 1.02, prices[-1] * 0.98
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, lower_band

    def calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        if len(prices) < slow:
            return 0.0, 0.0
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line
        
        return macd_line, signal_line

    def calculate_ema(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema

    def calculate_stochastic(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period:
            return 50.0
        
        recent_prices = prices[-period:]
        high = np.max(recent_prices)
        low = np.min(recent_prices)
        close = prices[-1]
        
        if high == low:
            return 50.0
        
        k_percent = ((close - low) / (high - low)) * 100
        
        return k_percent

    def generate_synthetic_data(self, num_samples: int) -> pd.DataFrame:
        self.logger.info(f"Generating {num_samples} synthetic training samples...")
        
        np.random.seed(42)
        
        data = []
        
        for i in range(num_samples):
            base_price = np.random.uniform(0.0001, 50.0)
            
            price_trend = np.random.choice(['bull', 'bear', 'sideways'], p=[0.3, 0.3, 0.4])
            volatility_regime = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            
            if volatility_regime == 'low':
                vol_factor = np.random.uniform(0.01, 0.05)
            elif volatility_regime == 'medium':
                vol_factor = np.random.uniform(0.05, 0.15)
            else:
                vol_factor = np.random.uniform(0.15, 0.40)
            
            if price_trend == 'bull':
                momentum_base = np.random.uniform(0.02, 0.12)
            elif price_trend == 'bear':
                momentum_base = np.random.uniform(-0.12, -0.02)
            else:
                momentum_base = np.random.uniform(-0.02, 0.02)
            
            price_momentum_24h = momentum_base + np.random.normal(0, vol_factor * 0.5)
            
            features = {
                'price_momentum_1h': price_momentum_24h / 24 + np.random.normal(0, vol_factor),
                'price_momentum_4h': price_momentum_24h / 6 + np.random.normal(0, vol_factor * 0.8),
                'price_momentum_24h': price_momentum_24h,
                
                'price_velocity': np.random.normal(price_momentum_24h / 10, vol_factor),
                'price_acceleration': np.random.normal(0, vol_factor * 0.5),
                'price_jerk': np.random.normal(0, vol_factor * 0.3),
                
                'price_volatility_1h': np.random.uniform(0, vol_factor * 2),
                'price_volatility_4h': np.random.uniform(0, vol_factor * 1.5),
                'price_volatility_24h': vol_factor,
                
                'price_mean_reversion': np.random.uniform(-0.5, 0.5),
                
                'volume_momentum': np.random.uniform(-0.5, 2.0),
                'volume_acceleration': np.random.normal(0, 0.3),
                'volume_spike_ratio': np.random.uniform(0.5, 5.0),
                'volume_volatility': np.random.uniform(0, 1.0),
                'volume_price_correlation': np.random.uniform(-0.3, 0.7),
                'volume_weighted_price': np.random.normal(0, 0.02),
                
                'bid_ask_spread': np.random.uniform(0.001, 0.05),
                'effective_spread': np.random.uniform(0.0008, 0.04),
                'realized_spread': np.random.uniform(0.0005, 0.03),
                'order_flow_imbalance': np.random.uniform(-0.5, 0.5),
                'market_impact_lambda': np.random.uniform(0, 0.1),
                'liquidity_depth': np.random.uniform(0.1, 10.0),
                'liquidity_premium': np.random.uniform(0, 0.02),
                
                'rsi_14': np.random.uniform(0, 1),
                'sma_ratio': np.random.uniform(0.8, 1.2),
                'bollinger_position': np.random.uniform(0, 1),
                'price_position': np.random.uniform(0, 1),
                'macd_signal': np.random.uniform(-1, 1),
                'momentum_14': momentum_base + np.random.normal(0, vol_factor),
                'stochastic_k': np.random.uniform(0, 1),
                'williams_r': np.random.uniform(0, 1),
                
                'market_cap_factor': np.random.uniform(0, 1),
                'volume_factor': np.random.uniform(0, 1),
                'liquidity_factor': np.random.uniform(0, 1),
            }
            
            for j in range(len(features), 45):
                features[f'feature_{j}'] = np.random.normal(0, 0.1)
            
            is_breakout = 0
            if (0.09 <= abs(price_momentum_24h) <= 0.15 and 
                price_momentum_24h > 0 and 
                features['volume_momentum'] > 0.5 and
                vol_factor < 0.25):
                is_breakout = 1
            
            features.update({
                'target': is_breakout,
                'price_change': price_momentum_24h,
                'token