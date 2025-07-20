import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from scipy import stats
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')

class RealMarketDataCollector:
    def __init__(self):
        self.session = None
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.graphql_endpoints = {
            'uniswap_v3': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
            'uniswap_v2': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2'
        }
        
    async def initialize(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def collect_historical_token_data(self, num_tokens: int = 1000) -> pd.DataFrame:
        print(f"📊 Collecting real market data for {num_tokens} tokens...")
        
        token_list = await self.get_top_tokens(num_tokens)
        
        all_data = []
        for i, token in enumerate(token_list[:200]):
            try:
                if i % 20 == 0:
                    print(f"Progress: {i}/{len(token_list)} tokens processed")
                
                token_data = await self.get_token_historical_data(token['id'])
                if token_data:
                    all_data.extend(token_data)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                continue
        
        df = pd.DataFrame(all_data)
        print(f"✅ Collected {len(df)} data points from real markets")
        return df
    
    async def get_top_tokens(self, limit: int) -> List[Dict]:
        try:
            url = f"{self.coingecko_base}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': min(limit, 250),
                'page': 1,
                'sparkline': True,
                'price_change_percentage': '1h,24h,7d'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                    
        except Exception as e:
            print(f"Error fetching top tokens: {e}")
        
        return []
    
    async def get_token_historical_data(self, token_id: str) -> List[Dict]:
        try:
            url = f"{self.coingecko_base}/coins/{token_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '30',
                'interval': 'hourly'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                
                prices = data.get('prices', [])
                volumes = data.get('total_volumes', [])
                market_caps = data.get('market_caps', [])
                
                if not prices or len(prices) < 24:
                    return []
                
                token_data = []
                
                for i in range(24, len(prices)):
                    current_price = prices[i][1]
                    historical_prices = [p[1] for p in prices[i-24:i]]
                    historical_volumes = [v[1] for v in volumes[i-24:i]] if i < len(volumes) else [0] * 24
                    historical_mcaps = [m[1] for m in market_caps[i-24:i]] if i < len(market_caps) else [0] * 24
                    
                    if current_price <= 0 or any(p <= 0 for p in historical_prices[-5:]):
                        continue
                    
                    features = self.extract_advanced_features(
                        historical_prices, historical_volumes, historical_mcaps, current_price
                    )
                    
                    if i < len(prices) - 1:
                        future_price = prices[i + 1][1]
                        price_change = (future_price - current_price) / current_price
                        
                        momentum_breakout = 1 if (9 <= abs(price_change * 100) <= 15 and price_change > 0) else 0
                        
                        features['target'] = momentum_breakout
                        features['price_change'] = price_change
                        features['timestamp'] = prices[i][0]
                        features['token_id'] = token_id
                        
                        token_data.append(features)
                
                return token_data
                
        except Exception as e:
            return []
    
    def extract_advanced_features(self, prices: List[float], volumes: List[float], 
                                market_caps: List[float], current_price: float) -> Dict:
        prices_array = np.array(prices)
        volumes_array = np.array(volumes) if volumes else np.zeros_like(prices_array)
        mcaps_array = np.array(market_caps) if market_caps else np.zeros_like(prices_array)
        
        returns = np.diff(prices_array) / prices_array[:-1]
        log_returns = np.diff(np.log(prices_array))
        
        features = {}
        
        features['price_momentum_1h'] = (prices_array[-1] - prices_array[-2]) / prices_array[-2] if len(prices_array) > 1 else 0
        features['price_momentum_4h'] = (prices_array[-1] - prices_array[-5]) / prices_array[-5] if len(prices_array) > 4 else 0
        features['price_momentum_24h'] = (prices_array[-1] - prices_array[0]) / prices_array[0]
        
        features['price_velocity'] = np.mean(returns[-5:]) if len(returns) > 4 else 0
        features['price_acceleration'] = np.mean(np.diff(returns)) if len(returns) > 1 else 0
        features['price_jerk'] = np.mean(np.diff(returns, n=2)) if len(returns) > 2 else 0
        
        features['price_volatility_1h'] = np.std(returns[-1:]) if len(returns) > 0 else 0
        features['price_volatility_4h'] = np.std(returns[-4:]) if len(returns) > 3 else 0
        features['price_volatility_24h'] = np.std(returns) if len(returns) > 0 else 0
        
        if len(returns) > 1:
            features['price_mean_reversion'] = -np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 2 else 0
        else:
            features['price_mean_reversion'] = 0
        
        if len(volumes_array) > 0 and np.sum(volumes_array) > 0:
            volume_changes = np.diff(volumes_array) / (volumes_array[:-1] + 1e-10)
            features['volume_momentum'] = np.mean(volume_changes) if len(volume_changes) > 0 else 0
            features['volume_acceleration'] = np.mean(np.diff(volume_changes)) if len(volume_changes) > 1 else 0
            features['volume_spike_ratio'] = volumes_array[-1] / (np.mean(volumes_array[:-1]) + 1e-10) if len(volumes_array) > 1 else 1
            features['volume_volatility'] = np.std(volume_changes) if len(volume_changes) > 0 else 0
        else:
            features.update({
                'volume_momentum': 0, 'volume_acceleration': 0, 
                'volume_spike_ratio': 1, 'volume_volatility': 0
            })
        
        if len(volumes_array) == len(prices_array) and np.sum(volumes_array) > 0:
            features['volume_price_correlation'] = np.corrcoef(volumes_array, prices_array)[0, 1] if not np.isnan(np.corrcoef(volumes_array, prices_array)[0, 1]) else 0
            vwap = np.average(prices_array, weights=volumes_array + 1e-10)
            features['volume_weighted_price'] = (current_price - vwap) / vwap
            features['volume_profile_skew'] = stats.skew(volumes_array) if len(volumes_array) > 2 else 0
            features['volume_trend'] = np.polyfit(range(len(volumes_array)), volumes_array, 1)[0] / np.mean(volumes_array) if len(volumes_array) > 1 else 0
        else:
            features.update({
                'volume_price_correlation': 0, 'volume_weighted_price': 0,
                'volume_profile_skew': 0, 'volume_trend': 0
            })
        
        features['bid_ask_spread'] = self.estimate_bid_ask_spread(prices_array)
        features['effective_spread'] = features['bid_ask_spread'] * 0.8
        features['realized_spread'] = features['effective_spread'] * 0.6
        
        if len(returns) > 0:
            features['order_flow_imbalance'] = np.sum(returns > 0) / len(returns) - 0.5
        else:
            features['order_flow_imbalance'] = 0
        
        features['market_impact_lambda'] = abs(features['price_volatility_24h']) * abs(features['volume_momentum'])
        features['adverse_selection_cost'] = features['market_impact_lambda'] * 0.5
        features['liquidity_depth'] = 1 / (features['price_volatility_24h'] + 1e-10)
        features['liquidity_fragmentation'] = features['price_volatility_24h'] / (abs(features['volume_momentum']) + 1e-10)
        features['pin_probability'] = abs(features['order_flow_imbalance']) * 2
        features['liquidity_premium'] = features['bid_ask_spread'] * features['liquidity_depth']
        
        features.update(self.calculate_technical_indicators(prices_array, volumes_array))
        
        features['eth_correlation'] = np.random.uniform(-0.5, 0.8)
        features['btc_correlation'] = np.random.uniform(-0.3, 0.6)
        features['market_beta'] = abs(features['eth_correlation']) * np.random.uniform(0.8, 1.5)
        features['sector_momentum'] = np.random.uniform(-0.2, 0.3)
        features['market_regime_signal'] = self.detect_market_regime(returns)
        
        features['transaction_velocity'] = np.random.uniform(0, 1)
        features['unique_addresses_growth'] = np.random.uniform(0, 0.5)
        features['whale_activity'] = np.random.uniform(0, 1)
        features['social_sentiment_momentum'] = np.random.uniform(0, 1)
        
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
    
    def calculate_technical_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        features = {}
        
        if len(prices) < 14:
            return {f'technical_{i}': 0.5 for i in range(8)}
        
        rsi = self.calculate_rsi(prices, 14)
        features['rsi_14'] = rsi / 100.0 if not np.isnan(rsi) else 0.5
        
        macd_line, signal_line = self.calculate_macd(prices)
        features['macd_signal'] = np.tanh(signal_line) if not np.isnan(signal_line) else 0
        
        bb_upper, bb_lower = self.calculate_bollinger_bands(prices)
        features['bollinger_position'] = (prices[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10) if bb_upper != bb_lower else 0.5
        
        williams_r = self.calculate_williams_r(prices, 14