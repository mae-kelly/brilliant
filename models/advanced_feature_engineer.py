import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
from scipy import stats
from scipy.signal import find_peaks
import ta

@dataclass
class AdvancedFeatures:
    basic_features: np.ndarray
    microstructure_features: np.ndarray
    technical_features: np.ndarray
    regime_features: np.ndarray
    combined_features: np.ndarray

class AdvancedFeatureEngineer:
    def __init__(self):
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=5000)
        self.feature_cache = {}
        
        self.feature_names = {
            'basic': [
                'price_delta', 'volume_delta', 'liquidity_delta',
                'volatility', 'velocity', 'momentum'
            ],
            'microstructure': [
                'order_flow_imbalance', 'microstructure_noise',
                'bid_ask_spread', 'market_impact', 'tick_rule',
                'trade_intensity', 'volume_imbalance'
            ],
            'technical': [
                'rsi', 'macd', 'bollinger_position', 'stochastic',
                'williams_r', 'atr', 'adx', 'cci'
            ],
            'regime': [
                'volatility_regime', 'trend_strength', 'market_phase',
                'liquidity_regime', 'jump_regime'
            ]
        }

    async def engineer_features(self, token_data: Dict, price_series: List[float], 
                              volume_series: List[float], trade_data: List[Dict]) -> AdvancedFeatures:
        
        self.update_history(price_series, volume_series, trade_data)
        
        basic_features = self.calculate_basic_features(token_data, price_series, volume_series)
        microstructure_features = self.calculate_microstructure_features(trade_data)
        technical_features = self.calculate_technical_features(price_series, volume_series)
        regime_features = self.calculate_regime_features(price_series, volume_series)
        
        combined_features = np.concatenate([
            basic_features,
            microstructure_features,
            technical_features,
            regime_features
        ])
        
        return AdvancedFeatures(
            basic_features=basic_features,
            microstructure_features=microstructure_features,
            technical_features=technical_features,
            regime_features=regime_features,
            combined_features=combined_features
        )

    def update_history(self, price_series: List[float], volume_series: List[float], trade_data: List[Dict]):
        for price in price_series:
            self.price_history.append(price)
        
        for volume in volume_series:
            self.volume_history.append(volume)
        
        for trade in trade_data:
            self.trade_history.append({
                'price': trade.get('price', 0),
                'size': trade.get('size', 0),
                'timestamp': trade.get('timestamp', time.time()),
                'side': trade.get('side', 'unknown')
            })

    def calculate_basic_features(self, token_data: Dict, price_series: List[float], 
                               volume_series: List[float]) -> np.ndarray:
        if len(price_series) < 2:
            return np.zeros(6)
        
        prices = np.array(price_series)
        volumes = np.array(volume_series) if volume_series else np.ones_like(prices)
        
        price_delta = (prices[-1] - prices[0]) / (prices[0] + 1e-10)
        
        volume_delta = 0.0
        if len(volumes) >= 10:
            recent_vol = np.mean(volumes[-5:])
            hist_vol = np.mean(volumes[-10:-5])
            volume_delta = (recent_vol - hist_vol) / (hist_vol + 1e-6)
        
        liquidity_delta = token_data.get('liquidity_delta', 0.0)
        
        volatility = np.std(np.diff(prices) / (prices[:-1] + 1e-10)) if len(prices) > 2 else 0.0
        
        velocity = price_delta / len(prices) if len(prices) > 0 else 0.0
        
        momentum = np.sum(np.diff(prices[-10:])) if len(prices) >= 10 else 0.0
        momentum = momentum / (np.mean(prices[-10:]) + 1e-10) if len(prices) >= 10 else 0.0
        
        return np.array([price_delta, volume_delta, liquidity_delta, volatility, velocity, momentum])

    def calculate_microstructure_features(self, trade_data: List[Dict]) -> np.ndarray:
        if len(trade_data) < 10:
            return np.zeros(7)
        
        recent_trades = trade_data[-100:] if len(trade_data) >= 100 else trade_data
        
        order_flow_imbalance = self.calculate_order_flow_imbalance(recent_trades)
        microstructure_noise = self.calculate_microstructure_noise(recent_trades)
        bid_ask_spread = self.estimate_bid_ask_spread(recent_trades)
        market_impact = self.calculate_market_impact(recent_trades)
        tick_rule = self.calculate_tick_rule_imbalance(recent_trades)
        trade_intensity = self.calculate_trade_intensity(recent_trades)
        volume_imbalance = self.calculate_volume_imbalance(recent_trades)
        
        return np.array([
            order_flow_imbalance, microstructure_noise, bid_ask_spread,
            market_impact, tick_rule, trade_intensity, volume_imbalance
        ])

    def calculate_order_flow_imbalance(self, trades: List[Dict]) -> float:
        if not trades:
            return 0.0
        
        buy_volume = sum(trade['size'] for trade in trades if trade.get('side') == 'buy')
        sell_volume = sum(trade['size'] for trade in trades if trade.get('side') == 'sell')
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0
        
        return (buy_volume - sell_volume) / total_volume

    def calculate_microstructure_noise(self, trades: List[Dict]) -> float:
        if len(trades) < 5:
            return 0.0
        
        prices = [trade['price'] for trade in trades]
        price_changes = np.diff(prices)
        
        if len(price_changes) == 0:
            return 0.0
        
        noise_ratio = np.std(price_changes) / (np.mean(prices) + 1e-10)
        return min(noise_ratio, 1.0)

    def estimate_bid_ask_spread(self, trades: List[Dict]) -> float:
        if len(trades) < 10:
            return 0.0
        
        prices = [trade['price'] for trade in trades]
        spread_estimate = (max(prices) - min(prices)) / (np.mean(prices) + 1e-10)
        return min(spread_estimate, 0.1)

    def calculate_market_impact(self, trades: List[Dict]) -> float:
        if len(trades) < 5:
            return 0.0
        
        volume_weighted_price_impact = 0.0
        total_volume = 0.0
        
        for i in range(1, len(trades)):
            prev_price = trades[i-1]['price']
            curr_price = trades[i]['price']
            volume = trades[i]['size']
            
            price_impact = abs(curr_price - prev_price) / (prev_price + 1e-10)
            volume_weighted_price_impact += price_impact * volume
            total_volume += volume
        
        return volume_weighted_price_impact / (total_volume + 1e-6)

    def calculate_tick_rule_imbalance(self, trades: List[Dict]) -> float:
        if len(trades) < 5:
            return 0.0
        
        tick_signs = []
        for i in range(1, len(trades)):
            if trades[i]['price'] > trades[i-1]['price']:
                tick_signs.append(1)
            elif trades[i]['price'] < trades[i-1]['price']:
                tick_signs.append(-1)
            else:
                tick_signs.append(0)
        
        if not tick_signs:
            return 0.0
        
        return np.mean(tick_signs)

    def calculate_trade_intensity(self, trades: List[Dict]) -> float:
        if len(trades) < 5:
            return 0.0
        
        timestamps = [trade['timestamp'] for trade in trades]
        time_diffs = np.diff(sorted(timestamps))
        
        if len(time_diffs) == 0:
            return 0.0
        
        avg_time_diff = np.mean(time_diffs)
        return 1.0 / (avg_time_diff + 1e-6)

    def calculate_volume_imbalance(self, trades: List[Dict]) -> float:
        if not trades:
            return 0.0
        
        volumes = [trade['size'] for trade in trades]
        
        large_trades = [v for v in volumes if v > np.percentile(volumes, 75)]
        small_trades = [v for v in volumes if v < np.percentile(volumes, 25)]
        
        if not large_trades or not small_trades:
            return 0.0
        
        large_avg = np.mean(large_trades)
        small_avg = np.mean(small_trades)
        
        return (large_avg - small_avg) / (large_avg + small_avg)

    def calculate_technical_features(self, price_series: List[float], volume_series: List[float]) -> np.ndarray:
        if len(price_series) < 20:
            return np.zeros(8)
        
        df = pd.DataFrame({
            'close': price_series,
            'volume': volume_series if volume_series else [1] * len(price_series)
        })
        
        df['high'] = df['close'] * (1 + np.random.uniform(-0.01, 0.01, len(df)))
        df['low'] = df['close'] * (1 + np.random.uniform(-0.01, 0.01, len(df)))
        
        try:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
            rsi = rsi / 100.0 if not np.isnan(rsi) else 0.5
        except:
            rsi = 0.5
        
        try:
            macd = ta.trend.MACD(df['close']).macd().iloc[-1]
            macd = np.tanh(macd * 100) if not np.isnan(macd) else 0.0
        except:
            macd = 0.0
        
        try:
            bollinger = ta.volatility.BollingerBands(df['close'])
            bb_position = (df['close'].iloc[-1] - bollinger.bollinger_lband().iloc[-1]) / (
                bollinger.bollinger_hband().iloc[-1] - bollinger.bollinger_lband().iloc[-1] + 1e-10
            )
            bb_position = bb_position if not np.isnan(bb_position) else 0.5
        except:
            bb_position = 0.5
        
        try:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch().iloc[-1]
            stoch = stoch / 100.0 if not np.isnan(stoch) else 0.5
        except:
            stoch = 0.5
        
        try:
            williams_r = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r().iloc[-1]
            williams_r = (williams_r + 100) / 100.0 if not np.isnan(williams_r) else 0.5
        except:
            williams_r = 0.5
        
        try:
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range().iloc[-1]
            atr = atr / df['close'].iloc[-1] if not np.isnan(atr) and df['close'].iloc[-1] > 0 else 0.0
        except:
            atr = 0.0
        
        try:
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx().iloc[-1]
            adx = adx / 100.0 if not np.isnan(adx) else 0.0
        except:
            adx = 0.0
        
        try:
            cci = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci().iloc[-1]
            cci = np.tanh(cci / 100.0) if not np.isnan(cci) else 0.0
        except:
            cci = 0.0
        
        return np.array([rsi, macd, bb_position, stoch, williams_r, atr, adx, cci])

    def calculate_regime_features(self, price_series: List[float], volume_series: List[float]) -> np.ndarray:
        if len(price_series) < 20:
            return np.zeros(5)
        
        prices = np.array(price_series)
        volumes = np.array(volume_series) if volume_series else np.ones_like(prices)
        
        volatility_regime = self.classify_volatility_regime(prices)
        trend_strength = self.calculate_trend_strength(prices)
        market_phase = self.identify_market_phase(prices, volumes)
        liquidity_regime = self.classify_liquidity_regime(volumes)
        jump_regime = self.detect_jump_regime(prices)
        
        return np.array([volatility_regime, trend_strength, market_phase, liquidity_regime, jump_regime])

    def classify_volatility_regime(self, prices: np.ndarray) -> float:
        if len(prices) < 10:
            return 0.5
        
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        current_vol = np.std(returns[-10:])
        historical_vol = np.std(returns[:-10]) if len(returns) > 10 else current_vol
        
        vol_ratio = current_vol / (historical_vol + 1e-10)
        
        if vol_ratio > 1.5:
            return 1.0  # High volatility regime
        elif vol_ratio < 0.7:
            return 0.0  # Low volatility regime
        else:
            return 0.5  # Normal volatility regime

    def calculate_trend_strength(self, prices: np.ndarray) -> float:
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        trend_strength = abs(r_value) if not np.isnan(r_value) else 0.0
        return trend_strength

    def identify_market_phase(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        if len(prices) < 20:
            return 0.5
        
        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
        
        if price_trend > 0 and volume_trend > 0:
            return 1.0  # Bullish expansion
        elif price_trend > 0 and volume_trend < 0:
            return 0.75  # Bullish but weakening
        elif price_trend < 0 and volume_trend > 0:
            return 0.25  # Bearish with volume
        else:
            return 0.0  # Bearish contraction

    def classify_liquidity_regime(self, volumes: np.ndarray) -> float:
        if len(volumes) < 10:
            return 0.5
        
        recent_avg_volume = np.mean(volumes[-5:])
        historical_avg_volume = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_avg_volume
        
        liquidity_ratio = recent_avg_volume / (historical_avg_volume + 1e-6)
        
        return min(liquidity_ratio / 2.0, 1.0)

    def detect_jump_regime(self, prices: np.ndarray) -> float:
        if len(prices) < 10:
            return 0.0
        
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        threshold = 3 * np.std(returns)
        
        recent_jumps = np.sum(np.abs(returns[-10:]) > threshold)
        jump_intensity = recent_jumps / 10.0
        
        return min(jump_intensity, 1.0)

    def get_feature_names(self) -> List[str]:
        all_names = []
        for category, names in self.feature_names.items():
            all_names.extend(names)
        return all_names

    def get_feature_importance(self, model) -> Dict[str, float]:
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        feature_names = self.get_feature_names()
        importances = model.feature_importances_
        
        return dict(zip(feature_names, importances))

advanced_feature_engineer = AdvancedFeatureEngineer()
