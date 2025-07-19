import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import entropy
from scipy.signal import find_peaks

class AdvancedFeatureEngine:
    def __init__(self):
        self.feature_cache = {}
        
    def calculate_order_flow_toxicity(self, trades: List[Dict]) -> float:
        if len(trades) < 10:
            return 0.0
        
        signed_trades = []
        for i, trade in enumerate(trades[1:], 1):
            prev_price = trades[i-1]['price']
            curr_price = trade['price']
            
            if curr_price > prev_price:
                signed_trades.append(trade['size'])
            elif curr_price < prev_price:
                signed_trades.append(-trade['size'])
            else:
                signed_trades.append(0)
        
        imbalance = np.sum(signed_trades)
        total_volume = np.sum([abs(t) for t in signed_trades])
        
        return abs(imbalance) / (total_volume + 1e-10)
    
    def calculate_microstructure_noise_ratio(self, prices: np.ndarray) -> float:
        if len(prices) < 20:
            return 0.0
        
        efficient_price = np.convolve(prices, np.ones(5)/5, mode='valid')
        noise = prices[2:-2] - efficient_price
        signal = efficient_price
        
        noise_power = np.var(noise)
        signal_power = np.var(signal)
        
        return noise_power / (signal_power + noise_power + 1e-10)
    
    def calculate_regime_transition_probability(self, features: Dict) -> float:
        volatility = features.get('volatility', 0.1)
        momentum = features.get('momentum', 0.0)
        volume_surge = features.get('volume_surge', 1.0)
        
        transition_score = 0.0
        
        if volatility > 0.15:
            transition_score += 0.3
        
        if abs(momentum) > 0.8:
            transition_score += 0.4
        
        if volume_surge > 3.0:
            transition_score += 0.3
        
        return min(transition_score, 1.0)
    
    def calculate_whale_detection_score(self, trades: List[Dict]) -> float:
        if not trades:
            return 0.0
        
        volumes = [t['size'] for t in trades]
        volume_threshold = np.percentile(volumes, 95)
        
        large_trades = [v for v in volumes if v > volume_threshold]
        whale_ratio = len(large_trades) / len(volumes)
        
        concentration = np.sum(large_trades) / np.sum(volumes)
        
        return min(whale_ratio * concentration * 2, 1.0)
    
    def calculate_momentum_persistence(self, prices: np.ndarray, window: int = 10) -> float:
        if len(prices) < window * 2:
            return 0.0
        
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        
        momentum_windows = []
        for i in range(len(returns) - window + 1):
            window_momentum = np.mean(returns[i:i+window])
            momentum_windows.append(1 if window_momentum > 0 else -1)
        
        if len(momentum_windows) < 2:
            return 0.0
        
        persistence = 0
        for i in range(1, len(momentum_windows)):
            if momentum_windows[i] == momentum_windows[i-1]:
                persistence += 1
        
        return persistence / (len(momentum_windows) - 1)
    
    def calculate_liquidity_shock_probability(self, liquidity_series: List[float]) -> float:
        if len(liquidity_series) < 10:
            return 0.0
        
        liquidity_changes = np.diff(liquidity_series) / (np.array(liquidity_series[:-1]) + 1e-10)
        
        shock_threshold = -0.2
        shock_count = np.sum(liquidity_changes < shock_threshold)
        
        return min(shock_count / len(liquidity_changes), 1.0)

advanced_features = AdvancedFeatureEngine()
