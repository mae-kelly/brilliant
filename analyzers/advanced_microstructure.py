import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
except ImportError:
    def get_dynamic_config(): return {"volatility_threshold": 0.1, "confidence_threshold": 0.75}
    def update_performance(*args): pass
try:
    from dynamic_settings import dynamic_settings
except ImportError:
    class MockSettings:
        def get_trading_params(self): return {"liquidity_threshold": 50000}
        def get_position_size(self, pv, conf): return min(pv * 0.1, 1.0)
    dynamic_settings = MockSettings()
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import asyncio
from scipy import stats
from scipy.signal import find_peaks
import time

@dataclass
class MicrostructureMetrics:
    order_flow_toxicity: float
    adverse_selection_cost: float
    price_impact_lambda: float
    bid_ask_spread: float
    effective_spread: float
    realized_spread: float
    pin_probability: float
    volume_weighted_spread: float
    market_depth: float
    flow_imbalance: float

class OrderFlowToxicityModel:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.trade_history = deque(maxlen=window_size)
        
    def calculate_vpin(self, buy_volume: float, sell_volume: float, 
                      total_volume: float) -> float:
        
        if total_volume == 0:
            return 0.0
        
        volume_imbalance = abs(buy_volume - sell_volume)
        vpin = volume_imbalance / total_volume
        
        self.trade_history.append(vpin)
        
        if len(self.trade_history) < 10:
            return vpin
        
        return np.mean(list(self.trade_history))
    
    def calculate_adverse_selection_component(self, prices: List[float], 
                                           volumes: List[float], 
                                           trade_directions: List[int]) -> float:
        
        if len(prices) < 10:
            return 0.0
        
        price_changes = np.diff(prices)
        volume_weighted_directions = np.array(trade_directions[:-1]) * np.array(volumes[:-1])
        
        if len(price_changes) != len(volume_weighted_directions):
            min_len = min(len(price_changes), len(volume_weighted_directions))
            price_changes = price_changes[:min_len]
            volume_weighted_directions = volume_weighted_directions[:min_len]
        
        correlation = np.corrcoef(price_changes, volume_weighted_directions)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0

class MarketMicrostructureAnalyzer:
    def __init__(self):
        self.toxicity_model = OrderFlowToxicityModel()
        self.price_impact_cache = {}
        self.spread_estimator = SpreadEstimator()
        
    async def analyze_microstructure(self, token_address: str, 
                                   price_data: List[float],
                                   volume_data: List[float],
                                   trade_data: List[Dict]) -> MicrostructureMetrics:
        
        if len(price_data) < 10:
            return self.default_metrics()
        
        prices = np.array(price_data)
        volumes = np.array(volume_data)
        
        buy_volume, sell_volume = self.classify_trade_volumes(trade_data)
        trade_directions = self.infer_trade_directions(prices)
        
        order_flow_toxicity = self.toxicity_model.calculate_vpin(
            buy_volume, sell_volume, sum(volumes)
        )
        
        adverse_selection = self.toxicity_model.calculate_adverse_selection_component(
            price_data, volume_data, trade_directions
        )
        
        price_impact = await self.calculate_price_impact_lambda(prices, volumes, trade_directions)
        
        spreads = self.spread_estimator.estimate_spreads(prices, volumes)
        
        pin_probability = self.calculate_pin_probability(trade_directions)
        
        market_depth = self.estimate_market_depth(volumes)
        
        flow_imbalance = self.calculate_flow_imbalance(buy_volume, sell_volume)
        
        return MicrostructureMetrics(
            order_flow_toxicity=order_flow_toxicity,
            adverse_selection_cost=adverse_selection,
            price_impact_lambda=price_impact,
            bid_ask_spread=spreads['bid_ask'],
            effective_spread=spreads['effective'],
            realized_spread=spreads['realized'],
            pin_probability=pin_probability,
            volume_weighted_spread=spreads['volume_weighted'],
            market_depth=market_depth,
            flow_imbalance=flow_imbalance
        )
    
    def classify_trade_volumes(self, trade_data: List[Dict]) -> Tuple[float, float]:
        buy_volume = 0.0
        sell_volume = 0.0
        
        for trade in trade_data:
            volume = trade.get('volume', 0)
            if trade.get('side') == 'buy':
                buy_volume += volume
            else:
                sell_volume += volume
        
        if buy_volume == 0 and sell_volume == 0:
            total_volume = sum(trade.get('volume', 0) for trade in trade_data)
            buy_volume = total_volume * 0.5
            sell_volume = total_volume * 0.5
        
        return buy_volume, sell_volume
    
    def infer_trade_directions(self, prices: List[float]) -> List[int]:
        if len(prices) < 2:
            return [0]
        
        directions = []
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                directions.append(1)
            elif prices[i] < prices[i-1]:
                directions.append(-1)
            else:
                directions.append(0)
        
        return directions
    
    async def calculate_price_impact_lambda(self, prices: np.ndarray, 
                                          volumes: np.ndarray, 
                                          directions: List[int]) -> float:
        
        if len(prices) < 10:
            return 0.0
        
        price_changes = np.diff(prices) / prices[:-1]
        volume_imbalance = np.array(directions) * volumes[:-1] if len(directions) == len(volumes) - 1 else np.array(directions[:len(volumes)-1]) * volumes[:-1]
        
        if len(price_changes) == 0 or len(volume_imbalance) == 0:
            return 0.0
        
        min_len = min(len(price_changes), len(volume_imbalance))
        price_changes = price_changes[:min_len]
        volume_imbalance = volume_imbalance[:min_len]
        
        if np.std(volume_imbalance) == 0:
            return 0.0
        
        slope, _, r_value, _, _ = stats.linregress(volume_imbalance, price_changes)
        
        return abs(slope) * (r_value ** 2) if not np.isnan(slope) else 0.0
    
    def calculate_pin_probability(self, trade_directions: List[int]) -> float:
        if len(trade_directions) < 10:
            return 0.0
        
        buy_trades = sum(1 for d in trade_directions if d > 0)
        sell_trades = sum(1 for d in trade_directions if d < 0)
        total_trades = len(trade_directions)
        
        if total_trades == 0:
            return 0.0
        
        imbalance = abs(buy_trades - sell_trades) / total_trades
        
        return min(imbalance * 2, 1.0)
    
    def estimate_market_depth(self, volumes: np.ndarray) -> float:
        if len(volumes) == 0:
            return 0.0
        
        return np.percentile(volumes, 75)
    
    def calculate_flow_imbalance(self, buy_volume: float, sell_volume: float) -> float:
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0
        
        return (buy_volume - sell_volume) / total_volume
    
    def default_metrics(self) -> MicrostructureMetrics:
        return MicrostructureMetrics(
            order_flow_toxicity=0.0,
            adverse_selection_cost=0.0,
            price_impact_lambda=0.0,
            bid_ask_spread=0.01,
            effective_spread=0.005,
            realized_spread=0.003,
            pin_probability=0.0,
            volume_weighted_spread=0.008,
            market_depth=0.0,
            flow_imbalance=0.0
        )

class SpreadEstimator:
    def __init__(self):
        self.roll_estimator = RollSpreadEstimator()
        
    def estimate_spreads(self, prices: List[float], volumes: List[float]) -> Dict[str, float]:
        if len(prices) < 10:
            return {
                'bid_ask': 0.01,
                'effective': 0.005,
                'realized': 0.003,
                'volume_weighted': 0.008
            }
        
        bid_ask_spread = self.roll_estimator.estimate_spread(prices)
        
        effective_spread = self.calculate_effective_spread(prices)
        
        realized_spread = effective_spread * 0.6
        
        volume_weighted_spread = self.calculate_volume_weighted_spread(prices, volumes)
        
        return {
            'bid_ask': bid_ask_spread,
            'effective': effective_spread,
            'realized': realized_spread,
            'volume_weighted': volume_weighted_spread
        }
    
    def calculate_effective_spread(self, prices: List[float]) -> float:
        if len(prices) < 5:
            return 0.005
        
        high_low_spreads = []
        for i in range(len(prices) - 1):
            window = prices[max(0, i-2):i+3]
            if len(window) >= 3:
                spread = (max(window) - min(window)) / np.mean(window)
                high_low_spreads.append(spread)
        
        return np.mean(high_low_spreads) if high_low_spreads else 0.005
    
    def calculate_volume_weighted_spread(self, prices: List[float], volumes: List[float]) -> float:
        if len(prices) != len(volumes) or len(prices) < 5:
            return 0.008
        
        spreads = []
        for i in range(1, len(prices)):
            price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
            spreads.append(price_change)
        
        if len(spreads) == 0:
            return 0.008
        
        weights = volumes[1:len(spreads)+1] if len(volumes) > len(spreads) else volumes[:len(spreads)]
        
        if len(weights) != len(spreads):
            return np.mean(spreads)
        
        weighted_spread = np.average(spreads, weights=weights)
        return weighted_spread if not np.isnan(weighted_spread) else 0.008

class RollSpreadEstimator:
    def estimate_spread(self, prices: List[float]) -> float:
        if len(prices) < 10:
            return 0.01
        
        price_changes = np.diff(prices)
        
        if len(price_changes) < 2:
            return 0.01
        
        autocovariance = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
        
        if np.isnan(autocovariance) or autocovariance >= 0:
            return self.fallback_spread_estimate(prices)
        
        spread = 2 * np.sqrt(-autocovariance * np.var(price_changes))
        
        return max(spread / np.mean(prices), 0.001) if not np.isnan(spread) else 0.01
    
    def fallback_spread_estimate(self, prices: List[float]) -> float:
        if len(prices) < 5:
            return 0.01
        
        return np.std(prices) / np.mean(prices) * 2

microstructure_analyzer = MarketMicrostructureAnalyzer()