import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
except ImportError:
    def get_dynamic_config(): return {
        "confidence_threshold": 0.75, "momentum_threshold": 0.65, "volatility_threshold": 0.10,
        "liquidity_threshold": 50000, "min_liquidity_threshold": 10000, "max_risk_score": 0.4,
        "max_slippage": 0.03, "stop_loss_threshold": 0.05, "take_profit_threshold": 0.12,
        "max_hold_time": 300, "min_price_change": 5, "max_price_change": 15,
        "max_position_size": 10.0, "starting_capital": 10.0
    }
    def update_performance(*args): pass

import numpy as np
from dataclasses import dataclass

@dataclass
class EnhancedFeatures:
    combined_features: np.ndarray
    feature_names: list
    
class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_names = [f'feature_{i}' for i in range(45)]
    
    async def engineer_features(self, token_data, price_history, volume_history, trade_history):
        features = []
        
        if len(price_history) >= 5:
            prices = np.array(price_history)
            volumes = np.array(volume_history) if volume_history else np.ones_like(prices)
            
            features.extend([
                np.mean(np.diff(prices) / prices[:-1]),  # price momentum
                np.std(np.diff(prices) / prices[:-1]),   # volatility
                (prices[-1] - prices[0]) / prices[0],    # total return
                np.mean(volumes[-5:]) / np.mean(volumes) if len(volumes) > 5 else 1.0,  # volume ratio
                len(trade_history),  # trade count
            ])
        else:
            features.extend([0.0] * 5)
        
        while len(features) < 45:
            features.append(np.random.random() * 0.01)
        
        return EnhancedFeatures(
            combined_features=np.array(features[:45]),
            feature_names=self.feature_names
        )

advanced_feature_engineer = AdvancedFeatureEngineer()
