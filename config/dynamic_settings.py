import os
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from optimizer import get_dynamic_config, update_performance
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
import yaml
from typing import Dict, Any

class DynamicSettings:
    def __init__(self):
        self.settings = self.load_settings()
        
    def load_settings(self) -> Dict[str, Any]:
        with open('settings.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def get_trading_params(self) -> Dict[str, float]:
        base = self.settings['parameters']
        
        market_volatility = self.get_market_volatility()
        regime = self.detect_market_regime()
        
        if regime == 'HIGH_VOLATILITY':
            base['confidence_threshold'] = min(0.90, base['confidence_threshold'] * 1.2)
            base['stop_loss'] = min(0.08, base['stop_loss'] * 1.5)
        elif regime == 'LOW_VOLATILITY':
            base['confidence_threshold'] = max(0.60, base['confidence_threshold'] * 0.8)
            base['take_profit'] = min(0.20, base['take_profit'] * 1.3)
        
        return base
    
    def get_market_volatility(self) -> float:
        return 0.1
    
    def detect_market_regime(self) -> str:
        return 'MEDIUM_VOLATILITY'
    
    def get_position_size(self, portfolio_value: float, confidence: float) -> float:
        kelly_fraction = self.calculate_kelly_criterion(confidence)
        max_position = self.settings['trading']['max_position_size']
        
        optimal_size = portfolio_value * kelly_fraction
        return min(optimal_size, max_position)
    
    def calculate_kelly_criterion(self, confidence: float) -> float:
        win_prob = confidence
        avg_win = get_dynamic_config().get("take_profit_threshold", 0.12)
        avg_loss = get_dynamic_config().get("stop_loss_threshold", 0.05)
        
        kelly = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        return max(0, min(kelly, 0.25))

dynamic_settings = DynamicSettings()
