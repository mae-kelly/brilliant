
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import time
import threading
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from production_config import production_config

@dataclass
class RiskMetrics:
    value_at_risk_95: float
    expected_shortfall: float
    max_correlation: float
    concentration_risk: float
    liquidity_risk: float

class AdvancedRiskManager:
    def __init__(self):
        self.positions = {}
        self.trade_history = deque(maxlen=1000)
        self.correlation_matrix = defaultdict(lambda: defaultdict(float))
        self.var_history = deque(maxlen=100)
        self.emergency_stops = 0
        self.lock = threading.Lock()
    
    def evaluate_trade_risk(self, token: str, amount_usd: float, market_data: Dict) -> Tuple[bool, str, float]:
        with self.lock:
            risk_score = 0.0
            reasons = []
            
            position_risk = self._assess_position_risk(token, amount_usd)
            if position_risk > 0.7:
                return False, "Position size risk too high", position_risk
            risk_score += position_risk * 0.3
            
            liquidity_risk = self._assess_liquidity_risk(market_data)
            if liquidity_risk > 0.8:
                return False, "Liquidity risk too high", liquidity_risk
            risk_score += liquidity_risk * 0.2
            
            correlation_risk = self._assess_correlation_risk(token)
            if correlation_risk > 0.6:
                return False, "Portfolio correlation risk too high", correlation_risk
            risk_score += correlation_risk * 0.2
            
            volatility_risk = self._assess_volatility_risk(market_data)
            if volatility_risk > 0.9:
                return False, "Market volatility too high", volatility_risk
            risk_score += volatility_risk * 0.3
            
            if risk_score > 0.65:
                return False, f"Overall risk score too high: {risk_score:.2f}", risk_score
            
            return True, "Risk acceptable", risk_score
    
    def _assess_position_risk(self, token: str, amount_usd: float) -> float:
        current_exposure = sum(pos['amount_usd'] for pos in self.positions.values())
        new_exposure = current_exposure + amount_usd
        
        if new_exposure > production_config.limits.max_position_usd * 5:
            return 1.0
        
        portfolio_concentration = amount_usd / (new_exposure + 1)
        if portfolio_concentration > 0.3:
            return 0.8
        
        return min(portfolio_concentration * 2, 0.5)
    
    def _assess_liquidity_risk(self, market_data: Dict) -> float:
        liquidity_usd = market_data.get('liquidity_usd', 0)
        volume_24h = market_data.get('volume_24h', 0)
        
        if liquidity_usd < production_config.limits.min_liquidity_usd:
            return 1.0
        
        if volume_24h < liquidity_usd * 0.1:
            return 0.8
        
        liquidity_ratio = liquidity_usd / production_config.limits.min_liquidity_usd
        return max(0, 1 - np.log(liquidity_ratio) / 5)
    
    def _assess_correlation_risk(self, token: str) -> float:
        if len(self.positions) < 2:
            return 0.0
        
        correlations = []
        for existing_token in self.positions.keys():
            if existing_token != token:
                corr = self.correlation_matrix[token][existing_token]
                correlations.append(abs(corr))
        
        if not correlations:
            return 0.0
        
        max_correlation = max(correlations)
        avg_correlation = np.mean(correlations)
        
        return min(max_correlation + avg_correlation * 0.5, 1.0)
    
    def _assess_volatility_risk(self, market_data: Dict) -> float:
        price_series = market_data.get('price_history', [])
        if len(price_series) < 10:
            return 0.5
        
        returns = np.diff(np.log(price_series))
        volatility = np.std(returns) * np.sqrt(len(returns))
        
        if volatility > 0.5:
            return 1.0
        elif volatility > 0.3:
            return 0.7
        elif volatility > 0.15:
            return 0.4
        else:
            return 0.1
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        if len(self.trade_history) < 30:
            return 0.0
        
        returns = [trade['profit_usd'] for trade in self.trade_history]
        returns_array = np.array(returns)
        
        var = np.percentile(returns_array, (1 - confidence) * 100)
        self.var_history.append(abs(var))
        
        return abs(var)
    
    def calculate_expected_shortfall(self, confidence: float = 0.95) -> float:
        if len(self.trade_history) < 30:
            return 0.0
        
        returns = [trade['profit_usd'] for trade in self.trade_history]
        returns_array = np.array(returns)
        
        var_threshold = np.percentile(returns_array, (1 - confidence) * 100)
        tail_losses = returns_array[returns_array <= var_threshold]
        
        return abs(np.mean(tail_losses)) if len(tail_losses) > 0 else 0.0
    
    def update_position(self, token: str, amount_usd: float, action: str):
        with self.lock:
            if action == 'buy':
                if token in self.positions:
                    self.positions[token]['amount_usd'] += amount_usd
                else:
                    self.positions[token] = {
                        'amount_usd': amount_usd,
                        'entry_time': time.time()
                    }
            elif action == 'sell' and token in self.positions:
                self.positions[token]['amount_usd'] -= amount_usd
                if self.positions[token]['amount_usd'] <= 0:
                    del self.positions[token]
    
    def emergency_risk_check(self) -> Tuple[bool, str]:
        total_exposure = sum(pos['amount_usd'] for pos in self.positions.values())
        
        if total_exposure > production_config.limits.max_position_usd * 10:
            self.emergency_stops += 1
            return False, f"Total exposure too high: ${total_exposure:.2f}"
        
        if len(self.var_history) > 0 and self.var_history[-1] > production_config.limits.max_daily_loss_usd:
            self.emergency_stops += 1
            return False, f"VaR exceeded: ${self.var_history[-1]:.2f}"
        
        recent_trades = list(self.trade_history)[-10:]
        if len(recent_trades) >= 10:
            recent_losses = sum(1 for trade in recent_trades if trade['profit_usd'] < 0)
            if recent_losses >= 8:
                return False, "Too many recent losses"
        
        return True, "Risk levels acceptable"
    
    def get_risk_report(self) -> RiskMetrics:
        var_95 = self.calculate_var(0.95)
        expected_shortfall = self.calculate_expected_shortfall(0.95)
        
        correlations = []
        for token1 in self.correlation_matrix:
            for token2 in self.correlation_matrix[token1]:
                if token1 != token2:
                    correlations.append(abs(self.correlation_matrix[token1][token2]))
        
        max_correlation = max(correlations) if correlations else 0.0
        
        total_exposure = sum(pos['amount_usd'] for pos in self.positions.values())
        concentration_risk = max(pos['amount_usd'] for pos in self.positions.values()) / total_exposure if total_exposure > 0 else 0.0
        
        return RiskMetrics(
            value_at_risk_95=var_95,
            expected_shortfall=expected_shortfall,
            max_correlation=max_correlation,
            concentration_risk=concentration_risk,
            liquidity_risk=0.0
        )

risk_manager = AdvancedRiskManager()
