
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
# Dynamic configuration import


from safe_operations import logger, retry_on_failure, safe_execute, file_ops, net_ops

import time
import threading
from decimal import Decimal
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import json

@dataclass
class TradeLimit:
    max_position_usd: float = 10.0
    max_daily_loss_usd: float = 50.0
    max_trades_per_hour: int = 10
    max_slippage_percent: float = 3.0
    min_liquidity_usd: float = 10000.0

@dataclass
class SecurityCheck:
    honeypot_verified: bool = False
    lp_locked: bool = False
    contract_verified: bool = False
    no_mint_function: bool = False
    no_pause_function: bool = False

class SafetyManager:
    def __init__(self):
        self.limits = TradeLimit()
        self.daily_losses = deque(maxlen=24*60)
        self.trade_history = deque(maxlen=1000)
        self.hourly_trades = deque(maxlen=60)
        self.blacklisted_tokens = set()
        self.emergency_stop = False
        self.lock = threading.Lock()
        
    def add_trade_record(self, token: str, amount_usd: float, profit_loss: float):
        with self.lock:
            timestamp = time.time()
            self.trade_history.append({
                'token': token,
                'amount': amount_usd,
                'pnl': profit_loss,
                'timestamp': timestamp
            })
            self.hourly_trades.append(timestamp)
            if profit_loss < 0:
                self.daily_losses.append(abs(profit_loss))
    
    def check_trade_limits(self, amount_usd: float) -> bool:
        with self.lock:
            if self.emergency_stop:
                return False
            
            if amount_usd > self.limits.max_position_usd:
                return False
            
            current_hour = time.time() - 3600
            recent_trades = sum(1 for t in self.hourly_trades if t > current_hour)
            if recent_trades >= self.limits.max_trades_per_hour:
                return False
            
            daily_loss = sum(self.daily_losses)
            if daily_loss > self.limits.max_daily_loss_usd:
                return False
            
            return True
    
    def verify_token_security(self, token: str, security_data: Dict) -> bool:
        if token in self.blacklisted_tokens:
            return False
        
        required_checks = [
            security_data.get('honeypot_safe', False),
            security_data.get('lp_locked', False),
            security_data.get('contract_verified', False),
            not security_data.get('has_mint_function', True),
            not security_data.get('has_pause_function', True)
        ]
        
        passed_checks = sum(required_checks)
        return passed_checks >= 4
    
    def check_slippage_safety(self, expected_out: float, actual_out: float) -> bool:
        if expected_out == 0:
            return False
        
        slippage = abs(expected_out - actual_out) / expected_out * 100
        return slippage <= self.limits.max_slippage_percent
    
    def emergency_shutdown(self, reason: str):
        with self.lock:
            self.emergency_stop = True
            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
    
    def can_trade(self) -> bool:
        return not self.emergency_stop
    
    def blacklist_token(self, token: str, reason: str):
        with self.lock:
            self.blacklisted_tokens.add(token)
            logger.warning(f"Token blacklisted: {token} - {reason}")

safety_manager = SafetyManager()
