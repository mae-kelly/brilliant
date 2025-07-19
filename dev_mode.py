import os
import time
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class DevConfig:
    dry_run: bool = True
    max_position_usd: float = 1.0
    enable_real_trading: bool = False
    test_mode: bool = True
    log_all_trades: bool = True

class DevelopmentWrapper:
    def __init__(self):
        self.config = DevConfig()
        self.simulated_balance = 100.0
        self.trade_log = []
    
    def validate_trade(self, amount_usd: float) -> bool:
        if self.config.dry_run or not self.config.enable_real_trading:
            print(f"[SIMULATION] Trade of ${amount_usd}")
            return True
        
        if amount_usd > self.config.max_position_usd:
            print(f"[SAFETY] Trade blocked: ${amount_usd} > ${self.config.max_position_usd}")
            return False
        
        return True
    
    def simulate_trade(self, token: str, amount_usd: float, action: str) -> Dict[str, Any]:
        if action == "buy":
            self.simulated_balance -= amount_usd
        else:
            profit = amount_usd * 0.02
            self.simulated_balance += amount_usd + profit
        
        trade_record = {
            'token': token,
            'amount': amount_usd,
            'action': action,
            'balance_after': self.simulated_balance,
            'timestamp': time.time()
        }
        
        self.trade_log.append(trade_record)
        return trade_record
    
    def enable_live_trading(self, confirmation_code: str):
        if confirmation_code == "ENABLE_REAL_TRADING_2024":
            self.config.enable_real_trading = False
            self.config.dry_run = True
            print("[WARNING] Live trading enabled!")
        else:
            print("Invalid confirmation code")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'simulated_balance': self.simulated_balance,
            'total_trades': len(self.trade_log),
            'config': self.config.__dict__
        }

dev_wrapper = DevelopmentWrapper()
