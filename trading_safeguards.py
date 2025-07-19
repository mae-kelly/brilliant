from safety_manager import safety_manager
from enhanced_honeypot_detector import enhanced_detector
import time
from decimal import Decimal

class TradingSafeguards:
    def __init__(self):
        self.consecutive_losses = 0
        self.last_trade_time = 0
        self.min_trade_interval = 5
        
    def pre_trade_validation(self, token_address: str, amount_usd: float) -> tuple[bool, str]:
        if not safety_manager.can_trade():
            return False, "Emergency stop active"
        
        if not safety_manager.check_trade_limits(amount_usd):
            return False, "Trade limits exceeded"
        
        if time.time() - self.last_trade_time < self.min_trade_interval:
            return False, "Minimum trade interval not met"
        
        honeypot_check = enhanced_detector.comprehensive_check(token_address)
        if not honeypot_check['is_safe']:
            safety_manager.blacklist_token(token_address, "Failed honeypot check")
            return False, "Token failed security checks"
        
        return True, "Validation passed"
    
    def post_trade_monitoring(self, token_address: str, expected_amount: float, actual_amount: float, profit_loss: float):
        if not safety_manager.check_slippage_safety(expected_amount, actual_amount):
            safety_manager.blacklist_token(token_address, "Excessive slippage detected")
        
        safety_manager.add_trade_record(token_address, actual_amount, profit_loss)
        
        if profit_loss < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 5:
                safety_manager.emergency_shutdown("5 consecutive losses")
        else:
            self.consecutive_losses = 0
        
        self.last_trade_time = time.time()

safeguards = TradingSafeguards()
