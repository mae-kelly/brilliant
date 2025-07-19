import os
from dev_mode import dev_wrapper

class SafeExecutor:
    def __init__(self):
        self.original_executor = None
        
    def buy(self, token_info):
        if not dev_wrapper.config.enable_real_trading:
            print(f"[SAFE MODE] Simulating buy of {token_info.get('token', 'UNKNOWN')}")
            return dev_wrapper.simulate_trade(token_info.get('token', 'TEST'), 1.0, 'buy')
        else:
            print("ERROR: Real trading not enabled in safe mode")
            return None
    
    def sell(self, token_info):
        if not dev_wrapper.config.enable_real_trading:
            print(f"[SAFE MODE] Simulating sell of {token_info.get('token', 'UNKNOWN')}")
            return dev_wrapper.simulate_trade(token_info.get('token', 'TEST'), 1.05, 'sell')
        else:
            print("ERROR: Real trading not enabled in safe mode")
            return None

safe_executor = SafeExecutor()
