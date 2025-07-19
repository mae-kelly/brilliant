
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal

@dataclass
class ProductionLimits:
    max_position_usd: float = 50.0
    max_daily_loss_usd: float = 200.0
    max_trades_per_hour: int = 20
    max_slippage_percent: float = 2.5
    min_liquidity_usd: float = 50000.0
    emergency_stop_loss_percent: float = 10.0
    max_gas_price_gwei: int = 150

@dataclass
class ProductionSafety:
    require_honeypot_check: bool = True
    require_lp_lock: bool = True
    require_contract_verification: bool = True
    min_token_age_minutes: int = 30
    blacklist_on_failure: bool = True
    circuit_breaker_enabled: bool = True

class ProductionConfig:
    def __init__(self):
        self.limits = ProductionLimits()
        self.safety = ProductionSafety()
        self.load_from_env()
    
    def load_from_env(self):
        self.limits.max_position_usd = float(os.getenv('MAX_POSITION_USD', 50.0))
        self.limits.max_daily_loss_usd = float(os.getenv('MAX_DAILY_LOSS_USD', 200.0))
        self.limits.max_trades_per_hour = int(os.getenv('MAX_TRADES_PER_HOUR', 20))
        self.limits.max_slippage_percent = float(os.getenv('MAX_SLIPPAGE_PERCENT', 2.5))
        self.limits.min_liquidity_usd = float(os.getenv('MIN_LIQUIDITY_USD', 50000.0))
        
        self.safety.require_honeypot_check = os.getenv('REQUIRE_HONEYPOT_CHECK', 'true').lower() == 'true'
        self.safety.require_lp_lock = os.getenv('REQUIRE_LP_LOCK', 'true').lower() == 'true'
        self.safety.require_contract_verification = os.getenv('REQUIRE_CONTRACT_VERIFICATION', 'true').lower() == 'true'
    
    def validate_production_ready(self) -> tuple[bool, List[str]]:
        issues = []
        
        if not os.getenv('WALLET_ADDRESS') or os.getenv('WALLET_ADDRESS').startswith('0x0000'):
            issues.append("Production wallet address not configured")
        
        if not os.getenv('PRIVATE_KEY') or os.getenv('PRIVATE_KEY').startswith('0x0000'):
            issues.append("Production private key not configured")
        
        if not os.getenv('API_KEY') or 'test' in os.getenv('API_KEY', '').lower():
            issues.append("Production API key not configured")
        
        return len(issues) == 0, issues

production_config = ProductionConfig()
