import sqlite3
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
import json
import time
import numpy as np
from typing import Dict, Any

class DynamicParameters:
    def __init__(self, db_path='cache/parameters.db'):
        self.db_path = db_path
        self.init_db()
        self.cache = {}
        self.last_update = 0
    
    def init_db(self):
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS parameters (
                key TEXT PRIMARY KEY,
                value REAL,
                updated_at REAL
            )
        ''')
        
        defaults = {
            'confidence_threshold': 0.75,
            'momentum_threshold': 0.65,
            'volatility_threshold': 0.10,
            'liquidity_threshold': 50000,
            'stop_loss_threshold': 0.05,
            'take_profit_threshold': 0.12,
            'position_size_multiplier': 1.0,
            'max_hold_time': 300
        }
        
        for key, value in defaults.items():
            conn.execute('INSERT OR IGNORE INTO parameters VALUES (?, ?, ?)',
                        (key, value, time.time()))
        conn.commit()
        conn.close()
    
    def get(self, key: str, default: float = 0.0) -> float:
        if time.time() - self.last_update > 60:
            self._refresh_cache()
        return self.cache.get(key, default)
    
    def update(self, key: str, value: float):
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT OR REPLACE INTO parameters VALUES (?, ?, ?)',
                    (key, value, time.time()))
        conn.commit()
        conn.close()
        self.cache[key] = value
    
    def _refresh_cache(self):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute('SELECT key, value FROM parameters').fetchall()
        self.cache = {key: value for key, value in rows}
        self.last_update = time.time()
        conn.close()

_params = DynamicParameters()

def get_dynamic_config() -> Dict[str, Any]:
    return {
        'confidence_threshold': _params.get('confidence_threshold', 0.75),
        'momentum_threshold': _params.get('momentum_threshold', 0.65),
        'volatility_threshold': _params.get('volatility_threshold', 0.10),
        'liquidity_threshold': _params.get('liquidity_threshold', 50000),
        'stop_loss_threshold': _params.get('stop_loss_threshold', 0.05),
        'take_profit_threshold': _params.get('take_profit_threshold', 0.12),
        'position_size_multiplier': _params.get('position_size_multiplier', 1.0),
        'max_hold_time': _params.get('max_hold_time', 300)
    }

def update_performance(roi: float, win_rate: float, sharpe: float, drawdown: float, trades: int):
    if trades < 10:
        return
    
    if roi < 0.02:
        _params.update('confidence_threshold', min(0.90, _params.get('confidence_threshold') * 1.05))
    elif roi > 0.15:
        _params.update('confidence_threshold', max(0.60, _params.get('confidence_threshold') * 0.95))
    
    if win_rate < 0.4:
        _params.update('momentum_threshold', min(0.85, _params.get('momentum_threshold') * 1.1))
    elif win_rate > 0.7:
        _params.update('momentum_threshold', max(0.50, _params.get('momentum_threshold') * 0.95))
    
    if drawdown > 0.15:
        _params.update('stop_loss_threshold', max(0.02, _params.get('stop_loss_threshold') * 0.8))
