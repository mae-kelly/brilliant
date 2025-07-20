import sqlite3
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
                updated_at REAL,
                performance_weight REAL DEFAULT 1.0
            )
        ''')
        
        # Comprehensive Renaissance-level parameters
        defaults = {
            # Core ML Thresholds
            'confidence_threshold': 0.75,
            'momentum_threshold': 0.65,
            'volatility_threshold': 0.10,
            
            # Liquidity & Risk
            'liquidity_threshold': 50000,
            'min_liquidity_threshold': 10000,
            'max_risk_score': 0.4,
            'honeypot_risk_threshold': 0.3,
            
            # Trading Execution
            'max_slippage': 0.03,
            'stop_loss_threshold': 0.05,
            'take_profit_threshold': 0.12,
            'max_hold_time': 300,
            'min_hold_time': 30,
            
            # Price Movement
            'min_price_change': 5,
            'max_price_change': 15,
            'price_momentum_decay': 0.95,
            
            # Position Management
            'max_position_size': 10.0,
            'starting_capital': 10.0,
            'kelly_multiplier': 0.25,
            'max_correlation': 0.6,
            
            # Microstructure
            'order_flow_threshold': 0.3,
            'microstructure_noise_limit': 0.1,
            'jump_intensity_threshold': 0.2,
            
            # Social Sentiment
            'sentiment_threshold': 0.6,
            'social_momentum_weight': 0.2,
            'whale_threshold': 100000,
            
            # Gas & MEV
            'max_gas_price': 50,
            'mev_protection_threshold': 0.01,
            'flashbots_threshold': 1.0,
            
            # Regime Detection
            'regime_change_threshold': 0.7,
            'volatility_regime_threshold': 0.15,
            'trend_regime_threshold': 0.05,
            
            # Performance Optimization
            'sharpe_target': 2.0,
            'max_drawdown_limit': 0.15,
            'win_rate_target': 0.6,
            'roi_target': 0.15
        }
        
        for key, value in defaults.items():
            conn.execute('INSERT OR IGNORE INTO parameters VALUES (?, ?, ?, ?)',
                        (key, value, time.time(), 1.0))
        conn.commit()
        conn.close()
    
    def get(self, key: str, default: float = 0.0) -> float:
        if time.time() - self.last_update > 30:  # Refresh every 30 seconds
            self._refresh_cache()
        return self.cache.get(key, default)
    
    def update(self, key: str, value: float, performance_weight: float = 1.0):
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT OR REPLACE INTO parameters VALUES (?, ?, ?, ?)',
                    (key, value, time.time(), performance_weight))
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
    """Get all dynamic parameters for Renaissance-level trading"""
    return {
        # Core ML Thresholds
        'confidence_threshold': _params.get('confidence_threshold', 0.75),
        'momentum_threshold': _params.get('momentum_threshold', 0.65),
        'volatility_threshold': _params.get('volatility_threshold', 0.10),
        
        # Liquidity & Risk
        'liquidity_threshold': _params.get('liquidity_threshold', 50000),
        'min_liquidity_threshold': _params.get('min_liquidity_threshold', 10000),
        'max_risk_score': _params.get('max_risk_score', 0.4),
        'honeypot_risk_threshold': _params.get('honeypot_risk_threshold', 0.3),
        
        # Trading Execution
        'max_slippage': _params.get('max_slippage', 0.03),
        'stop_loss_threshold': _params.get('stop_loss_threshold', 0.05),
        'take_profit_threshold': _params.get('take_profit_threshold', 0.12),
        'max_hold_time': _params.get('max_hold_time', 300),
        'min_hold_time': _params.get('min_hold_time', 30),
        
        # Price Movement
        'min_price_change': _params.get('min_price_change', 5),
        'max_price_change': _params.get('max_price_change', 15),
        'price_momentum_decay': _params.get('price_momentum_decay', 0.95),
        
        # Position Management
        'max_position_size': _params.get('max_position_size', 10.0),
        'starting_capital': _params.get('starting_capital', 10.0),
        'kelly_multiplier': _params.get('kelly_multiplier', 0.25),
        'max_correlation': _params.get('max_correlation', 0.6),
        
        # Microstructure
        'order_flow_threshold': _params.get('order_flow_threshold', 0.3),
        'microstructure_noise_limit': _params.get('microstructure_noise_limit', 0.1),
        'jump_intensity_threshold': _params.get('jump_intensity_threshold', 0.2),
        
        # Social Sentiment
        'sentiment_threshold': _params.get('sentiment_threshold', 0.6),
        'social_momentum_weight': _params.get('social_momentum_weight', 0.2),
        'whale_threshold': _params.get('whale_threshold', 100000),
        
        # Gas & MEV
        'max_gas_price': _params.get('max_gas_price', 50),
        'mev_protection_threshold': _params.get('mev_protection_threshold', 0.01),
        'flashbots_threshold': _params.get('flashbots_threshold', 1.0),
        
        # Regime Detection
        'regime_change_threshold': _params.get('regime_change_threshold', 0.7),
        'volatility_regime_threshold': _params.get('volatility_regime_threshold', 0.15),
        'trend_regime_threshold': _params.get('trend_regime_threshold', 0.05),
        
        # Performance Optimization
        'sharpe_target': _params.get('sharpe_target', 2.0),
        'max_drawdown_limit': _params.get('max_drawdown_limit', 0.15),
        'win_rate_target': _params.get('win_rate_target', 0.6),
        'roi_target': _params.get('roi_target', 0.15)
    }

def update_performance(roi: float, win_rate: float, sharpe: float, drawdown: float, trades: int):
    """Update parameters based on performance - Renaissance-level optimization"""
    if trades < 5:
        return
    
    config = get_dynamic_config()
    
    # Adjust confidence threshold based on performance
    if roi < config['roi_target']:
        new_threshold = min(0.95, config['confidence_threshold'] * 1.02)
        _params.update('confidence_threshold', new_threshold)
    elif roi > config['roi_target'] * 1.5:
        new_threshold = max(0.60, config['confidence_threshold'] * 0.98)
        _params.update('confidence_threshold', new_threshold)
    
    # Adjust momentum threshold based on win rate
    if win_rate < config['win_rate_target']:
        new_momentum = min(0.90, config['momentum_threshold'] * 1.05)
        _params.update('momentum_threshold', new_momentum)
    elif win_rate > config['win_rate_target'] * 1.2:
        new_momentum = max(0.50, config['momentum_threshold'] * 0.97)
        _params.update('momentum_threshold', new_momentum)
    
    # Adjust risk parameters based on drawdown
    if drawdown > config['max_drawdown_limit']:
        new_stop_loss = max(0.02, config['stop_loss_threshold'] * 0.8)
        _params.update('stop_loss_threshold', new_stop_loss)
        new_risk_score = max(0.2, config['max_risk_score'] * 0.9)
        _params.update('max_risk_score', new_risk_score)
    
    # Adjust Sharpe-based parameters
    if sharpe < config['sharpe_target']:
        new_volatility_threshold = min(0.20, config['volatility_threshold'] * 1.1)
        _params.update('volatility_threshold', new_volatility_threshold)
