import sqlite3
import json
import time
import numpy as np
import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import deque
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
import threading
import logging

@dataclass
class PerformanceMetrics:
    roi: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    avg_hold_time: float
    timestamp: float

@dataclass
class TradingParameters:
    momentum_threshold: float = 0.65
    confidence_threshold: float = 0.75
    volatility_threshold: float = 0.10
    liquidity_threshold: float = 50000
    min_liquidity_threshold: float = 10000
    max_slippage: float = 0.03
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.12
    max_hold_time: float = 300
    min_hold_time: float = 30
    min_price_change: float = 9
    max_price_change: float = 15
    price_momentum_decay: float = 0.95
    max_position_size: float = 10.0
    starting_capital: float = 10.0
    kelly_multiplier: float = 0.25
    max_correlation: float = 0.6
    order_flow_threshold: float = 0.3
    microstructure_noise_limit: float = 0.1
    jump_intensity_threshold: float = 0.2
    sentiment_threshold: float = 0.6
    social_momentum_weight: float = 0.2
    whale_threshold: float = 100000
    max_gas_price: float = 50
    mev_protection_threshold: float = 0.01
    flashbots_threshold: float = 1.0
    regime_change_threshold: float = 0.7
    volatility_regime_threshold: float = 0.15
    trend_regime_threshold: float = 0.05
    sharpe_target: float = 2.0
    max_drawdown_limit: float = 0.15
    win_rate_target: float = 0.6
    roi_target: float = 0.15

class MarketRegimeDetector:
    def __init__(self):
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.volatility_history = deque(maxlen=200)
        self.current_regime = 'normal'
        self.regime_confidence = 0.5
        
    def update_market_data(self, price: float, volume: float, volatility: float):
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.volatility_history.append(volatility)
        
        if len(self.price_history) >= 50:
            self.current_regime, self.regime_confidence = self.detect_regime()
    
    def detect_regime(self) -> Tuple[str, float]:
        if len(self.price_history) < 50:
            return 'normal', 0.5
        
        recent_vol = np.mean(list(self.volatility_history)[-20:])
        recent_returns = np.diff(list(self.price_history)[-50:])
        trend_strength = np.mean(recent_returns)
        volume_trend = np.mean(np.diff(list(self.volume_history)[-20:]))
        
        if recent_vol > 0.25:
            return 'high_volatility', 0.8
        elif recent_vol < 0.05:
            return 'low_volatility', 0.7
        elif abs(trend_strength) > 0.05:
            if trend_strength > 0:
                return 'bull_trend', 0.75
            else:
                return 'bear_trend', 0.75
        elif volume_trend > 0.3:
            return 'high_volume', 0.6
        else:
            return 'normal', 0.6

class RealTimeDynamicConfig:
    def __init__(self, db_path: str = 'data/dynamic_config.db'):
        self.db_path = db_path
        self.lock = threading.RLock()
        self.current_params = TradingParameters()
        self.performance_history = deque(maxlen=1000)
        self.regime_detector = MarketRegimeDetector()
        
        self.optimization_history = deque(maxlen=100)
        self.last_optimization = 0
        self.optimization_interval = 3600
        
        self.gp_regressor = None
        self.param_bounds = self._get_parameter_bounds()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._initialize_database()
        self._load_current_parameters()
        self._initialize_gp_regressor()

    def _initialize_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trading_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    parameters TEXT,
                    performance_score REAL,
                    roi REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    market_regime TEXT,
                    regime_confidence REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    roi REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    avg_hold_time REAL,
                    market_regime TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON trading_parameters(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)
            ''')

    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            'momentum_threshold': (0.50, 0.90),
            'confidence_threshold': (0.60, 0.95),
            'volatility_threshold': (0.05, 0.25),
            'liquidity_threshold': (10000, 200000),
            'min_liquidity_threshold': (5000, 50000),
            'max_slippage': (0.005, 0.08),
            'stop_loss_threshold': (0.02, 0.15),
            'take_profit_threshold': (0.06, 0.30),
            'max_hold_time': (60, 600),
            'min_hold_time': (10, 120),
            'min_price_change': (5, 15),
            'max_price_change': (10, 25),
            'kelly_multiplier': (0.1, 0.5),
            'max_correlation': (0.3, 0.8),
            'sentiment_threshold': (0.4, 0.8),
            'mev_protection_threshold': (0.005, 0.05),
            'regime_change_threshold': (0.5, 0.9)
        }

    def _initialize_gp_regressor(self):
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
        self.gp_regressor = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42
        )

    def _load_current_parameters(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT parameters FROM trading_parameters 
                    ORDER BY timestamp DESC LIMIT 1
                ''')
                
                row = cursor.fetchone()
                if row:
                    params_dict = json.loads(row[0])
                    for key, value in params_dict.items():
                        if hasattr(self.current_params, key):
                            setattr(self.current_params, key, value)
                    
                    self.logger.info("Loaded previous trading parameters")
                else:
                    self.logger.info("Using default trading parameters")
                    
        except Exception as e:
            self.logger.error(f"Failed to load parameters: {e}")

    def update_performance(self, roi: float, win_rate: float, sharpe_ratio: float, 
                         max_drawdown: float, total_trades: int, avg_hold_time: float = 0.0):
        
        with self.lock:
            timestamp = time.time()
            
            metrics = PerformanceMetrics(
                roi=roi,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                total_trades=total_trades,
                avg_hold_time=avg_hold_time,
                timestamp=timestamp
            )
            
            self.performance_history.append(metrics)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, roi, win_rate, sharpe_ratio, max_drawdown, total_trades, avg_hold_time, market_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, roi, win_rate, sharpe_ratio, max_drawdown, 
                    total_trades, avg_hold_time, self.regime_detector.current_regime
                ))
            
            if total_trades >= 10:
                performance_score = self._calculate_performance_score(metrics)
                
                param_record = {
                    'timestamp': timestamp,
                    'parameters': asdict(self.current_params),
                    'performance_score': performance_score,
                    'market_regime': self.regime_detector.current_regime,
                    'regime_confidence': self.regime_detector.regime_confidence
                }
                
                self.optimization_history.append(param_record)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO trading_parameters 
                        (timestamp, parameters, performance_score, roi, win_rate, sharpe_ratio, 
                         max_drawdown, total_trades, market_regime, regime_confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp, json.dumps(asdict(self.current_params)), performance_score,
                        roi, win_rate, sharpe_ratio, max_drawdown, total_trades,
                        self.regime_detector.current_regime, self.regime_detector.regime_confidence
                    ))
                
                if timestamp - self.last_optimization > self.optimization_interval:
                    self._trigger_optimization()

    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        roi_score = np.tanh(metrics.roi * 5)
        win_rate_score = metrics.win_rate
        sharpe_score = np.tanh(metrics.sharpe_ratio / 2)
        drawdown_penalty = max(0, 1 - metrics.max_drawdown * 3)
        trade_volume_bonus = min(1.0, metrics.total_trades / 100)
        
        score = (
            roi_score * 0.30 +
            win_rate_score * 0.25 +
            sharpe_score * 0.20 +
            drawdown_penalty * 0.15 +
            trade_volume_bonus * 0.10
        )
        
        return float(np.clip(score, -1.0, 1.0))

    def _trigger_optimization(self):
        try:
            if len(self.optimization_history) >= 20:
                self.logger.info("Triggering parameter optimization...")
                
                optimized_params = self._bayesian_optimize_parameters()
                
                if optimized_params:
                    self._update_parameters(optimized_params)
                    self.last_optimization = time.time()
                    
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")

    def _bayesian_optimize_parameters(self) -> Optional[Dict[str, float]]:
        if len(self.optimization_history) < 10:
            return None
        
        try:
            X_data = []
            y_data = []
            
            param_names = list(self.param_bounds.keys())
            
            for record in list(self.optimization_history)[-50:]:
                param_values = []
                for param_name in param_names:
                    value = record['parameters'].get(param_name, getattr(self.current_params, param_name))
                    param_values.append(value)
                
                X_data.append(param_values)
                y_data.append(record['performance_score'])
            
            X_data = np.array(X_data)
            y_data = np.array(y_data)
            
            if len(X_data) < 5:
                return None
            
            self.gp_regressor.fit(X_data, y_data)
            
            def objective(params):
                params_reshaped = params.reshape(1, -1)
                mean, std = self.gp_regressor.predict(params_reshaped, return_std=True)
                
                current_best = np.max(y_data)
                improvement = mean[0] - current_best
                z = improvement / (std[0] + 1e-8)
                
                from scipy.stats import norm
                ei = improvement * norm.cdf(z) + std[0] * norm.pdf(z)
                
                return -ei[0]
            
            bounds = [self.param_bounds[param] for param in param_names]
            
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=15,
                atol=1e-6,
                seed=42
            )
            
            if result.success:
                optimized_dict = {}
                for i, param_name in enumerate(param_names):
                    optimized_dict[param_name] = float(result.x[i])
                
                predicted_improvement = -result.fun
                
                if predicted_improvement > 0.01:
                    self.logger.info(f"Found optimized parameters with predicted improvement: {predicted_improvement:.4f}")
                    return optimized_dict
                    
        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {e}")
        
        return None

    def _update_parameters(self, optimized_params: Dict[str, float]):
        with self.lock:
            old_params = asdict(self.current_params)
            
            for param_name, new_value in optimized_params.items():
                if hasattr(self.current_params, param_name):
                    old_value = getattr(self.current_params, param_name)
                    
                    change_magnitude = abs(new_value - old_value) / old_value if old_value != 0 else 0
                    
                    if change_magnitude > 0.5:
                        dampened_value = old_value + (new_value - old_value) * 0.3
                        setattr(self.current_params, param_name, dampened_value)
                    else:
                        setattr(self.current_params, param_name, new_value)
            
            new_params = asdict(self.current_params)
            
            changes = []
            for param in optimized_params.keys():
                if param in old_params:
                    old_val = old_params[param]
                    new_val = new_params[param]
                    change_pct = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0
                    if abs(change_pct) > 1:
                        changes.append(f"{param}: {old_val:.4f} → {new_val:.4f} ({change_pct:+.1f}%)")
            
            if changes:
                self.logger.info("Updated parameters:")
                for change in changes[:5]:
                    self.logger.info(f"  {change}")

    def update_market_data(self, price: float, volume: float, volatility: float):
        self.regime_detector.update_market_data(price, volume, volatility)

    def get_regime_adjusted_parameters(self) -> Dict[str, Any]:
        base_params = asdict(self.current_params)
        regime = self.regime_detector.current_regime
        confidence = self.regime_detector.regime_confidence
        
        if regime == 'high_volatility' and confidence > 0.7:
            base_params['confidence_threshold'] = min(0.95, base_params['confidence_threshold'] * 1.15)
            base_params['stop_loss_threshold'] = min(0.12, base_params['stop_loss_threshold'] * 1.3)
            base_params['max_hold_time'] = max(60, base_params['max_hold_time'] * 0.7)
            
        elif regime == 'low_volatility' and confidence > 0.7:
            base_params['confidence_threshold'] = max(0.60, base_params['confidence_threshold'] * 0.9)
            base_params['take_profit_threshold'] = min(0.25, base_params['take_profit_threshold'] * 1.2)
            base_params['max_hold_time'] = min(600, base_params['max_hold_time'] * 1.3)
            
        elif regime in ['bull_trend', 'bear_trend'] and confidence > 0.7:
            base_params['momentum_threshold'] = max(0.5, base_params['momentum_threshold'] * 0.95)
            base_params['min_price_change'] = max(7, base_params['min_price_change'] * 0.9)
        
        return base_params

    def get_dynamic_config(self) -> Dict[str, Any]:
        with self.lock:
            return self.get_regime_adjusted_parameters()

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {'status': 'no_data'}
        
        recent_metrics = list(self.performance_history)[-10:]
        
        return {
            'recent_roi': np.mean([m.roi for m in recent_metrics]),
            'recent_win_rate': np.mean([m.win_rate for m in recent_metrics]),
            'recent_sharpe': np.mean([m.sharpe_ratio for m in recent_metrics]),
            'recent_drawdown': np.max([m.max_drawdown for m in recent_metrics]),
            'total_trades': sum([m.total_trades for m in recent_metrics]),
            'current_regime': self.regime_detector.current_regime,
            'regime_confidence': self.regime_detector.regime_confidence,
            'last_optimization': self.last_optimization,
            'optimization_count': len(self.optimization_history)
        }

    def force_optimization(self):
        self.last_optimization = 0
        self._trigger_optimization()

    def reset_to_defaults(self):
        with self.lock:
            self.current_params = TradingParameters()
            self.logger.info("Reset to default parameters")

global_config = RealTimeDynamicConfig()

def get_dynamic_config() -> Dict[str, Any]:
    return global_config.get_dynamic_config()

def update_performance(roi: float, win_rate: float, sharpe_ratio: float, 
                      max_drawdown: float, total_trades: int, avg_hold_time: float = 0.0):
    global_config.update_performance(roi, win_rate, sharpe_ratio, max_drawdown, total_trades, avg_hold_time)

def update_market_data(price: float, volume: float, volatility: float):
    global_config.update_market_data(price, volume, volatility)

def get_performance_summary() -> Dict[str, Any]:
    return global_config.get_performance_summary()

def force_optimization():
    global_config.force_optimization()