#!/usr/bin/env python3
"""
DYNAMIC PARAMETER OPTIMIZATION SYSTEM
Real-time parameter adjustment based on performance feedback
"""

import numpy as np
import json
import time
from typing import Dict, Any, List, Tuple
from collections import deque
from dataclasses import dataclass, asdict
import sqlite3
import threading
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

@dataclass
class ParameterSet:
    confidence_threshold: float
    min_momentum_score: float
    position_size_multiplier: float
    max_hold_time: int
    slippage_tolerance: float
    stop_loss_threshold: float
    take_profit_threshold: float
    volatility_threshold: float
    volume_threshold: float
    liquidity_threshold: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def to_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()))

class BayesianParameterOptimizer:
    def __init__(self, db_path: str = "cache/optimization.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.performance_history = deque(maxlen=1000)
        self.parameter_history = deque(maxlen=1000)
        
        # Initialize Gaussian Process for parameter optimization
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        # Parameter bounds (min, max)
        self.param_bounds = {
            'confidence_threshold': (0.60, 0.95),
            'min_momentum_score': (0.50, 0.90),
            'position_size_multiplier': (0.5, 2.0),
            'max_hold_time': (60, 600),
            'slippage_tolerance': (0.005, 0.05),
            'stop_loss_threshold': (0.02, 0.15),
            'take_profit_threshold': (0.05, 0.30),
            'volatility_threshold': (0.05, 0.25),
            'volume_threshold': (1000, 100000),
            'liquidity_threshold': (10000, 500000)
        }
        
        # Current best parameters
        self.current_params = ParameterSet(
            confidence_threshold=0.75,
            min_momentum_score=0.65,
            position_size_multiplier=1.0,
            max_hold_time=180,
            slippage_tolerance=0.015,
            stop_loss_threshold=0.05,
            take_profit_threshold=0.12,
            volatility_threshold=0.10,
            volume_threshold=25000,
            liquidity_threshold=50000
        )
        
        self.init_database()
        self.load_historical_data()
    
    def init_database(self):
        """Initialize optimization database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parameter_optimization (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    parameters TEXT,
                    performance_score REAL,
                    roi REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    num_trades INTEGER
                )
            """)
            conn.commit()
    
    def record_performance(self, roi: float, win_rate: float, 
                          sharpe_ratio: float, max_drawdown: float, 
                          num_trades: int):
        """Record performance with current parameters"""
        with self.lock:
            # Calculate composite performance score
            performance_score = self._calculate_performance_score(
                roi, win_rate, sharpe_ratio, max_drawdown, num_trades
            )
            
            # Store in memory
            self.performance_history.append(performance_score)
            self.parameter_history.append(self.current_params.to_array())
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO parameter_optimization 
                    (timestamp, parameters, performance_score, roi, win_rate, 
                     sharpe_ratio, max_drawdown, num_trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    time.time(),
                    json.dumps(self.current_params.to_dict()),
                    performance_score,
                    roi,
                    win_rate,
                    sharpe_ratio,
                    max_drawdown,
                    num_trades
                ))
                conn.commit()
            
            # Trigger optimization if enough data
            if len(self.performance_history) >= 10:
                self.optimize_parameters()
    
    def _calculate_performance_score(self, roi: float, win_rate: float,
                                   sharpe_ratio: float, max_drawdown: float,
                                   num_trades: int) -> float:
        """Calculate composite performance score"""
        # Normalize metrics
        roi_score = np.tanh(roi * 5)  # Scale ROI
        win_rate_score = win_rate
        sharpe_score = np.tanh(sharpe_ratio)
        drawdown_penalty = max(0, 1 - max_drawdown * 2)
        trade_volume_bonus = min(1.0, num_trades / 100)
        
        # Weighted combination
        score = (
            roi_score * 0.35 +
            win_rate_score * 0.25 +
            sharpe_score * 0.20 +
            drawdown_penalty * 0.15 +
            trade_volume_bonus * 0.05
        )
        
        return float(np.clip(score, -1.0, 1.0))
    
    def optimize_parameters(self):
        """Optimize parameters using Bayesian optimization"""
        if len(self.parameter_history) < 5:
            return
        
        try:
            # Prepare data for Gaussian Process
            X = np.array(list(self.parameter_history))
            y = np.array(list(self.performance_history))
            
            # Fit Gaussian Process
            self.gp.fit(X, y)
            
            # Find optimal parameters using acquisition function
            best_params = self._bayesian_optimization()
            
            # Update current parameters
            self.current_params = ParameterSet(
                confidence_threshold=best_params[0],
                min_momentum_score=best_params[1],
                position_size_multiplier=best_params[2],
                max_hold_time=int(best_params[3]),
                slippage_tolerance=best_params[4],
                stop_loss_threshold=best_params[5],
                take_profit_threshold=best_params[6],
                volatility_threshold=best_params[7],
                volume_threshold=best_params[8],
                liquidity_threshold=best_params[9]
            )
            
            print(f"🎯 Parameters optimized! New score estimate: {self.gp.predict([best_params])[0]:.3f}")
            
        except Exception as e:
            print(f"❌ Optimization error: {e}")
    
    def _bayesian_optimization(self) -> np.ndarray:
        """Bayesian optimization with Expected Improvement acquisition"""
        bounds = [self.param_bounds[key] for key in self.param_bounds.keys()]
        
        def acquisition(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            
            # Expected Improvement
            current_best = np.max(list(self.performance_history)[-50:])
            improvement = mu - current_best
            Z = improvement / (sigma + 1e-8)
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei[0]  # Minimize negative EI
        
        # Multiple random starts for global optimization
        best_result = None
        best_value = float('inf')
        
        for _ in range(10):
            x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            result = minimize(acquisition, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.fun < best_value:
                best_value = result.fun
                best_result = result
        
        return best_result.x if best_result else self.current_params.to_array()
    
    def get_current_parameters(self) -> ParameterSet:
        """Get current optimized parameters"""
        return self.current_params
    
    def load_historical_data(self):
        """Load historical optimization data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT parameters, performance_score 
                    FROM parameter_optimization 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """)
                
                for row in cursor.fetchall():
                    params_dict = json.loads(row[0])
                    param_array = np.array(list(params_dict.values()))
                    
                    self.parameter_history.append(param_array)
                    self.performance_history.append(row[1])
                    
        except Exception as e:
            print(f"⚠️ Could not load historical data: {e}")

# Global optimizer instance
parameter_optimizer = BayesianParameterOptimizer()

def get_dynamic_config() -> Dict[str, Any]:
    """Get current dynamic configuration"""
    params = parameter_optimizer.get_current_parameters()
    return params.to_dict()

def update_performance(roi: float, win_rate: float, sharpe_ratio: float, 
                      max_drawdown: float, num_trades: int):
    """Update performance metrics for optimization"""
    parameter_optimizer.record_performance(
        roi, win_rate, sharpe_ratio, max_drawdown, num_trades
    )

if __name__ == "__main__":
    from scipy.stats import norm
    
    # Test optimization
    optimizer = BayesianParameterOptimizer()
    
    # Simulate some performance data
    for i in range(20):
        roi = np.random.normal(0.05, 0.03)
        win_rate = np.random.uniform(0.4, 0.8)
        sharpe = np.random.normal(1.2, 0.5)
        drawdown = np.random.uniform(0.02, 0.15)
        trades = np.random.randint(10, 100)
        
        optimizer.record_performance(roi, win_rate, sharpe, drawdown, trades)
        
        if i % 5 == 0:
            print(f"Iteration {i}: Current params = {optimizer.get_current_parameters()}")
