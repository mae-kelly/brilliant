import numpy as np
import pandas as pd
import asyncio
import time
import logging
import json
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import sqlite3
from prometheus_client import Gauge, Counter
import yaml

class ContinuousOptimizer:
    
    def __init__(self, momentum_model, risk_manager):
        self.momentum_model = momentum_model
        self.risk_manager = risk_manager
        
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)
        
        self.optimization_history = []
        self.parameter_bounds = {
            'momentum_threshold': (0.05, 0.15),
            'velocity_threshold': (0.08, 0.20),
            'decay_threshold': (0.003, 0.010),
            'prediction_confidence': (0.65, 0.90),
            'position_size_multiplier': (0.5, 2.0),
            'stop_loss_threshold': (0.01, 0.05),
            'take_profit_threshold': (0.08, 0.25)
        }
        
        self.current_parameters = {
            'momentum_threshold': self.settings['trading']['momentum_threshold'],
            'velocity_threshold': self.settings['trading']['velocity_threshold'],
            'decay_threshold': self.settings['trading']['decay_threshold'],
            'prediction_confidence': self.settings['ml']['prediction_confidence'],
            'position_size_multiplier': 1.0,
            'stop_loss_threshold': self.settings['trading']['stop_loss_threshold'],
            'take_profit_threshold': self.settings['trading']['take_profit_threshold']
        }
        
        self.performance_buffer = []
        self.optimization_interval = 3600
        self.last_optimization = time.time()
        
        self.gp_model = GaussianProcessRegressor(
            kernel=ConstantKernel() * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        self.optimization_gauge = Gauge('optimization_score', 'Current optimization score')
        self.parameter_gauges = {
            param: Gauge(f'parameter_{param}', f'Current value of {param}')
            for param in self.parameter_bounds.keys()
        }
        self.optimization_counter = Counter('optimizations_performed', 'Total optimizations performed')
        
        self.conn = sqlite3.connect('optimization_history.db')
        self.init_database()
    
    def init_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                parameters TEXT,
                performance_score REAL,
                sharpe_ratio REAL,
                win_rate REAL,
                total_trades INTEGER,
                optimization_method TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameter_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                parameter_name TEXT,
                old_value REAL,
                new_value REAL,
                performance_impact REAL
            )
        ''')
        
        self.conn.commit()
    
    async def optimize_parameters(self):
        if time.time() - self.last_optimization < self.optimization_interval:
            return
        
        logging.info("Starting continuous parameter optimization")
        optimization_start = time.time()
        
        try:
            performance_data = await self.collect_performance_data()
            
            if len(performance_data) < 10:
                logging.warning("Insufficient performance data for optimization")
                return
            
            optimization_methods = [
                ('bayesian', self.bayesian_optimization),
                ('differential_evolution', self.differential_evolution_optimization),
                ('gradient_free', self.gradient_free_optimization)
            ]
            
            best_score = -np.inf
            best_parameters = self.current_parameters.copy()
            best_method = None
            
            for method_name, optimization_func in optimization_methods:
                try:
                    candidate_params, score = await optimization_func(performance_data)
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = candidate_params
                        best_method = method_name
                        
                except Exception as e:
                    logging.error(f"Optimization method {method_name} failed: {e}")
                    continue
            
            if best_score > self.calculate_current_performance_score(performance_data):
                await self.apply_parameter_updates(best_parameters, best_method)
                self.optimization_gauge.set(best_score)
                logging.info(f"Parameters optimized using {best_method}, score improved to {best_score:.4f}")
            else:
                logging.info("No parameter improvement found")
            
            self.optimization_counter.inc()
            self.last_optimization = time.time()
            
            optimization_time = time.time() - optimization_start
            logging.info(f"Optimization completed in {optimization_time:.2f}s")
            
        except Exception as e:
            logging.error(f"Parameter optimization failed: {e}")
    
    async def collect_performance_data(self) -> List[Dict]:
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM optimization_runs 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''', (time.time() - 86400,))
            
            rows = cursor.fetchall()
            performance_data = []
            
            for row in rows:
                performance_data.append({
                    'timestamp': row[1],
                    'parameters': json.loads(row[2]),
                    'performance_score': row[3],
                    'sharpe_ratio': row[4],
                    'win_rate': row[5],
                    'total_trades': row[6],
                    'optimization_method': row[7]
                })
            
            if hasattr(self.momentum_model, 'get_recent_performance'):
                recent_performance = self.momentum_model.get_recent_performance()
                if recent_performance:
                    performance_data.append({
                        'timestamp': time.time(),
                        'parameters': self.current_parameters,
                        'performance_score': recent_performance.get('score', 0),
                        'sharpe_ratio': recent_performance.get('sharpe_ratio', 0),
                        'win_rate': recent_performance.get('win_rate', 0),
                        'total_trades': recent_performance.get('total_trades', 0),
                        'optimization_method': 'current'
                    })
            
            return performance_data
            
        except Exception as e:
            logging.error(f"Failed to collect performance data: {e}")
            return []
    
    async def bayesian_optimization(self, performance_data: List[Dict]) -> Tuple[Dict, float]:
        try:
            if len(performance_data) < 5:
                raise ValueError("Insufficient data for Bayesian optimization")
            
            X = []
            y = []
            
            for data_point in performance_data:
                params = data_point['parameters']
                score = data_point['performance_score']
                
                param_vector = [params.get(key, self.current_parameters[key]) 
                              for key in self.parameter_bounds.keys()]
                X.append(param_vector)
                y.append(score)
            
            X = np.array(X)
            y = np.array(y)
            
            self.gp_model.fit(X, y)
            
            def acquisition_function(params):
                params_array = np.array(params).reshape(1, -1)
                mean, std = self.gp_model.predict(params_array, return_std=True)
                return -(mean[0] + 2.0 * std[0])  # Upper confidence bound
            
            bounds = [self.parameter_bounds[key] for key in self.parameter_bounds.keys()]
            
            result = minimize(
                acquisition_function,
                x0=[self.current_parameters[key] for key in self.parameter_bounds.keys()],
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                optimized_params = dict(zip(self.parameter_bounds.keys(), result.x))
                predicted_score = -result.fun
                
                return optimized_params, predicted_score
            else:
                raise ValueError("Bayesian optimization failed to converge")
                
        except Exception as e:
            logging.error(f"Bayesian optimization error: {e}")
            return self.current_parameters.copy(), 0.0
    
    async def differential_evolution_optimization(self, performance_data: List[Dict]) -> Tuple[Dict, float]:
        try:
            def objective_function(param_vector):
                params = dict(zip(self.parameter_bounds.keys(), param_vector))
                return -self.estimate_performance_score(params, performance_data)
            
            bounds = [self.parameter_bounds[key] for key in self.parameter_bounds.keys()]
            
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=10,
                seed=42,
                atol=1e-6
            )
            
            if result.success:
                optimized_params = dict(zip(self.parameter_bounds.keys(), result.x))
                predicted_score = -result.fun
                
                return optimized_params, predicted_score
            else:
                raise ValueError("Differential evolution failed to converge")
                
        except Exception as e:
            logging.error(f"Differential evolution error: {e}")
            return self.current_parameters.copy(), 0.0
    
    async def gradient_free_optimization(self, performance_data: List[Dict]) -> Tuple[Dict, float]:
        try:
            best_params = self.current_parameters.copy()
            best_score = self.calculate_current_performance_score(performance_data)
            
            search_radius = 0.1
            num_iterations = 20
            
            for iteration in range(num_iterations):
                candidate_params = {}
                
                for param_name, current_value in self.current_parameters.items():
                    if param_name in self.parameter_bounds:
                        lower_bound, upper_bound = self.parameter_bounds[param_name]
                        
                        perturbation = np.random.uniform(-search_radius, search_radius) * current_value
                        new_value = current_value + perturbation
                        new_value = np.clip(new_value, lower_bound, upper_bound)
                        
                        candidate_params[param_name] = new_value
                    else:
                        candidate_params[param_name] = current_value
                
                candidate_score = self.estimate_performance_score(candidate_params, performance_data)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_params = candidate_params.copy()
                
                search_radius *= 0.95
            
            return best_params, best_score
            
        except Exception as e:
            logging.error(f"Gradient-free optimization error: {e}")
            return self.current_parameters.copy(), 0.0
    
    def estimate_performance_score(self, parameters: Dict, performance_data: List[Dict]) -> float:
        try:
            if not performance_data:
                return 0.0
            
            recent_data = performance_data[-10:] if len(performance_data) >= 10 else performance_data
            
            similarity_weights = []
            performance_scores = []
            
            for data_point in recent_data:
                historical_params = data_point['parameters']
                
                similarity = self.calculate_parameter_similarity(parameters, historical_params)
                similarity_weights.append(similarity)
                performance_scores.append(data_point['performance_score'])
            
            if sum(similarity_weights) == 0:
                return np.mean(performance_scores)
            
            weighted_score = np.average(performance_scores, weights=similarity_weights)
            
            parameter_penalty = self.calculate_parameter_penalty(parameters)
            
            return weighted_score - parameter_penalty
            
        except Exception as e:
            logging.error(f"Performance estimation error: {e}")
            return 0.0
    
    def calculate_parameter_similarity(self, params1: Dict, params2: Dict) -> float:
        try:
            similarities = []
            
            for param_name in self.parameter_bounds.keys():
                value1 = params1.get(param_name, self.current_parameters[param_name])
                value2 = params2.get(param_name, self.current_parameters[param_name])
                
                lower_bound, upper_bound = self.parameter_bounds[param_name]
                param_range = upper_bound - lower_bound
                
                normalized_diff = abs(value1 - value2) / param_range
                similarity = np.exp(-normalized_diff * 5)
                similarities.append(similarity)
            
            return np.mean(similarities)
            
        except Exception as e:
            logging.error(f"Similarity calculation error: {e}")
            return 0.0
    
    def calculate_parameter_penalty(self, parameters: Dict) -> float:
        penalty = 0.0
        
        if parameters.get('momentum_threshold', 0) > 0.12:
            penalty += 0.1
        
        if parameters.get('velocity_threshold', 0) > 0.18:
            penalty += 0.05
        
        if parameters.get('decay_threshold', 0) < 0.004:
            penalty += 0.05
        
        return penalty
    
    def calculate_current_performance_score(self, performance_data: List[Dict]) -> float:
        try:
            if not performance_data:
                return 0.0
            
            recent_scores = [data['performance_score'] for data in performance_data[-5:]]
            return np.mean(recent_scores)
            
        except Exception as e:
            logging.error(f"Current performance calculation error: {e}")
            return 0.0
    
    async def apply_parameter_updates(self, new_parameters: Dict, optimization_method: str):
        try:
            parameter_changes = []
            
            for param_name, new_value in new_parameters.items():
                if param_name in self.current_parameters:
                    old_value = self.current_parameters[param_name]
                    
                    if abs(new_value - old_value) > 1e-6:
                        parameter_changes.append({
                            'name': param_name,
                            'old_value': old_value,
                            'new_value': new_value,
                            'change_percent': ((new_value - old_value) / old_value) * 100
                        })
                        
                        self.current_parameters[param_name] = new_value
                        
                        if param_name in self.parameter_gauges:
                            self.parameter_gauges[param_name].set(new_value)
            
            if hasattr(self.momentum_model, 'dynamic_threshold'):
                self.momentum_model.dynamic_threshold = new_parameters.get(
                    'prediction_confidence', self.momentum_model.dynamic_threshold
                )
            
            if self.risk_manager and hasattr(self.risk_manager, 'update_parameters'):
                risk_params = {
                    'position_size_multiplier': new_parameters.get('position_size_multiplier', 1.0),
                    'stop_loss_threshold': new_parameters.get('stop_loss_threshold', 0.02),
                    'take_profit_threshold': new_parameters.get('take_profit_threshold', 0.15)
                }
                self.risk_manager.update_parameters(risk_params)
            
            with open('settings.yaml', 'r') as f:
                settings = yaml.safe_load(f)
            
            for param_name, new_value in new_parameters.items():
                if param_name == 'momentum_threshold':
                    settings['trading']['momentum_threshold'] = new_value
                elif param_name == 'velocity_threshold':
                    settings['trading']['velocity_threshold'] = new_value
                elif param_name == 'decay_threshold':
                    settings['trading']['decay_threshold'] = new_value
                elif param_name == 'prediction_confidence':
                    settings['ml']['prediction_confidence'] = new_value
                elif param_name == 'stop_loss_threshold':
                    settings['trading']['stop_loss_threshold'] = new_value
                elif param_name == 'take_profit_threshold':
                    settings['trading']['take_profit_threshold'] = new_value
            
            with open('settings.yaml', 'w') as f:
                yaml.dump(settings, f, default_flow_style=False, indent=2)
            
            cursor = self.conn.cursor()
            for change in parameter_changes:
                cursor.execute('''
                    INSERT INTO parameter_evolution 
                    (timestamp, parameter_name, old_value, new_value, performance_impact)
                    VALUES (?, ?, ?, ?, ?)
                ''', (time.time(), change['name'], change['old_value'], 
                      change['new_value'], 0.0))
            
            self.conn.commit()
            
            logging.info(json.dumps({
                'event': 'parameters_updated',
                'optimization_method': optimization_method,
                'changes': parameter_changes,
                'timestamp': time.time()
            }))
            
        except Exception as e:
            logging.error(f"Parameter update failed: {e}")
    
    def log_performance_result(self, performance_metrics: Dict):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO optimization_runs 
                (timestamp, parameters, performance_score, sharpe_ratio, win_rate, total_trades, optimization_method)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                json.dumps(self.current_parameters),
                performance_metrics.get('performance_score', 0),
                performance_metrics.get('sharpe_ratio', 0),
                performance_metrics.get('win_rate', 0),
                performance_metrics.get('total_trades', 0),
                'live_trading'
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logging.error(f"Performance logging failed: {e}")
    
    def get_optimization_summary(self) -> Dict:
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM optimization_runs')
            total_optimizations = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT AVG(performance_score), AVG(sharpe_ratio), AVG(win_rate) 
                FROM optimization_runs 
                WHERE timestamp > ?
            ''', (time.time() - 86400,))
            
            daily_performance = cursor.fetchone()
            
            cursor.execute('''
                SELECT parameter_name, AVG(ABS(new_value - old_value)) as avg_change
                FROM parameter_evolution 
                WHERE timestamp > ?
                GROUP BY parameter_name
            ''', (time.time() - 86400,))
            
            parameter_volatility = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total_optimizations': total_optimizations,
                'daily_avg_performance': daily_performance[0] if daily_performance[0] else 0,
                'daily_avg_sharpe': daily_performance[1] if daily_performance[1] else 0,
                'daily_avg_win_rate': daily_performance[2] if daily_performance[2] else 0,
                'parameter_volatility': parameter_volatility,
                'current_parameters': self.current_parameters,
                'last_optimization': self.last_optimization
            }
            
        except Exception as e:
            logging.error(f"Optimization summary error: {e}")
            return {}
    
    async def auto_tune_model_ensemble(self):
        try:
            if hasattr(self.momentum_model, 'weights'):
                current_weights = self.momentum_model.weights.copy()
                
                performance_data = await self.collect_performance_data()
                if len(performance_data) < 5:
                    return
                
                best_performance = max(data['performance_score'] for data in performance_data[-5:])
                
                weight_perturbations = np.random.uniform(-0.1, 0.1, len(current_weights))
                new_weights = current_weights + weight_perturbations
                new_weights = np.clip(new_weights, 0.1, 0.9)
                new_weights = new_weights / np.sum(new_weights)
                
                self.momentum_model.weights = new_weights.tolist()
                
                await asyncio.sleep(300)
                
                recent_performance = await self.collect_performance_data()
                if recent_performance:
                    new_performance = recent_performance[-1]['performance_score']
                    
                    if new_performance > best_performance:
                        logging.info("Ensemble weights improved, keeping changes")
                    else:
                        self.momentum_model.weights = current_weights
                        logging.info("Ensemble weights reverted due to poor performance")
                
        except Exception as e:
            logging.error(f"Auto-tuning error: {e}")
    
    def __del__(self):
        try:
            self.conn.close()
        except:
            pass