import asyncio
import pandas as pd
import numpy as np
import logging
import json
import time
import sqlite3
from web3 import Web3
from prometheus_client import Gauge, Counter
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

class FeedbackLoop:
    def __init__(self, momentum_model):
        self.momentum_model = momentum_model
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)
        self.conn = sqlite3.connect('feedback_cache.db')
        self.init_database()
        self.performance_gauge = Gauge('model_performance', 'Model performance metrics', ['metric'])
        self.trade_counter = Counter('feedback_trades_total', 'Total trades logged', ['outcome'])
        self.optimization_gauge = Gauge('optimization_score', 'Current optimization score')
        self.threshold_gauge = Gauge('dynamic_threshold', 'Current dynamic threshold')
        
    def init_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chain TEXT,
                token_address TEXT,
                tx_hash TEXT,
                entry_price REAL,
                exit_price REAL,
                entry_score REAL,
                exit_score REAL,
                position_size REAL,
                pnl REAL,
                holding_time INTEGER,
                timestamp INTEGER,
                outcome TEXT,
                features TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                avg_return REAL,
                total_trades INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                parameter_name TEXT,
                old_value REAL,
                new_value REAL,
                performance_impact REAL
            )
        ''')
        
        self.conn.commit()

    async def log_trade(self, chain, token_address, tx_hash, momentum_score, position_size, features=None):
        try:
            cursor = self.conn.cursor()
            
            trade_data = {
                'chain': chain,
                'token_address': token_address,
                'tx_hash': tx_hash.hex() if tx_hash else None,
                'entry_score': momentum_score,
                'position_size': position_size,
                'timestamp': int(time.time()),
                'features': pickle.dumps(features) if features is not None else None
            }
            
            cursor.execute('''
                INSERT INTO trades (chain, token_address, tx_hash, entry_score, position_size, timestamp, features)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (trade_data['chain'], trade_data['token_address'], trade_data['tx_hash'],
                  trade_data['entry_score'], trade_data['position_size'], 
                  trade_data['timestamp'], trade_data['features']))
            
            self.conn.commit()
            
            self.trade_counter.labels(outcome='logged').inc()
            
            logging.info(json.dumps({
                'event': 'trade_logged',
                'chain': chain,
                'token': token_address,
                'score': momentum_score
            }))
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'trade_logging_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))

    async def update_trade_outcome(self, tx_hash, exit_price, exit_score, pnl, holding_time):
        try:
            cursor = self.conn.cursor()
            
            outcome = 'win' if pnl > 0 else 'loss'
            
            cursor.execute('''
                UPDATE trades 
                SET exit_price = ?, exit_score = ?, pnl = ?, holding_time = ?, outcome = ?
                WHERE tx_hash = ?
            ''', (exit_price, exit_score, pnl, holding_time, outcome, tx_hash.hex()))
            
            self.conn.commit()
            
            self.trade_counter.labels(outcome=outcome).inc()
            
            await self.analyze_trade_performance(tx_hash)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'trade_update_error',
                'tx_hash': tx_hash.hex() if tx_hash else 'unknown',
                'error': str(e)
            }))

    async def analyze_trade_performance(self, tx_hash):
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM trades WHERE tx_hash = ?', (tx_hash.hex(),))
            trade = cursor.fetchone()
            
            if not trade:
                return
            
            performance_data = {
                'entry_score': trade[6],
                'exit_score': trade[7] if trade[7] else 0,
                'pnl': trade[9] if trade[9] else 0,
                'holding_time': trade[10] if trade[10] else 0,
                'outcome': trade[12]
            }
            
            await self.update_model_feedback(performance_data, trade[13])
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'performance_analysis_error',
                'error': str(e)
            }))

    async def update_model_feedback(self, performance_data, features_blob):
        try:
            if features_blob and performance_data['pnl'] is not None:
                features = pickle.loads(features_blob)
                label = 1 if performance_data['pnl'] > 0 else 0
                
                if hasattr(self.momentum_model, 'training_data'):
                    self.momentum_model.training_data.append(features.values if hasattr(features, 'values') else features)
                    self.momentum_model.training_labels.append(label)
                
                score_accuracy = 1 if (performance_data['entry_score'] > 0.7 and label == 1) or (performance_data['entry_score'] <= 0.7 and label == 0) else 0
                
                if hasattr(self.momentum_model, 'momentum_scores'):
                    self.momentum_model.momentum_scores.append(performance_data['entry_score'])
                
                logging.info(json.dumps({
                    'event': 'model_feedback_updated',
                    'score_accuracy': score_accuracy,
                    'pnl': performance_data['pnl'],
                    'entry_score': performance_data['entry_score']
                }))
                
        except Exception as e:
            logging.error(json.dumps({
                'event': 'model_feedback_error',
                'error': str(e)
            }))

    async def optimize_model(self):
        try:
            recent_trades = self.get_recent_trades(100)
            
            if len(recent_trades) < 50:
                return
            
            performance_metrics = self.calculate_performance_metrics(recent_trades)
            
            self.performance_gauge.labels(metric='accuracy').set(performance_metrics['accuracy'])
            self.performance_gauge.labels(metric='win_rate').set(performance_metrics['win_rate'])
            self.performance_gauge.labels(metric='sharpe_ratio').set(performance_metrics['sharpe_ratio'])
            
            await self.optimize_thresholds(recent_trades)
            await self.optimize_model_parameters(recent_trades)
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO model_performance 
                (timestamp, accuracy, precision_score, recall_score, sharpe_ratio, max_drawdown, win_rate, avg_return, total_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (int(time.time()), performance_metrics['accuracy'], performance_metrics['precision'],
                  performance_metrics['recall'], performance_metrics['sharpe_ratio'], 
                  performance_metrics['max_drawdown'], performance_metrics['win_rate'],
                  performance_metrics['avg_return'], len(recent_trades)))
            
            self.conn.commit()
            
            logging.info(json.dumps({
                'event': 'model_optimized',
                'metrics': performance_metrics
            }))
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'optimization_error',
                'error': str(e)
            }))

    def get_recent_trades(self, limit=100):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM trades 
            WHERE pnl IS NOT NULL 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()

    def calculate_performance_metrics(self, trades):
        if not trades:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.0, 'avg_return': 0.0
            }
        
        try:
            predictions = [1 if trade[6] > self.momentum_model.dynamic_threshold else 0 for trade in trades]
            actuals = [1 if trade[9] > 0 else 0 for trade in trades]
            returns = [trade[9] for trade in trades if trade[9] is not None]
            
            accuracy = accuracy_score(actuals, predictions)
            precision = precision_score(actuals, predictions, zero_division=0)
            recall = recall_score(actuals, predictions, zero_division=0)
            
            if returns:
                returns_series = pd.Series(returns)
                sharpe_ratio = returns_series.mean() / (returns_series.std() + 1e-10) * np.sqrt(252)
                cumulative_returns = returns_series.cumsum()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = cumulative_returns - rolling_max
                max_drawdown = drawdown.min()
                win_rate = sum(1 for r in returns if r > 0) / len(returns)
                avg_return = returns_series.mean()
            else:
                sharpe_ratio = 0.0
                max_drawdown = 0.0
                win_rate = 0.0
                avg_return = 0.0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_return': avg_return
            }
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'metrics_calculation_error',
                'error': str(e)
            }))
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.0, 'avg_return': 0.0
            }

    async def optimize_thresholds(self, trades):
        try:
            current_threshold = self.momentum_model.dynamic_threshold
            
            best_threshold = current_threshold
            best_performance = -float('inf')
            
            threshold_range = np.arange(0.5, 0.95, 0.05)
            
            for threshold in threshold_range:
                self.momentum_model.dynamic_threshold = threshold
                
                predictions = [1 if trade[6] > threshold else 0 for trade in trades]
                actuals = [1 if trade[9] > 0 else 0 for trade in trades]
                
                if len(set(predictions)) > 1 and len(set(actuals)) > 1:
                    accuracy = accuracy_score(actuals, predictions)
                    precision = precision_score(actuals, predictions, zero_division=0)
                    
                    combined_score = (accuracy * 0.6) + (precision * 0.4)
                    
                    if combined_score > best_performance:
                        best_performance = combined_score
                        best_threshold = threshold
            
            if best_threshold != current_threshold:
                self.log_optimization('dynamic_threshold', current_threshold, best_threshold, best_performance)
                self.momentum_model.dynamic_threshold = best_threshold
                
                logging.info(json.dumps({
                    'event': 'threshold_optimized',
                    'old_threshold': current_threshold,
                    'new_threshold': best_threshold,
                    'performance_improvement': best_performance
                }))
            
            self.threshold_gauge.set(self.momentum_model.dynamic_threshold)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'threshold_optimization_error',
                'error': str(e)
            }))

    async def optimize_model_parameters(self, trades):
        try:
            if not hasattr(self.momentum_model, 'weights') or len(trades) < 20:
                return
            
            current_weights = self.momentum_model.weights.copy()
            best_weights = current_weights.copy()
            best_score = self.evaluate_weight_performance(trades, current_weights)
            
            weight_adjustments = [-0.1, -0.05, 0.05, 0.1]
            
            for i in range(len(current_weights)):
                for adjustment in weight_adjustments:
                    test_weights = current_weights.copy()
                    test_weights[i] = max(0.1, min(0.9, test_weights[i] + adjustment))
                    
                    test_weights = [w / sum(test_weights) for w in test_weights]
                    
                    score = self.evaluate_weight_performance(trades, test_weights)
                    
                    if score > best_score:
                        best_score = score
                        best_weights = test_weights.copy()
            
            if best_weights != current_weights:
                self.momentum_model.weights = best_weights
                
                self.log_optimization('ensemble_weights', current_weights, best_weights, best_score)
                
                logging.info(json.dumps({
                    'event': 'weights_optimized',
                    'old_weights': current_weights,
                    'new_weights': best_weights,
                    'performance_improvement': best_score
                }))
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'parameter_optimization_error',
                'error': str(e)
            }))

    def evaluate_weight_performance(self, trades, weights):
        try:
            total_score = 0
            for trade in trades:
                if trade[9] is not None:
                    predicted_score = trade[6]
                    actual_return = trade[9]
                    
                    if predicted_score > self.momentum_model.dynamic_threshold and actual_return > 0:
                        total_score += 1
                    elif predicted_score <= self.momentum_model.dynamic_threshold and actual_return <= 0:
                        total_score += 0.5
            
            return total_score / len(trades) if trades else 0
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'weight_evaluation_error',
                'error': str(e)
            }))
            return 0

    def log_optimization(self, parameter_name, old_value, new_value, performance_impact):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO optimization_history 
                (timestamp, parameter_name, old_value, new_value, performance_impact)
                VALUES (?, ?, ?, ?, ?)
            ''', (int(time.time()), parameter_name, 
                  old_value if isinstance(old_value, (int, float)) else str(old_value),
                  new_value if isinstance(new_value, (int, float)) else str(new_value),
                  performance_impact))
            
            self.conn.commit()
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'optimization_logging_error',
                'error': str(e)
            }))

    async def adaptive_learning(self):
        try:
            recent_performance = self.get_recent_performance(20)
            
            if len(recent_performance) < 10:
                return
            
            performance_trend = self.calculate_performance_trend(recent_performance)
            
            if performance_trend < -0.1:
                await self.increase_exploration()
                logging.info(json.dumps({
                    'event': 'adaptive_learning',
                    'action': 'increase_exploration',
                    'trend': performance_trend
                }))
            elif performance_trend > 0.1:
                await self.increase_exploitation()
                logging.info(json.dumps({
                    'event': 'adaptive_learning',
                    'action': 'increase_exploitation',
                    'trend': performance_trend
                }))
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'adaptive_learning_error',
                'error': str(e)
            }))

    def get_recent_performance(self, limit):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT win_rate FROM model_performance 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        return [row[0] for row in cursor.fetchall()]

    def calculate_performance_trend(self, performance_data):
        if len(performance_data) < 2:
            return 0
        
        recent_avg = np.mean(performance_data[:len(performance_data)//2])
        older_avg = np.mean(performance_data[len(performance_data)//2:])
        
        return (recent_avg - older_avg) / (older_avg + 0.01)

    async def increase_exploration(self):
        current_threshold = self.momentum_model.dynamic_threshold
        self.momentum_model.dynamic_threshold = max(0.5, current_threshold - 0.05)
        
        self.optimization_gauge.set(-0.1)

    async def increase_exploitation(self):
        current_threshold = self.momentum_model.dynamic_threshold
        self.momentum_model.dynamic_threshold = min(0.9, current_threshold + 0.02)
        
        self.optimization_gauge.set(0.1)

    def get_optimization_summary(self):
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL')
            total_trades = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(win_rate), AVG(sharpe_ratio) FROM model_performance WHERE timestamp > ?', 
                          (int(time.time()) - 86400,))
            daily_performance = cursor.fetchone()
            
            cursor.execute('SELECT COUNT(*) FROM optimization_history WHERE timestamp > ?',
                          (int(time.time()) - 86400,))
            daily_optimizations = cursor.fetchone()[0]
            
            return {
                'total_trades': total_trades,
                'daily_win_rate': daily_performance[0] if daily_performance[0] else 0,
                'daily_sharpe': daily_performance[1] if daily_performance[1] else 0,
                'daily_optimizations': daily_optimizations,
                'current_threshold': self.momentum_model.dynamic_threshold
            }
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'optimization_summary_error',
                'error': str(e)
            }))
            return {}

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass