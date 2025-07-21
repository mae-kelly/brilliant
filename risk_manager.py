import numpy as np
import pandas as pd
from scipy.stats import t
import logging
from prometheus_client import Gauge
import json
import yaml
import time

from error_handler import retry_with_backoff, log_performance, CircuitBreaker, safe_execute
from error_handler import TradingSystemError, NetworkError, ModelInferenceError

class RiskManager:
    def __init__(self):
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)
        self.max_position_size = self.settings['risk']['max_position_size']
        self.min_position_size = self.settings['risk']['min_position_size']
        self.base_position_size = self.settings['risk']['base_position_size']
        self.max_gas_budget = self.settings['risk']['max_gas_budget']
        self.max_portfolio_exposure = self.settings['risk']['max_portfolio_exposure']
        self.confidence_level = self.settings['risk']['confidence_level']
        
        self.var_gauge = Gauge('value_at_risk', 'Value at Risk at 95% confidence', ['chain'])
        self.es_gauge = Gauge('expected_shortfall', 'Expected Shortfall at 95% confidence', ['chain'])
        self.volatility_gauge = Gauge('volatility', 'Annualized volatility', ['chain'])
        self.exposure_gauge = Gauge('portfolio_exposure', 'Total portfolio exposure', ['chain'])
        self.kelly_gauge = Gauge('kelly_fraction', 'Kelly Criterion optimal fraction', ['chain'])
        self.sharpe_gauge = Gauge('sharpe_ratio', 'Current Sharpe ratio', ['chain'])
        
        self.position_history = {}
        self.performance_metrics = {}

    def calculate_position_size(self, features, chain):
        try:
            if features.empty or 'returns' not in features.columns:
                logging.warning(json.dumps({
                    'event': 'insufficient_features_for_position_sizing',
                    'chain': chain
                }))
                return self.min_position_size
            
            returns = features['returns'].dropna()
            if len(returns) < 10:
                return self.min_position_size
            
            volatility = self.calculate_volatility(returns)
            var = self.calculate_var(returns)
            es = self.calculate_expected_shortfall(returns)
            kelly_fraction = self.calculate_kelly_fraction(returns)
            
            self.volatility_gauge.labels(chain=chain).set(volatility)
            self.var_gauge.labels(chain=chain).set(var)
            self.es_gauge.labels(chain=chain).set(es)
            self.kelly_gauge.labels(chain=chain).set(kelly_fraction)
            
            momentum_factor = self.calculate_momentum_factor(features)
            volume_factor = self.calculate_volume_factor(features)
            volatility_adjustment = self.calculate_volatility_adjustment(volatility)
            
            risk_adjusted_size = (self.base_position_size * 
                                momentum_factor * 
                                volume_factor * 
                                volatility_adjustment * 
                                kelly_fraction)
            
            risk_constrained_size = min(risk_adjusted_size, 
                                      self.base_position_size / (1 + volatility + abs(es)))
            
            final_size = max(self.min_position_size, 
                           min(risk_constrained_size, self.max_position_size))
            
            logging.info(json.dumps({
                'event': 'position_size_calculated',
                'chain': chain,
                'base_size': self.base_position_size,
                'momentum_factor': momentum_factor,
                'volume_factor': volume_factor,
                'volatility_adjustment': volatility_adjustment,
                'kelly_fraction': kelly_fraction,
                'final_size': final_size,
                'var': var,
                'expected_shortfall': es
            }))
            
            return final_size
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'position_size_error',
                'chain': chain,
                'error': str(e)
            }))
            return self.min_position_size

    def calculate_volatility(self, returns):
        return returns.std() * np.sqrt(252)

    def calculate_var(self, returns, confidence_level=None, df=3):
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        returns = returns.dropna()
        if len(returns) < 5:
            return 0.05
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        var_parametric = -t.ppf(1 - confidence_level, df=df, loc=mean_return, scale=std_return)
        
        var_historical = -np.percentile(returns, (1 - confidence_level) * 100)
        
        var_combined = (var_parametric + var_historical) / 2
        
        return max(var_combined, 0.001)

    def calculate_expected_shortfall(self, returns, confidence_level=None):
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        returns = returns.dropna()
        if len(returns) < 5:
            return 0.05
        
        var = self.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            return var * 1.5
        
        es = -tail_returns.mean()
        return max(es, var)

    def calculate_kelly_fraction(self, returns):
        try:
            returns = returns.dropna()
            if len(returns) < 10:
                return 0.1
            
            mean_return = returns.mean()
            variance = returns.var()
            
            if variance == 0 or mean_return <= 0:
                return 0.05
            
            kelly_fraction = mean_return / variance
            
            kelly_capped = max(0.01, min(kelly_fraction, 0.25))
            
            return kelly_capped
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'kelly_calculation_error',
                'error': str(e)
            }))
            return 0.05

    def calculate_momentum_factor(self, features):
        try:
            if 'momentum' not in features.columns:
                return 1.0
            
            momentum = features['momentum'].iloc[-1] if not features['momentum'].empty else 0
            momentum_strength = features.get('momentum_strength', pd.Series([0])).iloc[-1]
            
            base_factor = 1.0
            
            if momentum > 0.1:
                base_factor = 1.5
            elif momentum > 0.05:
                base_factor = 1.2
            elif momentum < -0.05:
                base_factor = 0.5
            
            strength_multiplier = 1 + min(momentum_strength * 0.5, 0.5)
            
            return min(base_factor * strength_multiplier, 2.0)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'momentum_factor_error',
                'error': str(e)
            }))
            return 1.0

    def calculate_volume_factor(self, features):
        try:
            if 'swap_volume' not in features.columns and 'volume_ma' not in features.columns:
                return 1.0
            
            volume_col = 'swap_volume' if 'swap_volume' in features.columns else 'volume_ma'
            current_volume = features[volume_col].iloc[-1] if not features[volume_col].empty else 1000
            avg_volume = features[volume_col].mean()
            
            if avg_volume == 0:
                return 1.0
            
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio > 5:
                return 1.8
            elif volume_ratio > 3:
                return 1.5
            elif volume_ratio > 2:
                return 1.2
            elif volume_ratio < 0.5:
                return 0.7
            else:
                return 1.0
                
        except Exception as e:
            logging.error(json.dumps({
                'event': 'volume_factor_error',
                'error': str(e)
            }))
            return 1.0

    def calculate_volatility_adjustment(self, volatility):
        try:
            target_volatility = 0.3
            
            if volatility == 0:
                return 1.0
            
            volatility_ratio = target_volatility / volatility
            
            adjustment = min(max(volatility_ratio, 0.2), 2.0)
            
            return adjustment
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'volatility_adjustment_error',
                'error': str(e)
            }))
            return 1.0

    def check_portfolio_exposure(self, chain, position_size):
        try:
            current_exposure = self.exposure_gauge.labels(chain=chain)._value._value
            new_exposure = current_exposure + position_size
            
            chain_limit = self.max_portfolio_exposure / 3
            
            if new_exposure > chain_limit:
                logging.warning(json.dumps({
                    'event': 'chain_exposure_limit_exceeded',
                    'chain': chain,
                    'current_exposure': current_exposure,
                    'new_exposure': new_exposure,
                    'limit': chain_limit
                }))
                return False
            
            total_exposure = sum(self.exposure_gauge.labels(chain=c)._value._value 
                               for c in ['arbitrum', 'polygon', 'optimism'])
            total_new_exposure = total_exposure + position_size
            
            if total_new_exposure > self.max_portfolio_exposure:
                logging.warning(json.dumps({
                    'event': 'total_exposure_limit_exceeded',
                    'total_exposure': total_exposure,
                    'new_total_exposure': total_new_exposure,
                    'limit': self.max_portfolio_exposure
                }))
                return False
            
            self.exposure_gauge.labels(chain=chain).set(new_exposure)
            return True
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'exposure_check_error',
                'chain': chain,
                'error': str(e)
            }))
            return False

    def update_exposure(self, chain, position_change):
        try:
            current_exposure = self.exposure_gauge.labels(chain=chain)._value._value
            new_exposure = max(0, current_exposure + position_change)
            self.exposure_gauge.labels(chain=chain).set(new_exposure)
            
            logging.info(json.dumps({
                'event': 'exposure_updated',
                'chain': chain,
                'position_change': position_change,
                'new_exposure': new_exposure
            }))
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'exposure_update_error',
                'chain': chain,
                'error': str(e)
            }))

    def check_gas_budget(self, chain, estimated_gas_cost):
        try:
            from trade_executor import TradeExecutor
            
            current_cost = 0
            try:
                trade_executor = TradeExecutor(self.chains if hasattr(self, 'chains') else {})
                current_cost = trade_executor.cost_gauge.labels(chain=chain)._value._value
            except:
                current_cost = 0
            
            projected_cost = current_cost + estimated_gas_cost
            
            if projected_cost > self.max_gas_budget:
                logging.warning(json.dumps({
                    'event': 'gas_budget_exceeded',
                    'chain': chain,
                    'current_cost': current_cost,
                    'estimated_cost': estimated_gas_cost,
                    'projected_total': projected_cost,
                    'budget': self.max_gas_budget
                }))
                return False
            
            return True
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'gas_budget_check_error',
                'chain': chain,
                'error': str(e)
            }))
            return True

    def calculate_correlation_risk(self, new_position, existing_positions):
        try:
            if not existing_positions:
                return 0.0
            
            correlation_scores = []
            
            for existing_pos in existing_positions:
                token_similarity = self.calculate_token_similarity(
                    new_position.get('token_address', ''),
                    existing_pos.get('token_address', '')
                )
                
                chain_correlation = 1.0 if new_position.get('chain') == existing_pos.get('chain') else 0.3
                
                time_correlation = self.calculate_time_correlation(
                    new_position.get('timestamp', time.time()),
                    existing_pos.get('timestamp', time.time())
                )
                
                total_correlation = (token_similarity * 0.4 + 
                                   chain_correlation * 0.4 + 
                                   time_correlation * 0.2)
                
                correlation_scores.append(total_correlation)
            
            max_correlation = max(correlation_scores) if correlation_scores else 0
            
            return min(max_correlation, 1.0)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'correlation_risk_error',
                'error': str(e)
            }))
            return 0.5

    def calculate_token_similarity(self, token1, token2):
        try:
            if token1.lower() == token2.lower():
                return 1.0
            
            symbol1 = token1[-6:]
            symbol2 = token2[-6:]
            
            if symbol1 == symbol2:
                return 0.8
            
            return 0.2
            
        except:
            return 0.2

    def calculate_time_correlation(self, time1, time2):
        try:
            time_diff = abs(time1 - time2)
            
            if time_diff < 300:  # 5 minutes
                return 1.0
            elif time_diff < 1800:  # 30 minutes
                return 0.7
            elif time_diff < 3600:  # 1 hour
                return 0.4
            else:
                return 0.1
                
        except:
            return 0.5

    def calculate_sharpe_ratio(self, returns):
        try:
            returns = returns.dropna()
            if len(returns) < 10:
                return 0.0
            
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return == 0:
                return 0.0
            
            sharpe = (mean_return / std_return) * np.sqrt(252)
            
            return sharpe
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'sharpe_calculation_error',
                'error': str(e)
            }))
            return 0.0

    def update_performance_metrics(self, chain, trade_result):
        try:
            if chain not in self.performance_metrics:
                self.performance_metrics[chain] = {
                    'returns': [],
                    'trade_count': 0,
                    'win_count': 0,
                    'total_pnl': 0
                }
            
            metrics = self.performance_metrics[chain]
            
            pnl = trade_result.get('pnl', 0)
            metrics['returns'].append(pnl)
            metrics['trade_count'] += 1
            metrics['total_pnl'] += pnl
            
            if pnl > 0:
                metrics['win_count'] += 1
            
            if len(metrics['returns']) > 100:
                metrics['returns'] = metrics['returns'][-100:]
            
            if len(metrics['returns']) >= 10:
                returns_series = pd.Series(metrics['returns'])
                sharpe = self.calculate_sharpe_ratio(returns_series)
                self.sharpe_gauge.labels(chain=chain).set(sharpe)
            
            logging.info(json.dumps({
                'event': 'performance_metrics_updated',
                'chain': chain,
                'trade_count': metrics['trade_count'],
                'win_rate': metrics['win_count'] / metrics['trade_count'],
                'total_pnl': metrics['total_pnl']
            }))
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'performance_update_error',
                'chain': chain,
                'error': str(e)
            }))

    def get_risk_summary(self, chain):
        try:
            current_exposure = self.exposure_gauge.labels(chain=chain)._value._value
            current_var = self.var_gauge.labels(chain=chain)._value._value
            current_volatility = self.volatility_gauge.labels(chain=chain)._value._value
            
            exposure_utilization = current_exposure / (self.max_portfolio_exposure / 3)
            
            risk_level = 'LOW'
            if exposure_utilization > 0.8 or current_var > 0.1:
                risk_level = 'HIGH'
            elif exposure_utilization > 0.5 or current_var > 0.05:
                risk_level = 'MEDIUM'
            
            return {
                'chain': chain,
                'exposure': current_exposure,
                'exposure_utilization': exposure_utilization,
                'var': current_var,
                'volatility': current_volatility,
                'risk_level': risk_level,
                'available_capacity': max(0, (self.max_portfolio_exposure / 3) - current_exposure)
            }
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'risk_summary_error',
                'chain': chain,
                'error': str(e)
            }))
            return {
                'chain': chain,
                'risk_level': 'UNKNOWN',
                'available_capacity': 0
            }