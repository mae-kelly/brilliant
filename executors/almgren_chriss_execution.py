import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from optimizer import get_dynamic_config, update_performance
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
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
from abc import ABC, abstractmethod

@dataclass
class ExecutionSchedule:
    time_intervals: List[float]
    trade_sizes: List[float]
    expected_cost: float
    variance: float

class MarketImpactModel:
    def __init__(self, permanent_alpha=0.6, temporary_beta=0.6, volatility=0.2):
        self.permanent_alpha = permanent_alpha
        self.temporary_beta = temporary_beta
        self.volatility = volatility
    
    def permanent_impact(self, volume: float, daily_volume: float) -> float:
        if daily_volume <= 0:
            return 0
        return self.permanent_alpha * (volume / daily_volume) ** 0.5
    
    def temporary_impact(self, volume: float, liquidity: float) -> float:
        if liquidity <= 0:
            return 0
        return self.temporary_beta * (volume / liquidity) ** 0.5
    
    def volatility_cost(self, time_remaining: float) -> float:
        return self.volatility * np.sqrt(time_remaining)

class AlmgrenChrissOptimizer:
    def __init__(self, market_impact_model: MarketImpactModel, risk_aversion: float = 1e-6):
        self.market_impact = market_impact_model
        self.risk_aversion = risk_aversion
    
    def optimize_schedule(self, total_quantity: float, time_horizon: float, 
                         daily_volume: float, liquidity: float, 
                         num_intervals: int = 10) -> ExecutionSchedule:
        
        dt = time_horizon / num_intervals
        time_intervals = [i * dt for i in range(num_intervals + 1)]
        
        def objective(trade_list):
            total_cost = 0
            total_variance = 0
            remaining_quantity = total_quantity
            
            for i, trade_size in enumerate(trade_list):
                if trade_size < 0:
                    trade_size = 0
                
                time_remaining = time_horizon - (i * dt)
                
                permanent_cost = self.market_impact.permanent_impact(trade_size, daily_volume)
                temporary_cost = self.market_impact.temporary_impact(trade_size, liquidity)
                volatility_cost = self.market_impact.volatility_cost(time_remaining)
                
                trade_cost = permanent_cost + temporary_cost
                trade_variance = (volatility_cost * remaining_quantity) ** 2
                
                total_cost += trade_cost * trade_size
                total_variance += trade_variance * dt
                
                remaining_quantity -= trade_size
            
            return total_cost + self.risk_aversion * total_variance
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: sum(x) - total_quantity}
        ]
        
        bounds = [(0, total_quantity) for _ in range(num_intervals)]
        
        initial_guess = [total_quantity / num_intervals] * num_intervals
        
        result = minimize(
            objective, 
            initial_guess, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            trade_sizes = result.x.tolist()
        else:
            trade_sizes = initial_guess
        
        expected_cost = result.fun if result.success else objective(initial_guess)
        variance = self.calculate_variance(trade_sizes, total_quantity, time_horizon, dt)
        
        return ExecutionSchedule(
            time_intervals=time_intervals[:-1],
            trade_sizes=trade_sizes,
            expected_cost=expected_cost,
            variance=variance
        )
    
    def calculate_variance(self, trade_sizes: List[float], total_quantity: float, 
                          time_horizon: float, dt: float) -> float:
        variance = 0
        remaining_quantity = total_quantity
        
        for i, trade_size in enumerate(trade_sizes):
            time_remaining = time_horizon - (i * dt)
            volatility_cost = self.market_impact.volatility_cost(time_remaining)
            variance += (volatility_cost * remaining_quantity) ** 2 * dt
            remaining_quantity -= trade_size
        
        return variance

class OptimalTWAPExecutor:
    def __init__(self, market_impact_model: MarketImpactModel):
        self.market_impact = market_impact_model
        self.optimizer = AlmgrenChrissOptimizer(market_impact_model)
        
    async def execute_optimal_schedule(self, token_address: str, chain: str, 
                                     total_quantity: float, time_horizon: float,
                                     market_data: Dict) -> List[Dict]:
        
        daily_volume = market_data.get('volume_24h', 100000)
        liquidity = market_data.get('liquidity_usd', 50000)
        
        schedule = self.optimizer.optimize_schedule(
            total_quantity=total_quantity,
            time_horizon=time_horizon,
            daily_volume=daily_volume,
            liquidity=liquidity,
            num_intervals=min(10, int(time_horizon / 30))
        )
        
        execution_results = []
        start_time = time.time()
        
        for i, (interval_time, trade_size) in enumerate(zip(schedule.time_intervals, schedule.trade_sizes)):
            if trade_size < 0.001:
                continue
            
            await asyncio.sleep(interval_time if i > 0 else 0)
            
            execution_result = await self.execute_slice(
                token_address, chain, trade_size, market_data
            )
            
            execution_results.append({
                'slice_number': i + 1,
                'planned_size': trade_size,
                'actual_size': execution_result.get('filled_amount', trade_size),
                'execution_price': execution_result.get('execution_price', 0),
                'timestamp': time.time(),
                'success': execution_result.get('success', False),
                'market_impact': self.estimate_market_impact(trade_size, market_data)
            })
        
        return execution_results
    
    async def execute_slice(self, token_address: str, chain: str, 
                           trade_size: float, market_data: Dict) -> Dict:
        
        await asyncio.sleep(0.1)
        
        slippage = np.random.uniform(0.001, 0.01)
        execution_price = market_data.get('price', 1.0) * (1 + slippage)
        
        return {
            'success': True,
            'filled_amount': trade_size * np.random.uniform(0.95, 1.0),
            'execution_price': execution_price,
            'gas_used': 150000,
            'tx_hash': f"0x{'a' * 64}"
        }
    
    def estimate_market_impact(self, trade_size: float, market_data: Dict) -> float:
        daily_volume = market_data.get('volume_24h', 100000)
        liquidity = market_data.get('liquidity_usd', 50000)
        
        permanent = self.market_impact.permanent_impact(trade_size, daily_volume)
        temporary = self.market_impact.temporary_impact(trade_size, liquidity)
        
        return permanent + temporary

class ParticipationRateModel:
    def __init__(self, base_rate=0.1, max_rate=0.3):
        self.base_rate = base_rate
        self.max_rate = max_rate
        
    def calculate_optimal_rate(self, urgency: float, market_volatility: float, 
                             spread: float) -> float:
        
        urgency_adjustment = urgency * 0.5
        volatility_adjustment = min(market_volatility * 2, 0.3)
        spread_adjustment = min(spread * 5, 0.2)
        
        optimal_rate = (self.base_rate + 
                       urgency_adjustment + 
                       volatility_adjustment - 
                       spread_adjustment)
        
        return max(0.05, min(optimal_rate, self.max_rate))

class AdaptiveParticipationTWAP:
    def __init__(self):
        self.participation_model = ParticipationRateModel()
        self.market_impact = MarketImpactModel()
        
    async def execute_adaptive_twap(self, token_address: str, chain: str,
                                  total_quantity: float, time_horizon: float,
                                  market_data: Dict) -> List[Dict]:
        
        execution_results = []
        remaining_quantity = total_quantity
        start_time = time.time()
        
        interval_duration = min(60, time_horizon / 10)
        num_intervals = int(time_horizon / interval_duration)
        
        for interval in range(num_intervals):
            if remaining_quantity <= 0.001:
                break
                
            current_time = time.time() - start_time
            urgency = current_time / time_horizon
            
            market_volatility = self.estimate_market_volatility(market_data)
            spread = self.estimate_bid_ask_spread(market_data)
            
            participation_rate = self.participation_model.calculate_optimal_rate(
                urgency, market_volatility, spread
            )
            
            interval_volume = market_data.get('volume_24h', 100000) / (24 * 60 / interval_duration)
            max_trade_size = interval_volume * participation_rate
            
            trade_size = min(remaining_quantity, max_trade_size)
            
            if trade_size > 0.001:
                execution_result = await self.execute_participation_slice(
                    token_address, chain, trade_size, market_data
                )
                
                execution_results.append({
                    'interval': interval + 1,
                    'participation_rate': participation_rate,
                    'trade_size': trade_size,
                    'remaining_quantity': remaining_quantity - trade_size,
                    'market_volatility': market_volatility,
                    'urgency': urgency,
                    'execution_result': execution_result
                })
                
                remaining_quantity -= trade_size
            
            await asyncio.sleep(interval_duration)
        
        return execution_results
    
    async def execute_participation_slice(self, token_address: str, chain: str,
                                        trade_size: float, market_data: Dict) -> Dict:
        
        await asyncio.sleep(0.05)
        
        participation_slippage = np.random.uniform(0.0005, 0.005)
        execution_price = market_data.get('price', 1.0) * (1 + participation_slippage)
        
        return {
            'success': True,
            'filled_amount': trade_size,
            'execution_price': execution_price,
            'slippage': participation_slippage,
            'timestamp': time.time()
        }
    
    def estimate_market_volatility(self, market_data: Dict) -> float:
        return market_data.get('volatility', 0.1)
    
    def estimate_bid_ask_spread(self, market_data: Dict) -> float:
        return market_data.get('spread', 0.005)

optimal_twap_executor = OptimalTWAPExecutor(MarketImpactModel())
adaptive_participation_twap = AdaptiveParticipationTWAP()