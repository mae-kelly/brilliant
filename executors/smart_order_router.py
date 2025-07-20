
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

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ExecutionStrategy(Enum):
    MARKET = "market"
    TWAP = "twap" 
    VWAP = "vwap"
    ICEBERG = "iceberg"
    STEALTH = "stealth"

@dataclass
class OrderSlice:
    slice_id: str
    amount: float
    target_price: float
    max_slippage: float
    execution_time: float
    priority: int

@dataclass
class RouteResult:
    dex: str
    estimated_output: float
    price_impact: float
    gas_cost: float
    execution_time: float
    confidence: float

class SmartOrderRouter:
    def __init__(self):
        self.dex_routers = {
            'ethereum': {
                'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
                'curve': '0x8301AE4fc9c624d1D396cbDAa1ed877821D7C511'
            },
            'arbitrum': {
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'camelot': '0xc873fEcbd354f5A56E00E710B90EF4201db2448d',
                'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
            },
            'polygon': {
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'quickswap': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
                'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
            }
        }
        
        self.liquidity_cache = {}
        self.price_impact_models = {}
        self.execution_analytics = {}

    async def find_optimal_route(self, token_in: str, token_out: str, amount_in: float,
                                chain: str, strategy: ExecutionStrategy) -> List[RouteResult]:
        
        available_dexes = self.dex_routers.get(chain, {})
        route_results = []
        
        for dex_name, router_address in available_dexes.items():
            result = await self.evaluate_dex_route(
                dex_name, router_address, token_in, token_out, amount_in, chain
            )
            
            if result:
                route_results.append(result)
        
        return sorted(route_results, key=lambda x: x.estimated_output, reverse=True)

    async def evaluate_dex_route(self, dex_name: str, router_address: str,
                                token_in: str, token_out: str, amount_in: float,
                                chain: str) -> Optional[RouteResult]:
        try:
            liquidity = await self.get_dex_liquidity(dex_name, token_in, token_out, chain)
            
            if liquidity < amount_in * 10:
                return None
            
            estimated_output = await self.simulate_swap(
                dex_name, token_in, token_out, amount_in, chain
            )
            
            price_impact = self.calculate_price_impact(amount_in, liquidity)
            gas_cost = await self.estimate_gas_cost(dex_name, chain)
            execution_time = self.estimate_execution_time(dex_name, chain)
            confidence = self.calculate_route_confidence(dex_name, liquidity, price_impact)
            
            return RouteResult(
                dex=dex_name,
                estimated_output=estimated_output,
                price_impact=price_impact,
                gas_cost=gas_cost,
                execution_time=execution_time,
                confidence=confidence
            )
            
        except Exception as e:
            return None

    async def get_dex_liquidity(self, dex_name: str, token_in: str, token_out: str, chain: str) -> float:
        cache_key = f"{chain}_{dex_name}_{token_in}_{token_out}"
        
        if cache_key in self.liquidity_cache:
            cached = self.liquidity_cache[cache_key]
            if time.time() - cached['timestamp'] < 60:
                return cached['liquidity']
        
        simulated_liquidity = np.random.uniform(10000, 1000000)
        
        self.liquidity_cache[cache_key] = {
            'liquidity': simulated_liquidity,
            'timestamp': time.time()
        }
        
        return simulated_liquidity

    async def simulate_swap(self, dex_name: str, token_in: str, token_out: str,
                           amount_in: float, chain: str) -> float:
        
        base_rate = np.random.uniform(0.8, 1.2)
        
        dex_multipliers = {
            'uniswap_v3': 1.02,
            'uniswap_v2': 0.98,
            'sushiswap': 0.99,
            'camelot': 1.01,
            'quickswap': 0.97,
            'curve': 1.03
        }
        
        multiplier = dex_multipliers.get(dex_name, 1.0)
        estimated_output = amount_in * base_rate * multiplier
        
        return estimated_output

    def calculate_price_impact(self, amount_in: float, liquidity: float) -> float:
        if liquidity <= 0:
            return 1.0
        
        impact = (amount_in / liquidity) ** 0.5 * 0.1
        return min(impact, 0.5)

    async def estimate_gas_cost(self, dex_name: str, chain: str) -> float:
        base_gas_costs = {
            'ethereum': 150000,
            'arbitrum': 1200000,
            'polygon': 120000
        }
        
        dex_gas_multipliers = {
            'uniswap_v3': 1.2,
            'uniswap_v2': 1.0,
            'sushiswap': 1.1,
            'camelot': 1.15,
            'quickswap': 0.9,
            'curve': 1.3
        }
        
        base_gas = base_gas_costs.get(chain, 150000)
        multiplier = dex_gas_multipliers.get(dex_name, 1.0)
        
        return base_gas * multiplier

    def estimate_execution_time(self, dex_name: str, chain: str) -> float:
        base_times = {
            'ethereum': 15.0,
            'arbitrum': 2.0,
            'polygon': 3.0
        }
        
        dex_time_multipliers = {
            'uniswap_v3': 1.1,
            'uniswap_v2': 1.0,
            'sushiswap': 1.0,
            'camelot': 1.05,
            'quickswap': 0.95,
            'curve': 1.2
        }
        
        base_time = base_times.get(chain, 15.0)
        multiplier = dex_time_multipliers.get(dex_name, 1.0)
        
        return base_time * multiplier

    def calculate_route_confidence(self, dex_name: str, liquidity: float, price_impact: float) -> float:
        liquidity_score = min(liquidity / 100000, 1.0)
        impact_score = max(0, 1.0 - price_impact * 5)
        
        dex_reliability = {
            'uniswap_v3': 0.95,
            'uniswap_v2': 0.90,
            'sushiswap': 0.85,
            'camelot': 0.80,
            'quickswap': 0.75,
            'curve': 0.88
        }
        
        reliability = dex_reliability.get(dex_name, 0.70)
        
        confidence = (liquidity_score + impact_score + reliability) / 3.0
        return confidence

    async def execute_smart_order(self, token_in: str, token_out: str, total_amount: float,
                                 chain: str, strategy: ExecutionStrategy,
                                 max_slippage: float = 0.03) -> List[Dict]:
        
        if strategy == ExecutionStrategy.MARKET:
            return await self.execute_market_order(token_in, token_out, total_amount, chain)
        
        elif strategy == ExecutionStrategy.TWAP:
            return await self.execute_twap_order(token_in, token_out, total_amount, chain, 300)
        
        elif strategy == ExecutionStrategy.ICEBERG:
            return await self.execute_iceberg_order(token_in, token_out, total_amount, chain)
        
        elif strategy == ExecutionStrategy.STEALTH:
            return await self.execute_stealth_order(token_in, token_out, total_amount, chain)
        
        else:
            return await self.execute_market_order(token_in, token_out, total_amount, chain)

    async def execute_market_order(self, token_in: str, token_out: str, amount: float, chain: str) -> List[Dict]:
        routes = await self.find_optimal_route(token_in, token_out, amount, chain, ExecutionStrategy.MARKET)
        
        if not routes:
            return []
        
        best_route = routes[0]
        
        execution_result = {
            'dex': best_route.dex,
            'amount_in': amount,
            'amount_out': best_route.estimated_output,
            'price_impact': best_route.price_impact,
            'gas_cost': best_route.gas_cost,
            'execution_time': best_route.execution_time,
            'success': True,
            'tx_hash': f"0x{'a' * 64}"
        }
        
        return [execution_result]

    async def execute_twap_order(self, token_in: str, token_out: str, total_amount: float,
                                chain: str, duration_seconds: int) -> List[Dict]:
        
        num_slices = min(10, max(3, duration_seconds // 30))
        slice_amount = total_amount / num_slices
        interval = duration_seconds / num_slices
        
        execution_results = []
        
        for i in range(num_slices):
            await asyncio.sleep(interval if i > 0 else 0)
            
            routes = await self.find_optimal_route(token_in, token_out, slice_amount, chain, ExecutionStrategy.TWAP)
            
            if routes:
                best_route = routes[0]
                
                result = {
                    'slice': i + 1,
                    'dex': best_route.dex,
                    'amount_in': slice_amount,
                    'amount_out': best_route.estimated_output,
                    'price_impact': best_route.price_impact,
                    'gas_cost': best_route.gas_cost,
                    'success': True,
                    'tx_hash': f"0x{hash(str(i) + token_in) % (16**64):064x}"
                }
                
                execution_results.append(result)
        
        return execution_results

    async def execute_iceberg_order(self, token_in: str, token_out: str, total_amount: float, chain: str) -> List[Dict]:
        slice_sizes = [0.15, 0.20, 0.25, 0.25, 0.15]
        execution_results = []
        
        for i, size_ratio in enumerate(slice_sizes):
            slice_amount = total_amount * size_ratio
            
            routes = await self.find_optimal_route(token_in, token_out, slice_amount, chain, ExecutionStrategy.ICEBERG)
            
            if routes:
                selected_routes = routes[:2]
                
                for j, route in enumerate(selected_routes):
                    split_amount = slice_amount / len(selected_routes)
                    
                    result = {
                        'slice': f"{i+1}-{j+1}",
                        'dex': route.dex,
                        'amount_in': split_amount,
                        'amount_out': route.estimated_output * (split_amount / slice_amount),
                        'price_impact': route.price_impact,
                        'gas_cost': route.gas_cost,
                        'success': True,
                        'tx_hash': f"0x{hash(str(i) + str(j) + token_in) % (16**64):064x}"
                    }
                    
                    execution_results.append(result)
            
            await asyncio.sleep(np.random.uniform(10, 30))
        
        return execution_results

    async def execute_stealth_order(self, token_in: str, token_out: str, total_amount: float, chain: str) -> List[Dict]:
        num_random_slices = np.random.randint(8, 15)
        slice_amounts = np.random.dirichlet(np.ones(num_random_slices)) * total_amount
        
        execution_results = []
        
        for i, slice_amount in enumerate(slice_amounts):
            wait_time = np.random.exponential(20)
            await asyncio.sleep(wait_time if i > 0 else 0)
            
            routes = await self.find_optimal_route(token_in, token_out, slice_amount, chain, ExecutionStrategy.STEALTH)
            
            if routes:
                selected_dexes = np.random.choice(len(routes), size=min(2, len(routes)), replace=False)
                
                for dex_idx in selected_dexes:
                    route = routes[dex_idx]
                    partial_amount = slice_amount / len(selected_dexes)
                    
                    result = {
                        'slice': f"stealth-{i+1}",
                        'dex': route.dex,
                        'amount_in': partial_amount,
                        'amount_out': route.estimated_output * (partial_amount / slice_amount),
                        'price_impact': route.price_impact,
                        'gas_cost': route.gas_cost,
                        'success': True,
                        'tx_hash': f"0x{hash(str(i) + route.dex + token_in) % (16**64):064x}"
                    }
                    
                    execution_results.append(result)
        
        return execution_results

    def get_execution_analytics(self) -> Dict:
        total_executions = len(self.execution_analytics)
        
        if total_executions == 0:
            return {
                'total_executions': 0,
                'avg_price_impact': 0.0,
                'avg_gas_cost': 0.0,
                'success_rate': 0.0,
                'preferred_dexes': {}
            }
        
        price_impacts = [exec_data['price_impact'] for exec_data in self.execution_analytics.values()]
        gas_costs = [exec_data['gas_cost'] for exec_data in self.execution_analytics.values()]
        successes = [exec_data['success'] for exec_data in self.execution_analytics.values()]
        
        dex_counts = {}
        for exec_data in self.execution_analytics.values():
            dex = exec_data['dex']
            dex_counts[dex] = dex_counts.get(dex, 0) + 1
        
        return {
            'total_executions': total_executions,
            'avg_price_impact': np.mean(price_impacts),
            'avg_gas_cost': np.mean(gas_costs),
            'success_rate': np.mean(successes),
            'preferred_dexes': dex_counts
        }

smart_router = SmartOrderRouter()
