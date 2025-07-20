import asyncio
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config"))

from config.unified_config import global_config, is_trading_enabled
from executors.real_trade_executor import RealTradeExecutor

class ExecutionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"  
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TradeRequest:
    token_address: str
    chain: str
    side: str
    amount_usd: float
    max_slippage: float
    urgency: float
    deadline: float

@dataclass
class TradeResult:
    success: bool
    tx_hash: str
    executed_amount: float
    execution_price: float
    gas_cost: float
    slippage: float
    execution_time: float
    route_used: str

class ProductionExecutor:
    def __init__(self):
        self.real_executor = RealTradeExecutor()
        self.execution_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'avg_execution_time': 0.0,
            'avg_slippage': 0.0,
            'total_gas_used': 0,
            'total_volume': 0.0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        await self.real_executor.initialize()
        self.logger.info("Production executor initialized")

    async def execute_buy_trade(self, token_address: str, chain: str, amount_usd: float) -> Dict:
        start_time = time.time()
        
        try:
            if is_trading_enabled():
                result = await self.real_executor.execute_buy_trade(token_address, chain, amount_usd)
                
                trade_result = {
                    'success': result.success,
                    'tx_hash': result.tx_hash,
                    'executed_amount': result.executed_amount,
                    'execution_price': result.execution_price,
                    'gas_cost': result.gas_cost,
                    'slippage': result.slippage,
                    'execution_time': result.execution_time,
                    'route_used': result.route_used
                }
            else:
                trade_result = await self._simulate_trade(token_address, chain, amount_usd, 'buy')
            
            execution_time = time.time() - start_time
            self._update_stats(trade_result, execution_time)
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Buy trade failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'tx_hash': '',
                'executed_amount': 0.0,
                'execution_price': 0.0,
                'gas_cost': 0.001,
                'slippage': 0.0,
                'execution_time': time.time() - start_time,
                'route_used': 'Failed'
            }

    async def execute_sell_trade(self, token_address: str, chain: str, token_amount: float) -> Dict:
        start_time = time.time()
        
        try:
            if is_trading_enabled():
                result = await self.real_executor.execute_sell_trade(token_address, chain, token_amount)
                
                trade_result = {
                    'success': result.success,
                    'tx_hash': result.tx_hash,
                    'executed_amount': result.executed_amount,
                    'execution_price': result.execution_price,
                    'gas_cost': result.gas_cost,
                    'slippage': result.slippage,
                    'execution_time': result.execution_time,
                    'route_used': result.route_used
                }
            else:
                trade_result = await self._simulate_trade(token_address, chain, token_amount, 'sell')
            
            execution_time = time.time() - start_time
            self._update_stats(trade_result, execution_time)
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Sell trade failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'tx_hash': '',
                'executed_amount': 0.0,
                'execution_price': 0.0,
                'gas_cost': 0.001,
                'slippage': 0.0,
                'execution_time': time.time() - start_time,
                'route_used': 'Failed'
            }

    async def _simulate_trade(self, token_address: str, chain: str, amount: float, side: str) -> Dict:
        await asyncio.sleep(np.random.uniform(0.5, 2.0))
        
        success = np.random.random() > 0.05
        
        if success:
            executed_amount = amount * np.random.uniform(0.98, 1.0)
            slippage = np.random.uniform(0.001, 0.03)
            execution_price = (1 + slippage) if side == 'buy' else (1 - slippage)
            gas_cost = np.random.uniform(0.001, 0.01)
            tx_hash = f"0x{hash(str(time.time()) + token_address) % (16**64):064x}"
            
            return {
                'success': True,
                'tx_hash': tx_hash,
                'executed_amount': executed_amount,
                'execution_price': execution_price,
                'gas_cost': gas_cost,
                'slippage': slippage,
                'execution_time': time.time(),
                'route_used': f'Simulated {chain.title()}'
            }
        else:
            return {
                'success': False,
                'error': 'Simulated execution failure',
                'tx_hash': '',
                'executed_amount': 0.0,
                'execution_price': 0.0,
                'gas_cost': 0.001,
                'slippage': 0.0,
                'execution_time': time.time(),
                'route_used': 'Failed'
            }

    def _update_stats(self, result: Dict, total_time: float):
        self.execution_stats['total_trades'] += 1
        
        if result.get('success'):
            self.execution_stats['successful_trades'] += 1
            
            total_successful = self.execution_stats['successful_trades']
            current_avg_time = self.execution_stats['avg_execution_time']
            self.execution_stats['avg_execution_time'] = (
                (current_avg_time * (total_successful - 1) + total_time) / total_successful
            )
            
            slippage = result.get('slippage', 0)
            current_avg_slippage = self.execution_stats['avg_slippage']
            self.execution_stats['avg_slippage'] = (
                (current_avg_slippage * (total_successful - 1) + slippage) / total_successful
            )
            
            self.execution_stats['total_volume'] += result.get('executed_amount', 0)

    def get_execution_stats(self) -> Dict:
        total = self.execution_stats['total_trades']
        return {
            'total_executions': total,
            'success_rate': self.execution_stats['successful_trades'] / max(total, 1),
            'avg_execution_time': self.execution_stats['avg_execution_time'],
            'avg_slippage': self.execution_stats['avg_slippage'],
            'total_volume': self.execution_stats['total_volume'],
            'total_gas_used': self.execution_stats['total_gas_used']
        }

    async def shutdown(self):
        await self.real_executor.close()

real_executor = ProductionExecutor()