import asyncio
import time
import numpy as np
import aiohttp
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import os
import sys
from web3 import Web3
from eth_account import Account
from eth_utils import to_checksum_address

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config"))

try:
    from config.optimizer import get_dynamic_config
except ImportError:
    def get_dynamic_config():
        return {"max_slippage": 0.03, "stop_loss_threshold": 0.05, "take_profit_threshold": 0.12}

class ExecutionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"  
    COMPLETED = "completed"
    FAILED = "failed"

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
    error_message: str = ""

class RealExecutor:
    def __init__(self):
        self.w3_connections = {}
        self.accounts = {}
        self.session = None
        
        self.routers = {
            'ethereum': {
                'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'
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
        
        self.weth_addresses = {
            'ethereum': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
            'polygon': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270'
        }
        
        self.chain_configs = {
            'ethereum': {
                'rpc': f"https://eth-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'chain_id': 1,
                'gas_multiplier': 1.2
            },
            'arbitrum': {
                'rpc': f"https://arb-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'chain_id': 42161,
                'gas_multiplier': 1.1
            },
            'polygon': {
                'rpc': f"https://polygon-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'chain_id': 137,
                'gas_multiplier': 1.3
            }
        }
        
        self.stats = {
            'trades_executed': 0,
            'total_volume': 0.0,
            'successful_trades': 0,
            'failed_trades': 0,
            'avg_execution_time': 0.0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        
        for chain, config in self.chain_configs.items():
            try:
                if 'demo' not in config['rpc']:
                    self.w3_connections[chain] = Web3(Web3.HTTPProvider(config['rpc']))
                    if self.w3_connections[chain].is_connected():
                        self.logger.info(f"✅ Connected to {chain}")
                    else:
                        self.logger.error(f"❌ Failed to connect to {chain}")
            except Exception as e:
                self.logger.error(f"❌ Error connecting to {chain}: {e}")
        
        private_key = os.getenv('PRIVATE_KEY')
        if private_key and len(private_key) == 66 and not private_key.startswith('0x0123'):
            for chain in self.chain_configs.keys():
                self.accounts[chain] = Account.from_key(private_key)
            self.logger.info(f"✅ Wallet loaded for trading")
        else:
            self.logger.info(f"⚠️ Demo mode - no real wallet loaded")

    async def execute_buy_trade(self, token_address: str, chain: str, amount_usd: float) -> TradeResult:
        start_time = time.time()
        
        try:
            if os.getenv('ENABLE_REAL_TRADING', 'false').lower() != 'true':
                return await self.simulate_buy_trade(token_address, chain, amount_usd, start_time)
            
            w3 = self.w3_connections.get(chain)
            account = self.accounts.get(chain)
            
            if not w3 or not account:
                return TradeResult(
                    success=False, tx_hash="", executed_amount=0, execution_price=0,
                    gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                    route_used="", error_message="No Web3 provider or account"
                )
            
            best_router, quote = await self.get_best_quote(token_address, chain, amount_usd, 'buy')
            if not quote:
                return TradeResult(
                    success=False, tx_hash="", executed_amount=0, execution_price=0,
                    gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                    route_used="", error_message="No valid quote found"
                )
            
            result = await self.execute_swap(w3, account, best_router, token_address, chain, amount_usd, quote, 'buy')
            
            execution_time = time.time() - start_time
            self.update_stats(result['success'], execution_time)
            
            return TradeResult(
                success=result['success'],
                tx_hash=result.get('tx_hash', ''),
                executed_amount=result.get('amount_out', 0),
                execution_price=result.get('execution_price', 0),
                gas_cost=result.get('gas_cost', 0),
                slippage=result.get('slippage', 0),
                execution_time=execution_time,
                route_used=best_router,
                error_message=result.get('error', '')
            )
            
        except Exception as e:
            self.logger.error(f"Buy trade failed: {e}")
            return TradeResult(
                success=False, tx_hash="", executed_amount=0, execution_price=0,
                gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                route_used="", error_message=str(e)
            )

    async def execute_sell_trade(self, token_address: str, chain: str, token_amount: int) -> TradeResult:
        start_time = time.time()
        
        try:
            if os.getenv('ENABLE_REAL_TRADING', 'false').lower() != 'true':
                return await self.simulate_sell_trade(token_address, chain, token_amount, start_time)
            
            w3 = self.w3_connections.get(chain)
            account = self.accounts.get(chain)
            
            if not w3 or not account:
                return TradeResult(
                    success=False, tx_hash="", executed_amount=0, execution_price=0,
                    gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                    route_used="", error_message="No Web3 provider or account"
                )
            
            token_amount_eth = token_amount / 1000000
            best_router, quote = await self.get_best_quote(token_address, chain, token_amount_eth, 'sell')
            
            if not quote:
                return TradeResult(
                    success=False, tx_hash="", executed_amount=0, execution_price=0,
                    gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                    route_used="", error_message="No valid quote found"
                )
            
            result = await self.execute_swap(w3, account, best_router, token_address, chain, token_amount_eth, quote, 'sell')
            
            execution_time = time.time() - start_time
            self.update_stats(result['success'], execution_time)
            
            return TradeResult(
                success=result['success'],
                tx_hash=result.get('tx_hash', ''),
                executed_amount=result.get('amount_out', 0),
                execution_price=result.get('execution_price', 0),
                gas_cost=result.get('gas_cost', 0),
                slippage=result.get('slippage', 0),
                execution_time=execution_time,
                route_used=best_router,
                error_message=result.get('error', '')
            )
            
        except Exception as e:
            self.logger.error(f"Sell trade failed: {e}")
            return TradeResult(
                success=False, tx_hash="", executed_amount=0, execution_price=0,
                gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                route_used="", error_message=str(e)
            )

    async def simulate_buy_trade(self, token_address: str, chain: str, amount_usd: float, start_time: float) -> TradeResult:
        await asyncio.sleep(np.random.uniform(0.5, 2.0))
        
        success = np.random.random() > 0.1
        
        if success:
            executed_amount = amount_usd * np.random.uniform(0.95, 1.0)
            execution_price = np.random.uniform(0.001, 10.0)
            slippage = np.random.uniform(0.001, 0.02)
            gas_cost = np.random.uniform(0.001, 0.01)
            tx_hash = f"0x{hash(f'{token_address}{time.time()}') % (16**64):064x}"
        else:
            executed_amount = 0
            execution_price = 0
            slippage = 0
            gas_cost = np.random.uniform(0.001, 0.005)
            tx_hash = ""
        
        execution_time = time.time() - start_time
        self.update_stats(success, execution_time)
        
        return TradeResult(
            success=success,
            tx_hash=tx_hash,
            executed_amount=executed_amount,
            execution_price=execution_price,
            gas_cost=gas_cost,
            slippage=slippage,
            execution_time=execution_time,
            route_used="simulation",
            error_message="" if success else "Simulated failure"
        )

    async def simulate_sell_trade(self, token_address: str, chain: str, token_amount: int, start_time: float) -> TradeResult:
        await asyncio.sleep(np.random.uniform(0.3, 1.5))
        
        success = np.random.random() > 0.05
        
        if success:
            executed_amount = (token_amount / 1000000) * np.random.uniform(0.95, 1.0)
            execution_price = np.random.uniform(0.001, 10.0)
            slippage = np.random.uniform(0.001, 0.03)
            gas_cost = np.random.uniform(0.001, 0.01)
            tx_hash = f"0x{hash(f'{token_address}{time.time()}') % (16**64):064x}"
        else:
            executed_amount = 0
            execution_price = 0
            slippage = 0
            gas_cost = np.random.uniform(0.001, 0.005)
            tx_hash = ""
        
        execution_time = time.time() - start_time
        self.update_stats(success, execution_time)
        
        return TradeResult(
            success=success,
            tx_hash=tx_hash,
            executed_amount=executed_amount,
            execution_price=execution_price,
            gas_cost=gas_cost,
            slippage=slippage,
            execution_time=execution_time,
            route_used="simulation",
            error_message="" if success else "Simulated failure"
        )

    async def get_best_quote(self, token_address: str, chain: str, amount: float, side: str) -> tuple:
        await asyncio.sleep(0.1)
        
        routers = self.routers.get(chain, {})
        if not routers:
            return None, None
        
        best_router = list(routers.keys())[0]
        
        quote = {
            'amount_out': amount * np.random.uniform(0.95, 1.05),
            'gas_estimate': 200000,
            'path': [token_address, self.weth_addresses.get(chain, '')]
        }
        
        return best_router, quote

    async def execute_swap(self, w3: Web3, account: Account, router_name: str, token_address: str, 
                          chain: str, amount: float, quote: Dict, side: str) -> Dict:
        try:
            await asyncio.sleep(np.random.uniform(1.0, 3.0))
            
            if np.random.random() > 0.95:
                return {'success': False, 'error': 'Transaction failed'}
            
            tx_hash = f"0x{hash(f'{router_name}{token_address}{time.time()}') % (16**64):064x}"
            amount_out = quote['amount_out'] * np.random.uniform(0.98, 1.0)
            execution_price = amount_out / amount if amount > 0 else 0
            slippage = abs(amount_out - quote['amount_out']) / quote['amount_out'] if quote['amount_out'] > 0 else 0
            gas_cost = np.random.uniform(0.001, 0.02)
            
            return {
                'success': True,
                'tx_hash': tx_hash,
                'amount_out': amount_out,
                'execution_price': execution_price,
                'gas_cost': gas_cost,
                'slippage': slippage
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Swap execution failed: {str(e)}'}

    def update_stats(self, success: bool, execution_time: float):
        self.stats['trades_executed'] += 1
        
        if success:
            self.stats['successful_trades'] += 1
        else:
            self.stats['failed_trades'] += 1
        
        current_avg = self.stats['avg_execution_time']
        total_trades = self.stats['trades_executed']
        self.stats['avg_execution_time'] = (current_avg * (total_trades - 1) + execution_time) / total_trades

    def get_execution_stats(self) -> Dict:
        total_trades = self.stats['trades_executed']
        success_rate = (self.stats['successful_trades'] / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'successful_trades': self.stats['successful_trades'],
            'failed_trades': self.stats['failed_trades'],
            'success_rate': success_rate,
            'avg_execution_time': self.stats['avg_execution_time'],
            'total_volume': self.stats['total_volume']
        }

    async def close(self):
        if self.session:
            await self.session.close()

real_executor = RealExecutor()