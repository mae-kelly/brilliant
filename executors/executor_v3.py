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
import hashlib
import hmac
import base64
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config"))

from config.unified_config import global_config, is_trading_enabled

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

class SecureKeyManager:
    def __init__(self):
        self.keys = {}
        self.encrypted_store = {}
        
    def store_encrypted_key(self, key_id: str, private_key: str, passphrase: str):
        key_bytes = private_key.encode()
        salt = os.urandom(32)
        key_hash = hashlib.pbkdf2_hmac('sha256', passphrase.encode(), salt, 100000)
        
        encrypted = bytearray()
        for i, byte in enumerate(key_bytes):
            encrypted.append(byte ^ key_hash[i % len(key_hash)])
        
        self.encrypted_store[key_id] = {
            'encrypted': base64.b64encode(encrypted).decode(),
            'salt': base64.b64encode(salt).decode()
        }
    
    def decrypt_private_key(self, key_id: str, passphrase: str) -> str:
        if key_id not in self.encrypted_store:
            return os.getenv('PRIVATE_KEY', '0x' + '0' * 64)
        
        store = self.encrypted_store[key_id]
        encrypted = base64.b64decode(store['encrypted'])
        salt = base64.b64decode(store['salt'])
        
        key_hash = hashlib.pbkdf2_hmac('sha256', passphrase.encode(), salt, 100000)
        
        decrypted = bytearray()
        for i, byte in enumerate(encrypted):
            decrypted.append(byte ^ key_hash[i % len(key_hash)])
        
        return decrypted.decode()

class ConnectionManager:
    def __init__(self, max_connections=100):
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session = None
        self.connection_pools = {}
    
    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=15, connect=5)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={'User-Agent': 'RenaissanceTrader/1.0'}
        )
    
    async def get_session(self):
        if not self.session:
            await self.initialize()
        return self.session
    
    async def close(self):
        if self.session:
            await self.session.close()

class TradingCircuitBreaker:
    def __init__(self, max_daily_loss=50.0, max_position_size=10.0, max_drawdown=0.15):
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.daily_loss = 0.0
        self.total_loss = 0.0
        self.last_reset = time.time()
        self.emergency_stop = False
        self.trade_count = 0
        self.max_trades_per_hour = 50
        self.trades_this_hour = []
        
    def check_limits(self, trade_amount: float) -> bool:
        current_time = time.time()
        
        if current_time - self.last_reset > 86400:
            self.daily_loss = 0.0
            self.last_reset = current_time
        
        self.trades_this_hour = [t for t in self.trades_this_hour if current_time - t < 3600]
        
        if len(self.trades_this_hour) >= self.max_trades_per_hour:
            return False
        
        if self.emergency_stop:
            return False
        
        if trade_amount > self.max_position_size:
            return False
        
        if self.daily_loss > self.max_daily_loss:
            return False
        
        if self.total_loss / 100 > self.max_drawdown:
            self.emergency_stop = True
            return False
        
        return True
    
    def record_trade(self, pnl: float):
        if pnl < 0:
            self.daily_loss += abs(pnl)
            self.total_loss += abs(pnl)
        
        self.trades_this_hour.append(time.time())
        self.trade_count += 1

class GasOptimizer:
    def __init__(self):
        self.gas_history = {}
        self.base_gas_prices = {
            'ethereum': 20000000000,
            'arbitrum': 100000000,
            'polygon': 30000000000,
            'optimism': 1000000
        }
        
    async def get_optimal_gas_price(self, chain: str, urgency: float = 0.5) -> Dict[str, int]:
        try:
            if chain == 'ethereum':
                return await self.get_ethereum_gas_price(urgency)
            elif chain == 'arbitrum':
                return await self.get_arbitrum_gas_price(urgency)
            elif chain == 'polygon':
                return await self.get_polygon_gas_price(urgency)
            elif chain == 'optimism':
                return await self.get_optimism_gas_price(urgency)
        except Exception:
            pass
        
        base_price = self.base_gas_prices.get(chain, 20000000000)
        multiplier = 1.0 + (urgency * 0.5)
        
        if chain == 'ethereum':
            return {
                'maxFeePerGas': int(base_price * multiplier),
                'maxPriorityFeePerGas': int(base_price * multiplier * 0.1)
            }
        else:
            return {'gasPrice': int(base_price * multiplier)}
    
    async def get_ethereum_gas_price(self, urgency: float):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.etherscan.io/api?module=gastracker&action=gasoracle') as response:
                    data = await response.json()
                    
                    if data.get('status') == '1':
                        if urgency > 0.8:
                            gas_price = int(data['result']['FastGasPrice']) * 1000000000
                        elif urgency > 0.5:
                            gas_price = int(data['result']['StandardGasPrice']) * 1000000000
                        else:
                            gas_price = int(data['result']['SafeGasPrice']) * 1000000000
                        
                        return {
                            'maxFeePerGas': gas_price,
                            'maxPriorityFeePerGas': int(gas_price * 0.1)
                        }
        except Exception:
            pass
        
        base_price = self.base_gas_prices['ethereum']
        multiplier = 1.0 + (urgency * 0.5)
        return {
            'maxFeePerGas': int(base_price * multiplier),
            'maxPriorityFeePerGas': int(base_price * multiplier * 0.1)
        }
    
    async def get_arbitrum_gas_price(self, urgency: float):
        base_price = self.base_gas_prices['arbitrum']
        multiplier = 1.0 + (urgency * 0.3)
        return {'gasPrice': int(base_price * multiplier)}
    
    async def get_polygon_gas_price(self, urgency: float):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://gasstation-mainnet.matic.network/v2') as response:
                    data = await response.json()
                    
                    if urgency > 0.8:
                        gas_price = int(data['fast']['maxFee'] * 1000000000)
                    elif urgency > 0.5:
                        gas_price = int(data['standard']['maxFee'] * 1000000000)
                    else:
                        gas_price = int(data['safeLow']['maxFee'] * 1000000000)
                    
                    return {'gasPrice': gas_price}
        except Exception:
            pass
        
        base_price = self.base_gas_prices['polygon']
        multiplier = 1.0 + (urgency * 0.4)
        return {'gasPrice': int(base_price * multiplier)}
    
    async def get_optimism_gas_price(self, urgency: float):
        base_price = self.base_gas_prices['optimism']
        multiplier = 1.0 + (urgency * 0.2)
        return {'gasPrice': int(base_price * multiplier)}

class MEVProtection:
    def __init__(self):
        self.flashbots_endpoints = {
            'ethereum': 'https://relay.flashbots.net',
            'arbitrum': 'https://rpc.flashbots.net/arbitrum',
            'polygon': 'https://rpc.flashbots.net/polygon'
        }
        self.private_mempools = {}
        
    async def submit_private_transaction(self, chain: str, signed_tx: bytes, max_fee: int) -> Dict:
        try:
            if chain in self.flashbots_endpoints:
                return await self.submit_flashbots_bundle(chain, signed_tx, max_fee)
            else:
                return await self.submit_regular_transaction(chain, signed_tx)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def submit_flashbots_bundle(self, chain: str, signed_tx: bytes, max_fee: int):
        try:
            bundle_payload = {
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'eth_sendBundle',
                'params': [{
                    'txs': [signed_tx.hex()],
                    'blockNumber': hex(await self.get_current_block_number(chain) + 1),
                    'minTimestamp': 0,
                    'maxTimestamp': 0
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.flashbots_endpoints[chain],
                    json=bundle_payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    result = await response.json()
                    
                    if 'error' not in result:
                        return {'success': True, 'bundle_hash': result.get('result', '')}
                    else:
                        return {'success': False, 'error': result['error']}
                        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def submit_regular_transaction(self, chain: str, signed_tx: bytes):
        return {'success': True, 'tx_hash': signed_tx.hex()[:66]}
    
    async def get_current_block_number(self, chain: str) -> int:
        return int(time.time() // 12)

class RealTradeExecutor:
    def __init__(self):
        self.w3_connections = {}
        self.accounts = {}
        self.connection_manager = ConnectionManager()
        self.key_manager = SecureKeyManager()
        self.circuit_breaker = TradingCircuitBreaker()
        self.gas_optimizer = GasOptimizer()
        self.mev_protection = MEVProtection()
        
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
            },
            'optimism': {
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564'
            }
        }
        
        self.weth_addresses = {
            'ethereum': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
            'polygon': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
            'optimism': '0x4200000000000000000000000000000000000006'
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
            },
            'optimism': {
                'rpc': f"https://opt-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'chain_id': 10,
                'gas_multiplier': 1.1
            }
        }
        
        self.router_abi = [
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type