import os
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
import time
import asyncio
from web3 import Web3
from eth_account import Account
import aiohttp
import json
from typing import Dict, Optional, Tuple
from decimal import Decimal

class RealTradingEngine:
    def __init__(self):
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        self.enable_real_trading = os.getenv('ENABLE_REAL_TRADING', 'false').lower() == 'true'
        
        self.web3_instances = {}
        self.account = None
        
        if not self.dry_run and self.enable_real_trading:
            self._init_real_trading()
        
        self.trade_count = 0
        self.total_profit = 0.0
        
    def _init_real_trading(self):
        try:
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key or private_key.startswith('0x000'):
                raise ValueError("Real trading enabled but no valid private key")
            
            self.account = Account.from_key(private_key)
            
            alchemy_key = os.getenv('ALCHEMY_API_KEY')
            if not alchemy_key or 'your_' in alchemy_key:
                raise ValueError("Real trading enabled but no valid Alchemy key")
            
            rpcs = {
                'arbitrum': f'https://arb-mainnet.g.alchemy.com/v2/{alchemy_key}',
                'polygon': f'https://polygon-mainnet.g.alchemy.com/v2/{alchemy_key}',
                'optimism': f'https://opt-mainnet.g.alchemy.com/v2/{alchemy_key}'
            }
            
            for chain, rpc in rpcs.items():
                w3 = Web3(Web3.HTTPProvider(rpc))
                if w3.is_connected():
                    self.web3_instances[chain] = w3
                    
        except Exception as e:
            print(f"Real trading init failed: {e}")
            self.dry_run = True
    
    async def execute_buy(self, token_address: str, chain: str, amount_eth: float) -> Dict:
        start_time = time.time()
        
        if self.dry_run:
            return await self._simulate_buy(token_address, chain, amount_eth, start_time)
        
        try:
            w3 = self.web3_instances[chain]
            
            router_addresses = {
                'arbitrum': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',
                'polygon': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
                'optimism': '0x4A7b5Da61326A6379179b40d00F57E5bbDC962c2'
            }
            
            router_address = router_addresses[chain]
            
            weth_addresses = {
                'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
                'polygon': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
                'optimism': '0x4200000000000000000000000000000000000006'
            }
            
            weth_address = weth_addresses[chain]
            
            router_abi = [
                {
                    "inputs": [
                        {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                        {"internalType": "address[]", "name": "path", "type": "address[]"},
                        {"internalType": "address", "name": "to", "type": "address"},
                        {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                    ],
                    "name": "swapExactETHForTokens",
                    "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                    "stateMutability": "payable",
                    "type": "function"
                }
            ]
            
            router = w3.eth.contract(address=Web3.to_checksum_address(router_address), abi=router_abi)
            
            amount_in = w3.to_wei(amount_eth, 'ether')
            path = [Web3.to_checksum_address(weth_address), Web3.to_checksum_address(token_address)]
            deadline = int(time.time()) + 300
            
            gas_price = w3.eth.gas_price
            nonce = w3.eth.get_transaction_count(self.account.address)
            
            transaction = router.functions.swapExactETHForTokens(
                0,
                path,
                self.account.address,
                deadline
            ).build_transaction({
                'from': self.account.address,
                'value': amount_in,
                'gas': 300000,
                'gasPrice': gas_price,
                'nonce': nonce
            })
            
            signed_txn = self.account.sign_transaction(transaction)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            return {
                'success': receipt['status'] == 1,
                'tx_hash': tx_hash.hex(),
                'gas_used': receipt['gasUsed'],
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    async def execute_sell(self, token_address: str, chain: str, token_amount: int) -> Dict:
        start_time = time.time()
        
        if self.dry_run:
            return await self._simulate_sell(token_address, chain, token_amount, start_time)
        
        try:
            w3 = self.web3_instances[chain]
            
            erc20_abi = [
                {"inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}], 
                 "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function"}
            ]
            
            token_contract = w3.eth.contract(address=Web3.to_checksum_address(token_address), abi=erc20_abi)
            
            router_addresses = {
                'arbitrum': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',
                'polygon': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
                'optimism': '0x4A7b5Da61326A6379179b40d00F57E5bbDC962c2'
            }
            
            router_address = router_addresses[chain]
            
            approve_txn = token_contract.functions.approve(
                Web3.to_checksum_address(router_address),
                token_amount
            ).build_transaction({
                'from': self.account.address,
                'gas': 100000,
                'gasPrice': w3.eth.gas_price,
                'nonce': w3.eth.get_transaction_count(self.account.address)
            })
            
            signed_approve = self.account.sign_transaction(approve_txn)
            approve_hash = w3.eth.send_raw_transaction(signed_approve.rawTransaction)
            w3.eth.wait_for_transaction_receipt(approve_hash, timeout=60)
            
            router_abi = [
                {
                    "inputs": [
                        {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                        {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                        {"internalType": "address[]", "name": "path", "type": "address[]"},
                        {"internalType": "address", "name": "to", "type": "address"},
                        {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                    ],
                    "name": "swapExactTokensForETH",
                    "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                    "type": "function"
                }
            ]
            
            router = w3.eth.contract(address=Web3.to_checksum_address(router_address), abi=router_abi)
            
            weth_addresses = {
                'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
                'polygon': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
                'optimism': '0x4200000000000000000000000000000000000006'
            }
            
            path = [Web3.to_checksum_address(token_address), Web3.to_checksum_address(weth_addresses[chain])]
            deadline = int(time.time()) + 300
            
            swap_txn = router.functions.swapExactTokensForETH(
                token_amount,
                0,
                path,
                self.account.address,
                deadline
            ).build_transaction({
                'from': self.account.address,
                'gas': 300000,
                'gasPrice': w3.eth.gas_price,
                'nonce': w3.eth.get_transaction_count(self.account.address)
            })
            
            signed_swap = self.account.sign_transaction(swap_txn)
            tx_hash = w3.eth.send_raw_transaction(signed_swap.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            return {
                'success': receipt['status'] == 1,
                'tx_hash': tx_hash.hex(),
                'gas_used': receipt['gasUsed'],
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    async def _simulate_buy(self, token_address: str, chain: str, amount_eth: float, start_time: float) -> Dict:
        await asyncio.sleep(0.5)
        
        success = True
        if hash(token_address) % 100 < 5:
            success = False
        
        return {
            'success': success,
            'tx_hash': f"0x{hash(token_address + str(time.time())) % (16**64):064x}",
            'gas_used': 250000,
            'execution_time': time.time() - start_time
        }
    
    async def _simulate_sell(self, token_address: str, chain: str, token_amount: int, start_time: float) -> Dict:
        await asyncio.sleep(0.5)
        
        profit_multiplier = 0.95 + (0.10 * hash(token_address) % 100) / 100
        
        success = profit_multiplier > 1.0 or hash(token_address) % 100 < 80
        
        if success:
            self.total_profit += (profit_multiplier - 1.0) * 0.01
        
        return {
            'success': success,
            'tx_hash': f"0x{hash(token_address + str(time.time()) + 'sell') % (16**64):064x}",
            'gas_used': 280000,
            'execution_time': time.time() - start_time,
            'profit_multiplier': profit_multiplier
        }

real_trading_engine = RealTradingEngine()
