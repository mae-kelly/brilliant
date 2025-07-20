import asyncio
import time
import aiohttp
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from web3 import Web3
from eth_account import Account
import logging
import os

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

class RealTradeExecutor:
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
                'rpc': f"https://eth-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}",
                'chain_id': 1,
                'gas_multiplier': 1.2
            },
            'arbitrum': {
                'rpc': f"https://arb-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}",
                'chain_id': 42161,
                'gas_multiplier': 1.1
            },
            'polygon': {
                'rpc': f"https://polygon-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}",
                'chain_id': 137,
                'gas_multiplier': 1.3
            }
        }
        
        self.router_abi = [
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
            },
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
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        self.erc20_abi = [
            {"inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}], 
             "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
            {"inputs": [{"name": "_owner", "type": "address"}], 
             "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"},
            {"inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"}
        ]
        
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        
        for chain, config in self.chain_configs.items():
            try:
                self.w3_connections[chain] = Web3(Web3.HTTPProvider(config['rpc']))
                if self.w3_connections[chain].is_connected():
                    self.logger.info(f"Connected to {chain}")
                else:
                    self.logger.error(f"Failed to connect to {chain}")
            except Exception as e:
                self.logger.error(f"Error connecting to {chain}: {e}")
        
        private_key = os.getenv('PRIVATE_KEY')
        if private_key and len(private_key) == 66:
            for chain in self.chain_configs.keys():
                self.accounts[chain] = Account.from_key(private_key)
            self.logger.info(f"Wallet loaded for trading")

    async def execute_buy_trade(self, token_address: str, chain: str, amount_usd: float) -> TradeResult:
        start_time = time.time()
        
        try:
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

    async def execute_sell_trade(self, token_address: str, chain: str, token_amount: float) -> TradeResult:
        start_time = time.time()
        
        try:
            w3 = self.w3_connections.get(chain)
            account = self.accounts.get(chain)
            
            if not w3 or not account:
                return TradeResult(
                    success=False, tx_hash="", executed_amount=0, execution_price=0,
                    gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                    route_used="", error_message="No Web3 provider or account"
                )
            
            best_router, quote = await self.get_best_quote(token_address, chain, token_amount, 'sell')
            if not quote:
                return TradeResult(
                    success=False, tx_hash="", executed_amount=0, execution_price=0,
                    gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                    route_used="", error_message="No valid quote found"
                )
            
            result = await self.execute_swap(w3, account, best_router, token_address, chain, token_amount, quote, 'sell')
            
            execution_time = time.time() - start_time
            
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

    async def get_best_quote(self, token_address: str, chain: str, amount: float, side: str) -> tuple:
        best_router = None
        best_quote = None
        best_amount_out = 0
        
        routers = self.routers.get(chain, {})
        
        for router_name, router_address in routers.items():
            try:
                quote = await self.get_router_quote(router_address, token_address, chain, amount, side)
                if quote and quote['amount_out'] > best_amount_out:
                    best_amount_out = quote['amount_out']
                    best_quote = quote
                    best_router = router_name
            except Exception as e:
                continue
        
        return best_router, best_quote

    async def get_router_quote(self, router_address: str, token_address: str, chain: str, amount: float, side: str) -> Dict:
        try:
            w3 = self.w3_connections[chain]
            router_contract = w3.eth.contract(address=router_address, abi=self.router_abi)
            
            weth_address = self.weth_addresses[chain]
            
            if side == 'buy':
                amount_in = int(amount * 1e18)
                path = [weth_address, token_address]
            else:
                amount_in = int(amount * 1e18)
                path = [token_address, weth_address]
            
            amounts_out = await asyncio.get_event_loop().run_in_executor(
                None, router_contract.functions.getAmountsOut(amount_in, path).call
            )
            
            return {
                'amount_out': amounts_out[-1],
                'path': path,
                'gas_estimate': 200000
            }
            
        except Exception as e:
            return None

    async def execute_swap(self, w3: Web3, account: Account, router_name: str, token_address: str, 
                          chain: str, amount: float, quote: Dict, side: str) -> Dict:
        try:
            router_address = self.routers[chain][router_name]
            router_contract = w3.eth.contract(address=router_address, abi=self.router_abi)
            
            if side == 'sell':
                approval_result = await self.approve_token(w3, account, token_address, router_address, amount)
                if not approval_result:
                    return {'success': False, 'error': 'Token approval failed'}
            
            nonce = await asyncio.get_event_loop().run_in_executor(
                None, w3.eth.get_transaction_count, account.address
            )
            
            gas_price = await self.get_optimal_gas_price(w3, chain)
            deadline = int(time.time()) + 300
            
            amount_in = int(amount * 1e18)
            amount_out_min = int(quote['amount_out'] * 0.97)
            
            if side == 'buy':
                transaction = router_contract.functions.swapExactETHForTokens(
                    amount_out_min,
                    quote['path'],
                    account.address,
                    deadline
                ).build_transaction({
                    'chainId': self.chain_configs[chain]['chain_id'],
                    'gas': quote['gas_estimate'],
                    'gasPrice': gas_price,
                    'nonce': nonce,
                    'value': amount_in
                })
            else:
                transaction = router_contract.functions.swapExactTokensForETH(
                    amount_in,
                    amount_out_min,
                    quote['path'],
                    account.address,
                    deadline
                ).build_transaction({
                    'chainId': self.chain_configs[chain]['chain_id'],
                    'gas': quote['gas_estimate'],
                    'gasPrice': gas_price,
                    'nonce': nonce
                })
            
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key=account.key)
            
            tx_hash = await asyncio.get_event_loop().run_in_executor(
                None, w3.eth.send_raw_transaction, signed_txn.rawTransaction
            )
            
            receipt = await self.wait_for_transaction_receipt(w3, tx_hash)
            
            if receipt['status'] == 1:
                actual_amount_out = self.parse_swap_output(receipt, quote['path'][-1])
                execution_price = actual_amount_out / amount_in if amount_in > 0 else 0
                slippage = abs(actual_amount_out - quote['amount_out']) / quote['amount_out'] if quote['amount_out'] > 0 else 0
                gas_cost = receipt['gasUsed'] * gas_price / 1e18
                
                return {
                    'success': True,
                    'tx_hash': tx_hash.hex(),
                    'amount_out': actual_amount_out,
                    'execution_price': execution_price,
                    'gas_cost': gas_cost,
                    'slippage': slippage
                }
            else:
                return {'success': False, 'error': 'Transaction failed'}
                
        except Exception as e:
            return {'success': False, 'error': f'Swap execution failed: {str(e)}'}

    async def approve_token(self, w3: Web3, account: Account, token_address: str, spender: str, amount: float) -> bool:
        try:
            token_contract = w3.eth.contract(address=token_address, abi=self.erc20_abi)
            
            current_allowance = await asyncio.get_event_loop().run_in_executor(
                None, token_contract.functions.allowance(account.address, spender).call
            )
            
            amount_wei = int(amount * 1e18)
            if current_allowance >= amount_wei:
                return True
            
            nonce = await asyncio.get_event_loop().run_in_executor(
                None, w3.eth.get_transaction_count, account.address
            )
            
            gas_price = await self.get_optimal_gas_price(w3, 'ethereum')
            
            approve_txn = token_contract.functions.approve(spender, amount_wei * 2).build_transaction({
                'chainId': w3.eth.chain_id,
                'gas': 100000,
                'gasPrice': gas_price,
                'nonce': nonce
            })
            
            signed_txn = w3.eth.account.sign_transaction(approve_txn, private_key=account.key)
            tx_hash = await asyncio.get_event_loop().run_in_executor(
                None, w3.eth.send_raw_transaction, signed_txn.rawTransaction
            )
            
            receipt = await self.wait_for_transaction_receipt(w3, tx_hash)
            return receipt['status'] == 1
            
        except Exception as e:
            return False

    async def get_optimal_gas_price(self, w3: Web3, chain: str) -> int:
        try:
            gas_price = await asyncio.get_event_loop().run_in_executor(
                None, lambda: w3.eth.gas_price
            )
            multiplier = self.chain_configs[chain]['gas_multiplier']
            return int(gas_price * multiplier)
        except Exception as e:
            fallback_prices = {
                'ethereum': 20000000000,
                'arbitrum': 100000000,
                'polygon': 30000000000
            }
            return fallback_prices.get(chain, 20000000000)

    async def wait_for_transaction_receipt(self, w3: Web3, tx_hash: bytes, timeout: int = 120):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                receipt = await asyncio.get_event_loop().run_in_executor(
                    None, w3.eth.get_transaction_receipt, tx_hash
                )
                return receipt
            except:
                await asyncio.sleep(2)
        
        raise Exception("Transaction timeout")

    def parse_swap_output(self, receipt: Dict, output_token: str) -> float:
        try:
            for log in receipt['logs']:
                if log['address'].lower() == output_token.lower():
                    amount = int(log['data'], 16) if log['data'] else 0
                    return amount / 1e18
        except Exception as e:
            pass
        return 0.0

    async def close(self):
        if self.session:
            await self.session.close()

real_trade_executor = RealTradeExecutor()