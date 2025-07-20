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

class RealUniswapV3Executor:
    def __init__(self):
        self.w3_connections = {}
        self.routers = {
            'ethereum': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'arbitrum': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'polygon': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'optimism': '0xE592427A0AEce92De3Edee1F18E0157C05861564'
        }
        
        self.quoter_v2 = {
            'ethereum': '0x61fFE014bA17989E743c5F6cB21bF9697530B21e',
            'arbitrum': '0x61fFE014bA17989E743c5F6cB21bF9697530B21e',
            'polygon': '0x61fFE014bA17989E743c5F6cB21bF9697530B21e',
            'optimism': '0x61fFE014bA17989E743c5F6cB21bF9697530B21e'
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
                'gas_price_multiplier': 1.2
            },
            'arbitrum': {
                'rpc': f"https://arb-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'chain_id': 42161,
                'gas_price_multiplier': 1.1
            },
            'polygon': {
                'rpc': f"https://polygon-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'chain_id': 137,
                'gas_price_multiplier': 1.3
            },
            'optimism': {
                'rpc': f"https://opt-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'chain_id': 10,
                'gas_price_multiplier': 1.1
            }
        }
        
        self.router_abi = [
            {
                "inputs": [
                    {
                        "components": [
                            {"name": "tokenIn", "type": "address"},
                            {"name": "tokenOut", "type": "address"},
                            {"name": "fee", "type": "uint24"},
                            {"name": "recipient", "type": "address"},
                            {"name": "deadline", "type": "uint256"},
                            {"name": "amountIn", "type": "uint256"},
                            {"name": "amountOutMinimum", "type": "uint256"},
                            {"name": "sqrtPriceLimitX96", "type": "uint160"}
                        ],
                        "name": "params",
                        "type": "tuple"
                    }
                ],
                "name": "exactInputSingle",
                "outputs": [{"name": "amountOut", "type": "uint256"}],
                "type": "function"
            }
        ]
        
        self.quoter_abi = [
            {
                "inputs": [
                    {"name": "tokenIn", "type": "address"},
                    {"name": "tokenOut", "type": "address"},
                    {"name": "fee", "type": "uint24"},
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"}
                ],
                "name": "quoteExactInputSingle",
                "outputs": [
                    {"name": "amountOut", "type": "uint256"},
                    {"name": "sqrtPriceX96After", "type": "uint160"},
                    {"name": "initializedTicksCrossed", "type": "uint32"},
                    {"name": "gasEstimate", "type": "uint256"}
                ],
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
        
        self.session = None
        self.account = None
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
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        
        for chain, config in self.chain_configs.items():
            try:
                self.w3_connections[chain] = Web3(Web3.HTTPProvider(config['rpc']))
                if self.w3_connections[chain].is_connected():
                    self.logger.info(f"✅ Connected to {chain}")
                else:
                    self.logger.error(f"❌ Failed to connect to {chain}")
            except Exception as e:
                self.logger.error(f"❌ Error connecting to {chain}: {e}")
        
        private_key = os.getenv('PRIVATE_KEY')
        if private_key and not private_key.startswith('0x00'):
            self.account = Account.from_key(private_key)
            self.logger.info(f"✅ Wallet loaded: {self.account.address}")
        else:
            self.logger.warning("⚠️ No valid private key - using demo mode")
        
        self.logger.info("🚀 Real Uniswap V3 Executor initialized")

    async def execute_buy_trade(self, token_address: str, chain: str, amount_usd: float) -> Dict:
        start_time = time.time()
        
        try:
            if not self.account or os.getenv('DRY_RUN', 'true').lower() == 'true':
                return await self.simulate_trade(token_address, chain, amount_usd, 'buy')
            
            w3 = self.w3_connections.get(chain)
            if not w3:
                return {'success': False, 'error': f'No connection to {chain}'}
            
            weth_address = self.weth_addresses[chain]
            router_address = self.routers[chain]
            
            quote_result = await self.get_quote(
                weth_address, token_address, amount_usd, chain
            )
            
            if not quote_result['success']:
                return quote_result
            
            amount_in = int(amount_usd * 1e18)
            amount_out_min = int(quote_result['amount_out'] * 0.97)
            
            trade_result = await self.execute_swap(
                weth_address, token_address, amount_in, amount_out_min, chain
            )
            
            execution_time = time.time() - start_time
            self._update_stats(trade_result, execution_time)
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Buy trade failed: {e}")
            return {'success': False, 'error': str(e)}

    async def execute_sell_trade(self, token_address: str, chain: str, token_amount: int) -> Dict:
        start_time = time.time()
        
        try:
            if not self.account or os.getenv('DRY_RUN', 'true').lower() == 'true':
                return await self.simulate_trade(token_address, chain, token_amount / 1e18, 'sell')
            
            w3 = self.w3_connections.get(chain)
            if not w3:
                return {'success': False, 'error': f'No connection to {chain}'}
            
            weth_address = self.weth_addresses[chain]
            
            quote_result = await self.get_quote(
                token_address, weth_address, token_amount, chain
            )
            
            if not quote_result['success']:
                return quote_result
            
            amount_out_min = int(quote_result['amount_out'] * 0.97)
            
            trade_result = await self.execute_swap(
                token_address, weth_address, token_amount, amount_out_min, chain
            )
            
            execution_time = time.time() - start_time
            self._update_stats(trade_result, execution_time)
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Sell trade failed: {e}")
            return {'success': False, 'error': str(e)}

    async def get_quote(self, token_in: str, token_out: str, amount_in: float, chain: str) -> Dict:
        try:
            w3 = self.w3_connections[chain]
            quoter_address = self.quoter_v2[chain]
            
            quoter_contract = w3.eth.contract(
                address=quoter_address,
                abi=self.quoter_abi
            )
            
            amount_in_wei = int(amount_in * 1e18) if isinstance(amount_in, float) else int(amount_in)
            
            fee_tiers = [500, 3000, 10000]
            best_quote = None
            best_amount_out = 0
            
            for fee in fee_tiers:
                try:
                    quote = quoter_contract.functions.quoteExactInputSingle(
                        token_in,
                        token_out,
                        fee,
                        amount_in_wei,
                        0
                    ).call()
                    
                    amount_out = quote[0]
                    if amount_out > best_amount_out:
                        best_amount_out = amount_out
                        best_quote = {
                            'amount_out': amount_out,
                            'fee': fee,
                            'gas_estimate': quote[3]
                        }
                        
                except Exception as e:
                    continue
            
            if best_quote:
                return {
                    'success': True,
                    'amount_out': best_quote['amount_out'],
                    'fee': best_quote['fee'],
                    'gas_estimate': best_quote['gas_estimate']
                }
            else:
                return {'success': False, 'error': 'No valid quote found'}
                
        except Exception as e:
            return {'success': False, 'error': f'Quote failed: {str(e)}'}

    async def execute_swap(self, token_in: str, token_out: str, amount_in: int, amount_out_min: int, chain: str) -> Dict:
        try:
            w3 = self.w3_connections[chain]
            router_address = self.routers[chain]
            
            if token_in != self.weth_addresses[chain]:
                approval_result = await self.approve_token(token_in, router_address, amount_in, chain)
                if not approval_result['success']:
                    return approval_result
            
            router_contract = w3.eth.contract(
                address=router_address,
                abi=self.router_abi
            )
            
            quote_result = await self.get_quote(token_in, token_out, amount_in, chain)
            if not quote_result['success']:
                return quote_result
            
            fee = quote_result['fee']
            deadline = int(time.time()) + 300
            
            swap_params = {
                'tokenIn': token_in,
                'tokenOut': token_out,
                'fee': fee,
                'recipient': self.account.address,
                'deadline': deadline,
                'amountIn': amount_in,
                'amountOutMinimum': amount_out_min,
                'sqrtPriceLimitX96': 0
            }
            
            nonce = w3.eth.get_transaction_count(self.account.address)
            gas_price = await self.get_optimal_gas_price(chain)
            
            transaction = router_contract.functions.exactInputSingle(swap_params).build_transaction({
                'chainId': self.chain_configs[chain]['chain_id'],
                'gas': 300000,
                'gasPrice': gas_price,
                'nonce': nonce,
                'value': amount_in if token_in == self.weth_addresses[chain] else 0
            })
            
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key=self.account.key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = await self.wait_for_transaction(w3, tx_hash)
            
            if receipt['status'] == 1:
                actual_amount_out = self.parse_swap_output(receipt, token_out)
                slippage = self.calculate_slippage(quote_result['amount_out'], actual_amount_out)
                
                return {
                    'success': True,
                    'tx_hash': tx_hash.hex(),
                    'executed_amount': actual_amount_out,
                    'execution_price': actual_amount_out / amount_in,
                    'gas_cost': receipt['gasUsed'] * gas_price / 1e18,
                    'slippage': slippage,
                    'execution_time': time.time(),
                    'route_used': f'Uniswap V3 Fee {fee}'
                }
            else:
                return {'success': False, 'error': 'Transaction failed'}
                
        except Exception as e:
            return {'success': False, 'error': f'Swap execution failed: {str(e)}'}

    async def approve_token(self, token_address: str, spender: str, amount: int, chain: str) -> Dict:
        try:
            w3 = self.w3_connections[chain]
            token_contract = w3.eth.contract(address=token_address, abi=self.erc20_abi)
            
            current_allowance = token_contract.functions.allowance(
                self.account.address, spender
            ).call()
            
            if current_allowance >= amount:
                return {'success': True, 'message': 'Already approved'}
            
            nonce = w3.eth.get_transaction_count(self.account.address)
            gas_price = await self.get_optimal_gas_price(chain)
            
            approve_txn = token_contract.functions.approve(spender, amount * 2).build_transaction({
                'chainId': self.chain_configs[chain]['chain_id'],
                'gas': 100000,
                'gasPrice': gas_price,
                'nonce': nonce
            })
            
            signed_txn = w3.eth.account.sign_transaction(approve_txn, private_key=self.account.key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = await self.wait_for_transaction(w3, tx_hash)
            
            if receipt['status'] == 1:
                return {'success': True, 'tx_hash': tx_hash.hex()}
            else:
                return {'success': False, 'error': 'Approval failed'}
                
        except Exception as e:
            return {'success': False, 'error': f'Approval failed: {str(e)}'}

    async def get_optimal_gas_price(self, chain: str) -> int:
        try:
            w3 = self.w3_connections[chain]
            
            if chain == 'ethereum':
                gas_price = w3.eth.gas_price
                return int(gas_price * self.chain_configs[chain]['gas_price_multiplier'])
            else:
                gas_price = w3.eth.gas_price
                return max(int(gas_price * self.chain_configs[chain]['gas_price_multiplier']), 1000000000)
                
        except Exception as e:
            default_prices = {
                'ethereum': 20000000000,
                'arbitrum': 100000000,
                'polygon': 30000000000,
                'optimism': 1000000
            }
            return default_prices.get(chain, 20000000000)

    async def wait_for_transaction(self, w3: Web3, tx_hash: bytes, timeout: int = 120):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                receipt = w3.eth.get_transaction_receipt(tx_hash)
                return receipt
            except:
                await asyncio.sleep(2)
        
        raise Exception("Transaction timeout")

    def parse_swap_output(self, receipt: Dict, token_out: str) -> float:
        for log in receipt['logs']:
            if log['topics'][0].hex() == '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':
                if log['address'].lower() == token_out.lower():
                    return int(log['data'], 16) / 1e18
        return 0.0

    def calculate_slippage(self, expected: float, actual: float) -> float:
        if expected == 0:
            return 0.0
        return abs((actual - expected) / expected)

    async def simulate_trade(self, token_address: str, chain: str, amount: float, side: str) -> Dict:
        await asyncio.sleep(np.random.uniform(0.5, 2.0))
        
        success = np.random.random() > 0.05
        
        if success:
            if side == 'buy':
                executed_amount = amount * np.random.uniform(0.98, 1.0)
                slippage = np.random.uniform(0.001, 0.03)
                execution_price = (1 + slippage) if side == 'buy' else (1 - slippage)
            else:
                executed_amount = amount * np.random.uniform(0.98, 1.0)
                slippage = np.random.uniform(0.001, 0.03)
                execution_price = (1 - slippage) if side == 'sell' else (1 + slippage)
            
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
                'route_used': f'Simulated Uniswap V3'
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

    async def get_token_balance(self, token_address: str, chain: str) -> float:
        try:
            if not self.account:
                return 0.0
            
            w3 = self.w3_connections.get(chain)
            if not w3:
                return 0.0
            
            if token_address.lower() == self.weth_addresses[chain].lower():
                balance = w3.eth.get_balance(self.account.address)
                return balance / 1e18
            else:
                token_contract = w3.eth.contract(address=token_address, abi=self.erc20_abi)
                balance = token_contract.functions.balanceOf(self.account.address).call()
                decimals = token_contract.functions.decimals().call()
                return balance / (10 ** decimals)
                
        except Exception as e:
            return 0.0

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
        if self.session:
            await self.session.close()

real_executor = RealUniswapV3Executor()