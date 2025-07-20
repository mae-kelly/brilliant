import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from web3 import Web3
from eth_account import Account
from eth_utils import to_checksum_address
import os

@dataclass
class ExecutionResult:
    success: bool
    tx_hash: str
    amount_out: float
    execution_price: float
    gas_used: int
    gas_cost: float
    slippage: float
    execution_time: float
    route_path: List[str]
    error_message: str = ""

class RealDEXExecutor:
    def __init__(self):
        self.w3_connections = {}
        self.accounts = {}
        self.session = None
        
        self.router_contracts = {
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
                'quickswap': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff'
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
        
        self.uniswap_v2_abi = [
            {
                "constant": False,
                "inputs": [
                    {"name": "amountOutMin", "type": "uint256"},
                    {"name": "path", "type": "address[]"},
                    {"name": "to", "type": "address"},
                    {"name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactETHForTokens",
                "outputs": [{"name": "amounts", "type": "uint256[]"}],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "amountOutMin", "type": "uint256"},
                    {"name": "path", "type": "address[]"},
                    {"name": "to", "type": "address"},
                    {"name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactTokensForETH",
                "outputs": [{"name": "amounts", "type": "uint256[]"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [
                    {"name": "amountIn", "type": "uint256"},
                    {"name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [{"name": "amounts", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
    async def initialize(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        
        private_key = os.getenv('PRIVATE_KEY')
        if not private_key or private_key.startswith('0x0123'):
            raise ValueError("Valid private key required for real trading")
        
        for chain, config in self.chain_configs.items():
            try:
                self.w3_connections[chain] = Web3(Web3.HTTPProvider(config['rpc']))
                if self.w3_connections[chain].is_connected():
                    self.accounts[chain] = Account.from_key(private_key)
                    print(f"✅ Connected to {chain}")
                else:
                    raise Exception(f"Failed to connect to {chain}")
            except Exception as e:
                print(f"❌ Error connecting to {chain}: {e}")
                raise

    async def execute_buy_trade(self, token_address: str, chain: str, amount_eth: float) -> ExecutionResult:
        start_time = time.time()
        
        try:
            w3 = self.w3_connections[chain]
            account = self.accounts[chain]
            
            best_router, quote = await self.get_best_quote(token_address, chain, amount_eth, 'buy')
            
            if not quote:
                return ExecutionResult(
                    success=False, tx_hash="", amount_out=0, execution_price=0,
                    gas_used=0, gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                    route_path=[], error_message="No quote available"
                )
            
            router_address = self.router_contracts[chain][best_router]
            router_contract = w3.eth.contract(
                address=to_checksum_address(router_address),
                abi=self.uniswap_v2_abi
            )
            
            weth_address = self.weth_addresses[chain]
            path = [to_checksum_address(weth_address), to_checksum_address(token_address)]
            deadline = int(time.time()) + 300
            amount_out_min = int(quote['amount_out'] * 0.97)
            amount_in_wei = int(amount_eth * 1e18)
            
            gas_price = await self.get_optimal_gas_price(chain)
            
            transaction = router_contract.functions.swapExactETHForTokens(
                amount_out_min,
                path,
                account.address,
                deadline
            ).build_transaction({
                'from': account.address,
                'value': amount_in_wei,
                'gasPrice': gas_price,
                'gas': 250000,
                'nonce': w3.eth.get_transaction_count(account.address)
            })
            
            signed_txn = account.sign_transaction(transaction)
            
            try:
                tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                
                if receipt['status'] == 1:
                    actual_gas_used = receipt['gasUsed']
                    gas_cost = (actual_gas_used * gas_price) / 1e18
                    
                    logs = router_contract.events.filter_logs(receipt['logs'])
                    amount_out = 0
                    
                    for log in logs:
                        if hasattr(log, 'args') and hasattr(log.args, 'amounts'):
                            amount_out = log.args.amounts[-1]
                            break
                    
                    if amount_out == 0:
                        amount_out = quote['amount_out']
                    
                    execution_price = amount_out / amount_in_wei if amount_in_wei > 0 else 0
                    expected_price = quote['amount_out'] / amount_in_wei
                    slippage = abs(execution_price - expected_price) / expected_price if expected_price > 0 else 0
                    
                    return ExecutionResult(
                        success=True,
                        tx_hash=tx_hash.hex(),
                        amount_out=amount_out,
                        execution_price=execution_price,
                        gas_used=actual_gas_used,
                        gas_cost=gas_cost,
                        slippage=slippage,
                        execution_time=time.time() - start_time,
                        route_path=[weth_address, token_address]
                    )
                else:
                    return ExecutionResult(
                        success=False, tx_hash=tx_hash.hex(), amount_out=0, execution_price=0,
                        gas_used=receipt['gasUsed'], gas_cost=0, slippage=0,
                        execution_time=time.time() - start_time, route_path=[],
                        error_message="Transaction reverted"
                    )
                    
            except Exception as e:
                return ExecutionResult(
                    success=False, tx_hash="", amount_out=0, execution_price=0,
                    gas_used=0, gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                    route_path=[], error_message=f"Transaction failed: {str(e)}"
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False, tx_hash="", amount_out=0, execution_price=0,
                gas_used=0, gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                route_path=[], error_message=f"Execution error: {str(e)}"
            )

    async def execute_sell_trade(self, token_address: str, chain: str, token_amount: int) -> ExecutionResult:
        start_time = time.time()
        
        try:
            w3 = self.w3_connections[chain]
            account = self.accounts[chain]
            
            token_contract = w3.eth.contract(
                address=to_checksum_address(token_address),
                abi=[
                    {
                        "constant": False,
                        "inputs": [
                            {"name": "_spender", "type": "address"},
                            {"name": "_value", "type": "uint256"}
                        ],
                        "name": "approve",
                        "outputs": [{"name": "", "type": "bool"}],
                        "stateMutability": "nonpayable",
                        "type": "function"
                    }
                ]
            )
            
            best_router, quote = await self.get_best_quote(token_address, chain, token_amount, 'sell')
            
            if not quote:
                return ExecutionResult(
                    success=False, tx_hash="", amount_out=0, execution_price=0,
                    gas_used=0, gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                    route_path=[], error_message="No quote available"
                )
            
            router_address = self.router_contracts[chain][best_router]
            
            approve_txn = token_contract.functions.approve(
                to_checksum_address(router_address),
                token_amount
            ).build_transaction({
                'from': account.address,
                'gasPrice': await self.get_optimal_gas_price(chain),
                'gas': 100000,
                'nonce': w3.eth.get_transaction_count(account.address)
            })
            
            signed_approve = account.sign_transaction(approve_txn)
            approve_hash = w3.eth.send_raw_transaction(signed_approve.rawTransaction)
            w3.eth.wait_for_transaction_receipt(approve_hash, timeout=120)
            
            router_contract = w3.eth.contract(
                address=to_checksum_address(router_address),
                abi=self.uniswap_v2_abi
            )
            
            weth_address = self.weth_addresses[chain]
            path = [to_checksum_address(token_address), to_checksum_address(weth_address)]
            deadline = int(time.time()) + 300
            amount_out_min = int(quote['amount_out'] * 0.97)
            
            swap_txn = router_contract.functions.swapExactTokensForETH(
                token_amount,
                amount_out_min,
                path,
                account.address,
                deadline
            ).build_transaction({
                'from': account.address,
                'gasPrice': await self.get_optimal_gas_price(chain),
                'gas': 250000,
                'nonce': w3.eth.get_transaction_count(account.address)
            })
            
            signed_swap = account.sign_transaction(swap_txn)
            swap_hash = w3.eth.send_raw_transaction(signed_swap.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(swap_hash, timeout=120)
            
            if receipt['status'] == 1:
                gas_cost = (receipt['gasUsed'] * swap_txn['gasPrice']) / 1e18
                amount_out = quote['amount_out']
                execution_price = amount_out / token_amount if token_amount > 0 else 0
                
                return ExecutionResult(
                    success=True,
                    tx_hash=swap_hash.hex(),
                    amount_out=amount_out,
                    execution_price=execution_price,
                    gas_used=receipt['gasUsed'],
                    gas_cost=gas_cost,
                    slippage=0.01,
                    execution_time=time.time() - start_time,
                    route_path=[token_address, weth_address]
                )
            else:
                return ExecutionResult(
                    success=False, tx_hash=swap_hash.hex(), amount_out=0, execution_price=0,
                    gas_used=receipt['gasUsed'], gas_cost=0, slippage=0,
                    execution_time=time.time() - start_time, route_path=[],
                    error_message="Swap transaction reverted"
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False, tx_hash="", amount_out=0, execution_price=0,
                gas_used=0, gas_cost=0, slippage=0, execution_time=time.time() - start_time,
                route_path=[], error_message=f"Sell execution error: {str(e)}"
            )

    async def get_best_quote(self, token_address: str, chain: str, amount: float, side: str) -> Tuple[str, Optional[Dict]]:
        routers = list(self.router_contracts[chain].keys())
        best_router = None
        best_quote = None
        best_output = 0
        
        for router in routers:
            try:
                quote = await self.get_router_quote(router, token_address, chain, amount, side)
                if quote and quote['amount_out'] > best_output:
                    best_output = quote['amount_out']
                    best_quote = quote
                    best_router = router
            except Exception as e:
                continue
        
        return best_router, best_quote

    async def get_router_quote(self, router: str, token_address: str, chain: str, amount: float, side: str) -> Optional[Dict]:
        try:
            w3 = self.w3_connections[chain]
            router_address = self.router_contracts[chain][router]
            router_contract = w3.eth.contract(
                address=to_checksum_address(router_address),
                abi=self.uniswap_v2_abi
            )
            
            weth_address = self.weth_addresses[chain]
            
            if side == 'buy':
                path = [to_checksum_address(weth_address), to_checksum_address(token_address)]
                amount_in = int(amount * 1e18)
            else:
                path = [to_checksum_address(token_address), to_checksum_address(weth_address)]
                amount_in = int(amount)
            
            amounts_out = router_contract.functions.getAmountsOut(amount_in, path).call()
            
            return {
                'amount_out': amounts_out[-1],
                'path': path,
                'router': router
            }
            
        except Exception as e:
            return None

    async def get_optimal_gas_price(self, chain: str) -> int:
        try:
            w3 = self.w3_connections[chain]
            
            if chain == 'ethereum':
                gas_price = w3.eth.gas_price
                return int(gas_price * 1.1)
            elif chain == 'arbitrum':
                return 1000000000
            elif chain == 'polygon':
                gas_price = w3.eth.gas_price
                return int(gas_price * 1.2)
            else:
                return 20000000000
                
        except Exception as e:
            return 20000000000

    async def estimate_gas_cost(self, chain: str, gas_limit: int = 250000) -> float:
        gas_price = await self.get_optimal_gas_price(chain)
        return (gas_limit * gas_price) / 1e18

    async def close(self):
        if self.session:
            await self.session.close()

real_dex_executor = RealDEXExecutor()