
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
from web3 import Web3
from eth_account import Account
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RouteResult:
    success: bool
    tx_hash: str
    amount_out: float
    gas_used: int
    slippage: float
    execution_time: float

class FlashbotsRelay:
    def __init__(self):
        self.relay_url = "https://relay.flashbots.net"
        self.relay_signing_key = Account.create()
        
    async def send_bundle(self, transactions: List[dict], target_block: int) -> bool:
        try:
            bundle = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_sendBundle",
                "params": [
                    {
                        "txs": [tx['rawTransaction'].hex() for tx in transactions],
                        "blockNumber": hex(target_block)
                    }
                ]
            }
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(self.relay_url, json=bundle) as response:
                    result = await response.json()
                    return 'error' not in result
                    
        except Exception as e:
            return False

class ProductionDEXRouter:
    def __init__(self):
        self.chains = {
            'arbitrum': {
                'rpc': 'https://arb1.arbitrum.io/rpc',
                'chain_id': 42161,
                'routers': {
                    'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                    'camelot': '0xc873fEcbd354f5A56E00E710B90EF4201db2448d',
                    'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
                },
                'weth': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1'
            },
            'optimism': {
                'rpc': 'https://mainnet.optimism.io',
                'chain_id': 10,
                'routers': {
                    'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                    'velodrome': '0x9c12939390052919aF3155f41Bf4160Fd3666A6f'
                },
                'weth': '0x4200000000000000000000000000000000000006'
            },
            'polygon': {
                'rpc': 'https://polygon-rpc.com',
                'chain_id': 137,
                'routers': {
                    'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                    'quickswap': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
                    'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
                },
                'weth': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270'
            }
        }
        
        self.web3_instances = {}
        self.accounts = {}
        self.flashbots = FlashbotsRelay()
        
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
            }
        ]

    async def initialize(self):
        private_key = os.getenv('PRIVATE_KEY')
        if not private_key or private_key.startswith('your_'):
            raise ValueError("PRIVATE_KEY not configured")
        
        for chain_name, config in self.chains.items():
            try:
                w3 = Web3(Web3.HTTPProvider(config['rpc']))
                if w3.is_connected():
                    self.web3_instances[chain_name] = w3
                    self.accounts[chain_name] = Account.from_key(private_key)
                    
            except Exception as e:
                continue

    async def get_best_route(self, token_in: str, token_out: str, amount_in: int, chain: str) -> Tuple[str, int]:
        if chain not in self.chains:
            return None, 0
        
        w3 = self.web3_instances[chain]
        chain_config = self.chains[chain]
        
        best_router = None
        best_amount_out = 0
        
        for router_name, router_address in chain_config['routers'].items():
            try:
                router_contract = w3.eth.contract(
                    address=Web3.to_checksum_address(router_address),
                    abi=self.router_abi
                )
                
                path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]
                amounts_out = router_contract.functions.getAmountsOut(amount_in, path).call()
                
                if amounts_out[1] > best_amount_out:
                    best_amount_out = amounts_out[1]
                    best_router = router_address
                    
            except Exception as e:
                continue
        
        return best_router, best_amount_out

    async def execute_swap(self, token_in: str, token_out: str, amount_in: int, min_amount_out: int, chain: str, use_flashbots: bool = True) -> RouteResult:
        start_time = time.time()
        
        if chain not in self.web3_instances:
            return RouteResult(False, "", 0, 0, 0, time.time() - start_time)
        
        w3 = self.web3_instances[chain]
        account = self.accounts[chain]
        chain_config = self.chains[chain]
        
        try:
            best_router, expected_out = await self.get_best_route(token_in, token_out, amount_in, chain)
            if not best_router:
                return RouteResult(False, "", 0, 0, 0, time.time() - start_time)
            
            router_contract = w3.eth.contract(
                address=Web3.to_checksum_address(best_router),
                abi=self.router_abi
            )
            
            nonce = w3.eth.get_transaction_count(account.address)
            gas_price = await self.get_optimal_gas_price(w3, chain)
            deadline = int(time.time()) + get_dynamic_config().get("max_hold_time", 300)
            
            if token_in.lower() == chain_config['weth'].lower():
                tx_data = router_contract.functions.swapExactETHForTokens(
                    min_amount_out,
                    [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)],
                    account.address,
                    deadline
                ).build_transaction({
                    'from': account.address,
                    'value': amount_in,
                    'gas': get_dynamic_config().get("max_hold_time", 300)000,
                    'gasPrice': gas_price,
                    'nonce': nonce
                })
            else:
                await self.approve_token(token_in, best_router, amount_in, chain)
                
                tx_data = router_contract.functions.swapExactTokensForETH(
                    amount_in,
                    min_amount_out,
                    [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)],
                    account.address,
                    deadline
                ).build_transaction({
                    'from': account.address,
                    'gas': 350000,
                    'gasPrice': gas_price,
                    'nonce': nonce
                })
            
            signed_tx = account.sign_transaction(tx_data)
            
            if use_flashbots and chain == 'ethereum':
                current_block = w3.eth.block_number
                bundle_sent = await self.flashbots.send_bundle([signed_tx], current_block + 1)
                
                if bundle_sent:
                    for i in range(20):
                        try:
                            receipt = w3.eth.get_transaction_receipt(signed_tx.hash)
                            if receipt:
                                break
                        except:
                            await asyncio.sleep(3)
                            continue
                else:
                    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            else:
                tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            success = receipt['status'] == 1
            gas_used = receipt['gasUsed']
            actual_slippage = abs(expected_out - min_amount_out) / expected_out if expected_out > 0 else 0
            
            return RouteResult(
                success=success,
                tx_hash=receipt['transactionHash'].hex(),
                amount_out=float(expected_out),
                gas_used=gas_used,
                slippage=actual_slippage,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return RouteResult(False, "", 0, 0, 0, time.time() - start_time)

    async def approve_token(self, token_address: str, spender: str, amount: int, chain: str):
        w3 = self.web3_instances[chain]
        account = self.accounts[chain]
        
        erc20_abi = [
            {
                "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
        token_contract = w3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=erc20_abi
        )
        
        nonce = w3.eth.get_transaction_count(account.address)
        gas_price = await self.get_optimal_gas_price(w3, chain)
        
        approve_tx = token_contract.functions.approve(
            Web3.to_checksum_address(spender),
            amount
        ).build_transaction({
            'from': account.address,
            'gas': 100000,
            'gasPrice': gas_price,
            'nonce': nonce
        })
        
        signed_approve = account.sign_transaction(approve_tx)
        approve_hash = w3.eth.send_raw_transaction(signed_approve.rawTransaction)
        w3.eth.wait_for_transaction_receipt(approve_hash, timeout=60)

    async def get_optimal_gas_price(self, w3: Web3, chain: str) -> int:
        try:
            if hasattr(w3.eth, 'get_block'):
                latest_block = w3.eth.get_block('latest')
                base_fee = latest_block.get('baseFeePerGas', 0)
                
                if base_fee > 0:
                    priority_fee = w3.eth.max_priority_fee
                    return int((base_fee + priority_fee) * 1.1)
            
            return int(w3.eth.gas_price * 1.1)
            
        except:
            return 20 * 10**9

production_router = ProductionDEXRouter()
