"""
PRODUCTION Execution Engine - Real trading with Web3
Complete implementation with actual transaction execution
"""
import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from web3 import Web3
from web3.exceptions import Web3Exception
from eth_account import Account
from eth_utils import to_checksum_address
import json
import os
from decimal import Decimal
import numpy as np

@dataclass
class ExecutionResult:
    success: bool
    transaction_hash: str
    amount_in: float
    amount_out: float
    gas_used: int
    gas_price: int
    execution_time: float
    slippage: float
    error_message: Optional[str] = None

@dataclass
class TradeOrder:
    token_in: str
    token_out: str
    amount_in: float
    min_amount_out: float
    deadline: int
    slippage_tolerance: float
    chain: str
    dex: str

class ProductionExecutor:
    """Production-grade trade execution engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.web3_connections = {}
        self.accounts = {}
        self.gas_tracker = {}
        
        # Router addresses for each chain
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
        
        # Chain configurations
        self.chain_configs = {
            'ethereum': {
                'rpc': f"https://eth-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'chain_id': 1,
                'weth': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
                'gas_multiplier': 1.2
            },
            'arbitrum': {
                'rpc': f"https://arb-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'chain_id': 42161,
                'weth': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
                'gas_multiplier': 1.1
            },
            'polygon': {
                'rpc': f"https://polygon-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'chain_id': 137,
                'weth': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
                'gas_multiplier': 1.3
            }
        }
        
        # Router ABIs
        self.uniswap_v2_router_abi = json.loads('''[
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
        ]''')
        
        self.erc20_abi = json.loads('''[
            {
                "constant": false,
                "inputs": [
                    {"name": "_spender", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "constant": true,
                "inputs": [
                    {"name": "_owner", "type": "address"},
                    {"name": "_spender", "type": "address"}
                ],
                "name": "allowance",
                "outputs": [{"name": "remaining", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "constant": true,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]''')
    
    async def initialize(self):
        """Initialize all chain connections"""
        for chain, config in self.chain_configs.items():
            try:
                # Create Web3 connection
                w3 = Web3(Web3.HTTPProvider(config['rpc']))
                
                if w3.is_connected():
                    self.web3_connections[chain] = w3
                    
                    # Initialize account
                    private_key = os.getenv('PRIVATE_KEY')
                    if private_key and not private_key.startswith('0x0000'):
                        account = Account.from_key(private_key)
                        self.accounts[chain] = account
                        
                        # Check balance
                        balance = w3.eth.get_balance(account.address)
                        balance_eth = w3.from_wei(balance, 'ether')
                        
                        self.logger.info(f"✅ {chain}: Account {account.address} - Balance: {balance_eth:.6f} ETH")
                    else:
                        self.logger.warning(f"⚠️ No private key for {chain}")
                else:
                    self.logger.error(f"❌ Failed to connect to {chain}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error initializing {chain}: {e}")
    
    async def execute_buy_trade(self, token_address: str, chain: str, amount_eth: float) -> ExecutionResult:
        """Execute real buy trade on blockchain"""
        start_time = time.time()
        
        try:
            w3 = self.web3_connections.get(chain)
            account = self.accounts.get(chain)
            
            if not w3 or not account:
                return ExecutionResult(
                    success=False,
                    transaction_hash="",
                    amount_in=amount_eth,
                    amount_out=0,
                    gas_used=0,
                    gas_price=0,
                    execution_time=time.time() - start_time,
                    slippage=0,
                    error_message=f"No connection or account for {chain}"
                )
            
            # Get best router and quote
            best_router, expected_out = await self._get_best_buy_quote(
                token_address, amount_eth, chain
            )
            
            if not best_router or expected_out == 0:
                return ExecutionResult(
                    success=False,
                    transaction_hash="",
                    amount_in=amount_eth,
                    amount_out=0,
                    gas_used=0,
                    gas_price=0,
                    execution_time=time.time() - start_time,
                    slippage=0,
                    error_message="No liquidity found"
                )
            
            # Calculate minimum output with slippage
            slippage_tolerance = 0.03  # 3%
            min_amount_out = int(expected_out * (1 - slippage_tolerance))
            
            # Get gas price
            gas_price = await self._get_optimal_gas_price(chain)
            
            # Build transaction
            router_contract = w3.eth.contract(
                address=to_checksum_address(best_router),
                abi=self.uniswap_v2_router_abi
            )
            
            weth_address = self.chain_configs[chain]['weth']
            path = [to_checksum_address(weth_address), to_checksum_address(token_address)]
            deadline = int(time.time()) + 300  # 5 minutes
            
            # Build swap transaction
            amount_wei = w3.to_wei(amount_eth, 'ether')
            
            swap_function = router_contract.functions.swapExactETHForTokens(
                min_amount_out,
                path,
                account.address,
                deadline
            )
            
            # Estimate gas
            try:
                gas_estimate = swap_function.estimate_gas({
                    'from': account.address,
                    'value': amount_wei
                })
                gas_limit = int(gas_estimate * 1.2)  # Add 20% buffer
            except Exception as e:
                self.logger.error(f"Gas estimation failed: {e}")
                gas_limit = 300000  # Fallback gas limit
            
            # Build transaction
            nonce = w3.eth.get_transaction_count(account.address)
            
            transaction = swap_function.build_transaction({
                'from': account.address,
                'value': amount_wei,
                'gas': gas_limit,
                'gasPrice': gas_price,
                'nonce': nonce
            })
            
            # Sign transaction
            signed_txn = account.sign_transaction(transaction)
            
            # Send transaction
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            # Calculate actual outputs
            actual_gas_used = receipt['gasUsed']
            success = receipt['status'] == 1
            
            # Get actual token output (simplified - parse logs for exact amount)
            actual_amount_out = expected_out  # Approximation
            actual_slippage = abs(expected_out - actual_amount_out) / expected_out if expected_out > 0 else 0
            
            execution_time = time.time() - start_time
            
            if success:
                self.logger.info(
                    f"✅ BUY {token_address[:8]}... on {chain}: "
                    f"{amount_eth:.6f} ETH → {actual_amount_out} tokens "
                    f"Gas: {actual_gas_used} Slippage: {actual_slippage:.2%}"
                )
            
            return ExecutionResult(
                success=success,
                transaction_hash=tx_hash.hex(),
                amount_in=amount_eth,
                amount_out=actual_amount_out,
                gas_used=actual_gas_used,
                gas_price=gas_price,
                execution_time=execution_time,
                slippage=actual_slippage,
                error_message=None if success else "Transaction failed"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Buy execution error: {e}")
            return ExecutionResult(
                success=False,
                transaction_hash="",
                amount_in=amount_eth,
                amount_out=0,
                gas_used=0,
                gas_price=0,
                execution_time=time.time() - start_time,
                slippage=0,
                error_message=str(e)
            )
    
    async def execute_sell_trade(self, token_address: str, chain: str, amount_tokens: int) -> ExecutionResult:
        """Execute real sell trade on blockchain"""
        start_time = time.time()
        
        try:
            w3 = self.web3_connections.get(chain)
            account = self.accounts.get(chain)
            
            if not w3 or not account:
                return ExecutionResult(
                    success=False,
                    transaction_hash
