
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import os
import time
import asyncio
from web3 import Web3
from eth_account import Account
from decimal import Decimal
import json
import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

@dataclass
class TradeResult:
    success: bool
    tx_hash: str
    gas_used: int
    profit_loss: float
    execution_time: float

class RealDEXExecutor:
    def __init__(self):
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        self.enable_real_trading = os.getenv('ENABLE_REAL_TRADING', 'false').lower() == 'true'
        
        rpc_endpoints = {
            'arbitrum': 'https://arb1.arbitrum.io/rpc',
            'polygon': 'https://polygon-rpc.com',
            'optimism': 'https://mainnet.optimism.io',
            'ethereum': 'https://ethereum.publicnode.com'
        }
        
        self.chains = {}
        for chain, rpc in rpc_endpoints.items():
            try:
                w3 = Web3(Web3.HTTPProvider(rpc))
                if w3.is_connected():
                    self.chains[chain] = w3
                    logging.info(f"✅ Connected to {chain}")
                else:
                    logging.warning(f"⚠️ Failed to connect to {chain}")
            except Exception as e:
                logging.error(f"❌ {chain} connection error: {e}")
        
        self.wallet_address = os.getenv('WALLET_ADDRESS', '0x0000000000000000000000000000000000000000')
        self.private_key = os.getenv('PRIVATE_KEY', '0x0000000000000000000000000000000000000000000000000000000000000000')
        
        self.router_addresses = {
            'arbitrum': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',
            'polygon': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
            'optimism': '0x4A7b5Da61326A6379179b40d00F57E5bbDC962c2',
            'ethereum': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D'
        }
        
        self.weth_addresses = {
            'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
            'polygon': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
            'optimism': '0x4200000000000000000000000000000000000006',
            'ethereum': '0xC02aaA39b223FE8dD0e0e3C4c4c4c4c4c4c4c4c4'
        }
        
        self.simulation_mode = not self.enable_real_trading or self.dry_run
        self.trade_count = 0
        self.total_profit = 0.0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_router_abi(self) -> list:
        return [
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

    def get_erc20_abi(self) -> list:
        return [
            {
                "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]

    async def execute_buy_trade(self, token_address: str, chain: str, amount_eth: float) -> TradeResult:
        start_time = time.time()
        
        if self.simulation_mode:
            return await self.simulate_buy(token_address, chain, amount_eth, start_time)
        
        if chain not in self.chains:
            self.logger.error(f"Chain {chain} not available")
            return TradeResult(False, "", 0, 0.0, time.time() - start_time)
        
        try:
            w3 = self.chains[chain]
            router_address = self.router_addresses[chain]
            weth_address = self.weth_addresses[chain]
            
            router_contract = w3.eth.contract(
                address=Web3.to_checksum_address(router_address),
                abi=self.get_router_abi()
            )
            
            amount_in = w3.to_wei(amount_eth, 'ether')
            path = [Web3.to_checksum_address(weth_address), Web3.to_checksum_address(token_address)]
            deadline = int(time.time()) + 300
            
            amounts_out = router_contract.functions.getAmountsOut(amount_in, path).call()
            min_tokens_out = int(amounts_out[1] * 0.97)
            
            swap_txn = router_contract.functions.swapExactETHForTokens(
                min_tokens_out,
                path,
                Web3.to_checksum_address(self.wallet_address),
                deadline
            ).build_transaction({
                'from': Web3.to_checksum_address(self.wallet_address),
                'value': amount_in,
                'gas': 300000,
                'gasPrice': w3.eth.gas_price,
                'nonce': w3.eth.get_transaction_count(Web3.to_checksum_address(self.wallet_address))
            })
            
            signed_txn = w3.eth.account.sign_transaction(swap_txn, private_key=self.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            success = receipt['status'] == 1
            gas_used = receipt['gasUsed']
            
            self.logger.info(f"🟢 Buy executed: {token_address[:8]}... Gas: {gas_used}")
            
            return TradeResult(
                success=success,
                tx_hash=tx_hash.hex(),
                gas_used=gas_used,
                profit_loss=-amount_eth,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ Buy execution failed: {e}")
            return TradeResult(False, "", 0, 0.0, time.time() - start_time)

    async def execute_sell_trade(self, token_address: str, chain: str, token_amount: int) -> TradeResult:
        start_time = time.time()
        
        if self.simulation_mode:
            return await self.simulate_sell(token_address, chain, token_amount, start_time)
        
        if chain not in self.chains:
            self.logger.error(f"Chain {chain} not available")
            return TradeResult(False, "", 0, 0.0, time.time() - start_time)
        
        try:
            w3 = self.chains[chain]
            router_address = self.router_addresses[chain]
            weth_address = self.weth_addresses[chain]
            
            token_contract = w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self.get_erc20_abi()
            )
            
            approve_txn = token_contract.functions.approve(
                Web3.to_checksum_address(router_address),
                token_amount
            ).build_transaction({
                'from': Web3.to_checksum_address(self.wallet_address),
                'gas': 100000,
                'gasPrice': w3.eth.gas_price,
                'nonce': w3.eth.get_transaction_count(Web3.to_checksum_address(self.wallet_address))
            })
            
            signed_approve = w3.eth.account.sign_transaction(approve_txn, private_key=self.private_key)
            approve_hash = w3.eth.send_raw_transaction(signed_approve.rawTransaction)
            w3.eth.wait_for_transaction_receipt(approve_hash, timeout=60)
            
            router_contract = w3.eth.contract(
                address=Web3.to_checksum_address(router_address),
                abi=self.get_router_abi()
            )
            
            path = [Web3.to_checksum_address(token_address), Web3.to_checksum_address(weth_address)]
            deadline = int(time.time()) + 300
            
            amounts_out = router_contract.functions.getAmountsOut(token_amount, path).call()
            min_eth_out = int(amounts_out[1] * 0.97)
            
            swap_txn = router_contract.functions.swapExactTokensForETH(
                token_amount,
                min_eth_out,
                path,
                Web3.to_checksum_address(self.wallet_address),
                deadline
            ).build_transaction({
                'from': Web3.to_checksum_address(self.wallet_address),
                'gas': 300000,
                'gasPrice': w3.eth.gas_price,
                'nonce': w3.eth.get_transaction_count(Web3.to_checksum_address(self.wallet_address))
            })
            
            signed_txn = w3.eth.account.sign_transaction(swap_txn, private_key=self.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            success = receipt['status'] == 1
            gas_used = receipt['gasUsed']
            eth_received = w3.from_wei(amounts_out[1], 'ether')
            
            self.logger.info(f"🔴 Sell executed: {token_address[:8]}... ETH: {eth_received:.6f}")
            
            return TradeResult(
                success=success,
                tx_hash=tx_hash.hex(),
                gas_used=gas_used,
                profit_loss=float(eth_received),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ Sell execution failed: {e}")
            return TradeResult(False, "", 0, 0.0, time.time() - start_time)

    async def simulate_buy(self, token_address: str, chain: str, amount_eth: float, start_time: float) -> TradeResult:
        await asyncio.sleep(0.5)
        
        self.trade_count += 1
        slippage = 0.98 + (0.04 * hash(token_address) % 100) / 10000
        simulated_tokens = int(amount_eth * 1000000 * slippage)
        
        self.logger.info(f"[SIM] 🟢 Buy: {token_address[:8]}... {amount_eth} ETH -> {simulated_tokens} tokens")
        
        return TradeResult(
            success=True,
            tx_hash=f"0x{hash(token_address + str(time.time())) % (16**64):064x}",
            gas_used=250000,
            profit_loss=-amount_eth,
            execution_time=time.time() - start_time
        )

    async def simulate_sell(self, token_address: str, chain: str, token_amount: int, start_time: float) -> TradeResult:
        await asyncio.sleep(0.5)
        
        profit_multiplier = 0.95 + (0.10 * hash(token_address + str(self.trade_count)) % 100) / 100
        eth_received = (token_amount / 1000000) * profit_multiplier
        profit = eth_received - 0.01
        
        self.total_profit += profit
        
        self.logger.info(f"[SIM] 🔴 Sell: {token_address[:8]}... {token_amount} tokens -> {eth_received:.6f} ETH (P&L: {profit:+.6f})")
        
        return TradeResult(
            success=True,
            tx_hash=f"0x{hash(token_address + str(time.time()) + 'sell') % (16**64):064x}",
            gas_used=280000,
            profit_loss=profit,
            execution_time=time.time() - start_time
        )

    def get_performance_stats(self) -> Dict:
        return {
            'total_trades': self.trade_count,
            'total_profit': self.total_profit,
            'avg_profit_per_trade': self.total_profit / max(self.trade_count, 1),
            'simulation_mode': self.simulation_mode,
            'connected_chains': list(self.chains.keys())
        }

real_executor = RealDEXExecutor()

    def get_performance_stats(self) -> dict:
        return {
            'total_trades': self.trade_count,
            'total_profit': self.total_profit,
            'avg_profit_per_trade': self.total_profit / max(self.trade_count, 1),
            'simulation_mode': self.simulation_mode,
            'connected_chains': list(self.chains.keys())
        }

