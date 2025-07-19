
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import os
import time
import json
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
import requests
from typing import Dict, Optional, Tuple
import asyncio

class RealDEXExecutor:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(os.getenv('RPC_URL')))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        self.account = Account.from_key(os.getenv('PRIVATE_KEY'))
        self.wallet_address = self.account.address
        
        self.uniswap_v2_router = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        self.uniswap_v3_router = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
        self.weth_address = "0xC02aaA39b223FE8610E0c426Dc3d83D0F6b24d3E"
        
        self.flashbots_relay = "https://relay.flashbots.net"
        
        with open('abi/uniswap_router.json', 'r') as f:
            self.router_abi = json.load(f)
            
        self.router_contract = self.web3.eth.contract(
            address=self.uniswap_v2_router, 
            abi=self.router_abi
        )

    def get_gas_price(self) -> int:
        try:
            response = requests.get('https://api.ethgasstation.info/api/ethgasAPI.json')
            gas_data = response.json()
            fast_gas = int(gas_data['fast'] / 10) * 10**9
            return min(fast_gas, 150 * 10**9)
        except:
            return self.web3.eth.gas_price

    def check_wallet_balance(self) -> Tuple[float, float]:
        eth_balance = self.web3.eth.get_balance(self.wallet_address)
        eth_balance_formatted = self.web3.from_wei(eth_balance, 'ether')
        
        min_gas_reserve = 0.01
        available_eth = max(0, float(eth_balance_formatted) - min_gas_reserve)
        
        return float(eth_balance_formatted), available_eth

    def get_token_balance(self, token_address: str) -> float:
        token_contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=[{
                "inputs": [{"name": "account", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"type": "uint256"}],
                "type": "function"
            }, {
                "inputs": [],
                "name": "decimals", 
                "outputs": [{"type": "uint8"}],
                "type": "function"
            }]
        )
        
        balance = token_contract.functions.balanceOf(self.wallet_address).call()
        decimals = token_contract.functions.decimals().call()
        
        return balance / (10 ** decimals)

    def estimate_gas_limit(self, tx_data: dict) -> int:
        try:
            estimated = self.web3.eth.estimate_gas(tx_data)
            return int(estimated * 1.2)
        except:
            return 300000

    def build_swap_transaction(self, token_in: str, token_out: str, amount_in: int, min_amount_out: int) -> dict:
        nonce = self.web3.eth.get_transaction_count(self.wallet_address)
        gas_price = self.get_gas_price()
        deadline = int(time.time()) + 300
        
        if token_in == self.weth_address:
            tx_data = self.router_contract.functions.swapExactETHForTokens(
                min_amount_out,
                [token_in, token_out],
                self.wallet_address,
                deadline
            ).build_transaction({
                'from': self.wallet_address,
                'value': amount_in,
                'gas': 0,
                'gasPrice': gas_price,
                'nonce': nonce
            })
        else:
            tx_data = self.router_contract.functions.swapExactTokensForETH(
                amount_in,
                min_amount_out,
                [token_in, token_out],
                self.wallet_address,
                deadline
            ).build_transaction({
                'from': self.wallet_address,
                'gas': 0,
                'gasPrice': gas_price,
                'nonce': nonce
            })
        
        tx_data['gas'] = self.estimate_gas_limit(tx_data)
        return tx_data

    def submit_flashbots_bundle(self, transactions: list) -> Optional[str]:
        try:
            bundle_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_sendBundle",
                "params": [{
                    "txs": [tx.rawTransaction.hex() for tx in transactions],
                    "blockNumber": hex(self.web3.eth.block_number + 1)
                }]
            }
            
            response = requests.post(
                self.flashbots_relay,
                json=bundle_data,
                headers={'Content-Type': 'application/json'}
            )
            
            return response.json().get('result')
        except:
            return None

    def execute_trade(self, token_address: str, trade_type: str, amount_usd: float) -> Optional[str]:
        total_balance, available_balance = self.check_wallet_balance()
        
        if available_balance < 0.001:
            return None
            
        token_address = Web3.to_checksum_address(token_address)
        
        if trade_type == 'buy':
            eth_amount = min(amount_usd / self.get_eth_price(), available_balance)
            amount_in = self.web3.to_wei(eth_amount, 'ether')
            
            amounts_out = self.router_contract.functions.getAmountsOut(
                amount_in, [self.weth_address, token_address]
            ).call()
            
            min_amount_out = int(amounts_out[1] * 0.97)
            
            tx_data = self.build_swap_transaction(
                self.weth_address, token_address, amount_in, min_amount_out
            )
            
        else:
            token_balance = self.get_token_balance(token_address)
            if token_balance == 0:
                return None
                
            amount_in = int(token_balance * 0.99 * (10 ** 18))
            
            amounts_out = self.router_contract.functions.getAmountsOut(
                amount_in, [token_address, self.weth_address]
            ).call()
            
            min_amount_out = int(amounts_out[1] * 0.97)
            
            tx_data = self.build_swap_transaction(
                token_address, self.weth_address, amount_in, min_amount_out
            )
        
        signed_tx = self.account.sign_transaction(tx_data)
        
        flashbots_result = self.submit_flashbots_bundle([signed_tx])
        if flashbots_result:
            return flashbots_result
            
        try:
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            return tx_hash.hex()
        except Exception as e:
            return None

    def get_eth_price(self) -> float:
        try:
            response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd')
            return response.json()['ethereum']['usd']
        except:
            return 2000.0

real_executor = RealDEXExecutor()
