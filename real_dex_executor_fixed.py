import os
import time
import json
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account
import requests
from typing import Dict, Optional, Tuple
import asyncio

class RealDEXExecutor:
    def __init__(self):
        rpc_url = os.getenv('RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/' + os.getenv('ALCHEMY_API_KEY', ''))
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        
        private_key = os.getenv('PRIVATE_KEY')
        if not private_key or private_key.startswith('your_'):
            raise ValueError("PRIVATE_KEY not configured")
            
        self.account = Account.from_key(private_key)
        self.wallet_address = self.account.address
        
        self.uniswap_v2_router = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        self.uniswap_v3_router = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
        self.weth_address = "0xC02aaA39b223FE8610E0c426Dc3d83D0F6b24d3E"
        
        self.flashbots_relay = "https://relay.flashbots.net"
        
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
            
        self.router_contract = self.web3.eth.contract(
            address=self.uniswap_v2_router, 
            abi=self.router_abi
        )

    def validate_connection(self) -> bool:
        try:
            return self.web3.is_connected()
        except:
            return False

    def get_gas_price(self) -> int:
        try:
            if hasattr(self.web3.eth, 'get_block'):
                latest_block = self.web3.eth.get_block('latest')
                base_fee = latest_block.get('baseFeePerGas', 0)
                priority_fee = self.web3.eth.max_priority_fee
                return int((base_fee + priority_fee) * 1.1)
        except:
            pass
        
        try:
            return int(self.web3.eth.gas_price * 1.1)
        except:
            return 20 * 10**9

    def check_wallet_balance(self) -> Tuple[float, float]:
        try:
            eth_balance = self.web3.eth.get_balance(self.wallet_address)
            eth_balance_formatted = self.web3.from_wei(eth_balance, 'ether')
            
            min_gas_reserve = 0.01
            available_eth = max(0, float(eth_balance_formatted) - min_gas_reserve)
            
            return float(eth_balance_formatted), available_eth
        except Exception as e:
            print(f"Error checking balance: {e}")
            return 0.0, 0.0

    def get_token_balance(self, token_address: str) -> float:
        try:
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
        except:
            return 0.0

    def estimate_gas_limit(self, tx_data: dict) -> int:
        try:
            estimated = self.web3.eth.estimate_gas(tx_data)
            return int(estimated * 1.2)
        except:
            return 300000

    def build_swap_transaction(self, token_in: str, token_out: str, amount_in: int, min_amount_out: int) -> dict:
        try:
            nonce = self.web3.eth.get_transaction_count(self.wallet_address)
            gas_price = self.get_gas_price()
            deadline = int(time.time()) + 300
            
            if token_in.lower() == self.weth_address.lower():
                tx_data = self.router_contract.functions.swapExactETHForTokens(
                    min_amount_out,
                    [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)],
                    self.wallet_address,
                    deadline
                ).build_transaction({
                    'from': self.wallet_address,
                    'value': amount_in,
                    'gas': 300000,
                    'gasPrice': gas_price,
                    'nonce': nonce
                })
            else:
                tx_data = self.router_contract.functions.swapExactTokensForETH(
                    amount_in,
                    min_amount_out,
                    [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)],
                    self.wallet_address,
                    deadline
                ).build_transaction({
                    'from': self.wallet_address,
                    'gas': 300000,
                    'gasPrice': gas_price,
                    'nonce': nonce
                })
            
            tx_data['gas'] = self.estimate_gas_limit(tx_data)
            return tx_data
        except Exception as e:
            print(f"Error building transaction: {e}")
            return None

    def execute_trade(self, token_address: str, trade_type: str, amount_usd: float) -> Optional[str]:
        if not self.validate_connection():
            print("Web3 connection failed")
            return None
            
        total_balance, available_balance = self.check_wallet_balance()
        
        if available_balance < 0.001:
            print(f"Insufficient balance: {available_balance} ETH")
            return None
            
        token_address = Web3.to_checksum_address(token_address)
        
        try:
            if trade_type == 'buy':
                eth_price = self.get_eth_price()
                eth_amount = min(amount_usd / eth_price, available_balance)
                amount_in = self.web3.to_wei(eth_amount, 'ether')
                
                amounts_out = self.router_contract.functions.getAmountsOut(
                    amount_in, [self.weth_address, token_address]
                ).call()
                
                min_amount_out = int(amounts_out[1] * 0.95)
                
                tx_data = self.build_swap_transaction(
                    self.weth_address, token_address, amount_in, min_amount_out
                )
                
            else:
                token_balance = self.get_token_balance(token_address)
                if token_balance == 0:
                    print("No token balance to sell")
                    return None
                    
                decimals = 18
                amount_in = int(token_balance * 0.99 * (10 ** decimals))
                
                amounts_out = self.router_contract.functions.getAmountsOut(
                    amount_in, [token_address, self.weth_address]
                ).call()
                
                min_amount_out = int(amounts_out[1] * 0.95)
                
                tx_data = self.build_swap_transaction(
                    token_address, self.weth_address, amount_in, min_amount_out
                )
            
            if not tx_data:
                return None
                
            signed_tx = self.account.sign_transaction(tx_data)
            
            try:
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                return tx_hash.hex()
            except Exception as e:
                print(f"Transaction failed: {e}")
                return None
                
        except Exception as e:
            print(f"Trade execution error: {e}")
            return None

    def get_eth_price(self) -> float:
        try:
            response = requests.get(
                'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd',
                timeout=5
            )
            return response.json()['ethereum']['usd']
        except:
            return 2000.0

real_executor = RealDEXExecutor()
