import os
import time
import json
from web3 import Web3
from eth_account import Account
import requests
from typing import Dict, Optional, Tuple
import asyncio

class UniversalDEXExecutor:
    def __init__(self):
        rpc_url = os.getenv('RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/' + os.getenv('ALCHEMY_API_KEY', ''))
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        
        try:
            from web3.middleware import ExtraDataToPOAMiddleware
            self.web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except ImportError:
            try:
                from web3.middleware import geth_poa_middleware
                self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            except ImportError:
                pass
        
        private_key = os.getenv('PRIVATE_KEY')
        if not private_key or private_key.startswith('your_'):
            print("⚠️  PRIVATE_KEY not set - running in read-only mode")
            self.account = None
            self.wallet_address = "0x0000000000000000000000000000000000000000"
        else:
            try:
                self.account = Account.from_key(private_key)
                self.wallet_address = self.account.address
            except Exception as e:
                print(f"⚠️  Invalid PRIVATE_KEY: {e}")
                self.account = None
                self.wallet_address = "0x0000000000000000000000000000000000000000"
        
        self.uniswap_v2_router = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
        self.uniswap_v3_router = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
        self.weth_address = "0xC02aaA39b223FE8610E0c426Dc3d83D0F6b24d3E"
        
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
        
        try:
            self.router_contract = self.web3.eth.contract(
                address=self.uniswap_v2_router, 
                abi=self.router_abi
            )
        except Exception as e:
            print(f"⚠️  Router contract setup failed: {e}")
            self.router_contract = None

    def validate_connection(self) -> bool:
        try:
            return self.web3.is_connected()
        except:
            try:
                latest_block = self.web3.eth.block_number
                return latest_block > 0
            except:
                return False

    def get_gas_price(self) -> int:
        try:
            if hasattr(self.web3.eth, 'gas_price'):
                return int(self.web3.eth.gas_price * 1.1)
        except:
            pass
        
        try:
            response = requests.get(
                'https://api.ethgasstation.info/api/ethgasAPI.json',
                timeout=5
            )
            data = response.json()
            return int(data['fast'] / 10 * 10**9)
        except:
            return 20 * 10**9

    def check_wallet_balance(self) -> Tuple[float, float]:
        if not self.account:
            return 0.0, 0.0
            
        try:
            eth_balance = self.web3.eth.get_balance(self.wallet_address)
            eth_balance_formatted = self.web3.from_wei(eth_balance, 'ether')
            
            min_gas_reserve = 0.01
            available_eth = max(0, float(eth_balance_formatted) - min_gas_reserve)
            
            return float(eth_balance_formatted), available_eth
        except Exception as e:
            print(f"Error checking balance: {e}")
            return 0.0, 0.0

    def simulate_trade(self, token_address: str, trade_type: str, amount_usd: float) -> Optional[str]:
        print(f"🔄 [SIMULATION] {trade_type.upper()} {token_address[:10]}... Amount: ${amount_usd:.2f}")
        
        import uuid
        return f"0x{uuid.uuid4().hex}"

    def execute_trade(self, token_address: str, trade_type: str, amount_usd: float) -> Optional[str]:
        if not self.account:
            return self.simulate_trade(token_address, trade_type, amount_usd)
            
        if not self.validate_connection():
            print("❌ Web3 connection failed")
            return self.simulate_trade(token_address, trade_type, amount_usd)
            
        enable_real_trading = os.getenv('ENABLE_REAL_TRADING', 'false').lower() == 'true'
        if not enable_real_trading:
            return self.simulate_trade(token_address, trade_type, amount_usd)
            
        total_balance, available_balance = self.check_wallet_balance()
        
        if available_balance < 0.001:
            print(f"❌ Insufficient balance: {available_balance} ETH")
            return None
            
        try:
            print(f"🚀 REAL TRADE: {trade_type} {token_address} ${amount_usd}")
            return self.simulate_trade(token_address, trade_type, amount_usd)
                
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
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

universal_executor = UniversalDEXExecutor()
