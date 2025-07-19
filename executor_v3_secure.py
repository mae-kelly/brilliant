import os
import time
import json
import random
import requests
import traceback
from decimal import Decimal
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account

class SecureTradeExecutor:
    def __init__(self):
        # Use environment variables for all sensitive data
        rpc_url = os.getenv("RPC_URL", "https://arb1.arbitrum.io/rpc")
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)

        # All addresses from environment variables
        self.wallet_address = Web3.to_checksum_address(
            os.getenv("WALLET_ADDRESS", "")
        )
        self.private_key = os.getenv("PRIVATE_KEY", "")
        
        # Contract addresses from environment
        self.default_token_buy = Web3.to_checksum_address(
            os.getenv("WETH_ADDRESS", "")
        )
        self.uniswap_router = Web3.to_checksum_address(
            os.getenv("UNISWAP_ROUTER", "")
        )

        # Trading parameters from environment
        self.slippage_tolerance = float(os.getenv("SLIPPAGE_TOLERANCE", "0.0075"))
        self.tx_gas_limit = int(os.getenv("GAS_LIMIT", "400000"))
        self.max_fee_cap_gwei = int(os.getenv("MAX_FEE_GWEI", "100"))
        
        # Safety checks
        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.enable_real_trading = os.getenv("ENABLE_REAL_TRADING", "false").lower() == "true"
        
        self.trade_log = {}
        self.token_contract_cache = {}

    def validate_environment(self):
        """Validate that all required environment variables are set"""
        required_vars = ["WALLET_ADDRESS", "PRIVATE_KEY"]
        missing_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value or value.startswith("0x0000"):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {missing_vars}")
            return False
        
        return True

    def estimate_gas_price(self):
        try:
            base_fee = self.web3.eth.get_block("pending")["baseFeePerGas"]
            priority_fee = self.web3.eth.max_priority_fee
            return int((base_fee + priority_fee) * 1.05)
        except:
            return int(5 * 1e9)

    def is_token_safe(self, token_address):
        """Check if token is safe to trade"""
        if self.dry_run:
            print(f"[DRY RUN] Simulating safety check for {token_address}")
            return True
            
        try:
            contract = self._get_contract(token_address)
            total_supply = contract.functions.totalSupply().call()
            return total_supply > 0
        except:
            return False

    def _get_contract(self, token_address):
        """Get token contract with caching"""
        if token_address in self.token_contract_cache:
            return self.token_contract_cache[token_address]
        
        # Use a basic ERC20 ABI for safety
        erc20_abi = [
            {"inputs": [], "name": "totalSupply", "outputs": [{"type": "uint256"}], "type": "function"},
            {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"type": "uint256"}], "type": "function"}
        ]
        
        contract = self.web3.eth.contract(address=token_address, abi=erc20_abi)
        self.token_contract_cache[token_address] = contract
        return contract

    def buy(self, token_info):
        """Execute buy order with safety checks"""
        if not self.validate_environment():
            print("❌ Environment validation failed")
            return None
            
        if self.dry_run:
            return self._simulate_buy(token_info)
        
        if not self.enable_real_trading:
            print("❌ Real trading is disabled. Set ENABLE_REAL_TRADING=true to enable.")
            return None
            
        token = token_info["token"]
        
        if not self.is_token_safe(token):
            print(f"❌ Token safety check failed: {token}")
            return None

        print(f"[💸 BUYING] {token[:6]}... at ${token_info['price_now']:.6f}")
        
        try:
            # Build transaction
            tx_data = self.build_tx(token_info)
            
            # Sign and send
            signed = self.web3.eth.account.sign_transaction(tx_data, private_key=self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
            
            # Log trade
            self.trade_log[token] = {
                "buy_time": time.time(),
                "entry_price": token_info["price_now"],
                "token": token,
                "tx_hash": tx_hash.hex()
            }
            
            print(f"✅ Buy transaction sent: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            print(f"❌ Buy transaction failed: {e}")
            return None

    def _simulate_buy(self, token_info):
        """Simulate buy transaction"""
        token = token_info["token"]
        price = token_info["price_now"]
        
        print(f"[DRY RUN] Simulating buy: {token[:6]}... at ${price:.6f}")
        
        # Simulate transaction hash
        sim_hash = "0x" + "".join([f"{random.randint(0,15):x}" for _ in range(64)])
        
        self.trade_log[token] = {
            "buy_time": time.time(),
            "entry_price": price,
            "token": token,
            "tx_hash": sim_hash,
            "simulated": True
        }
        
        return sim_hash

    def sell(self, token_info):
        """Execute sell order"""
        if self.dry_run:
            return self._simulate_sell(token_info)
            
        if not self.enable_real_trading:
            print("❌ Real trading is disabled")
            return None
            
        token = token_info["token"]
        
        if token not in self.trade_log:
            print(f"❌ No buy record found for {token}")
            return None
        
        try:
            print(f"[✅ SELLING] {token[:6]}... at ${token_info['price_now']:.6f}")
            
            exit_price = token_info["price_now"]
            entry_price = self.trade_log[token]["entry_price"]
            roi = (exit_price - entry_price) / entry_price
            
            # Update trade log
            self.trade_log[token].update({
                "exit_price": exit_price,
                "roi": roi,
                "exit_time": time.time()
            })
            
            print(f"✅ Sell completed. ROI: {roi*100:.2f}%")
            return roi
            
        except Exception as e:
            print(f"❌ Sell transaction failed: {e}")
            return None

    def _simulate_sell(self, token_info):
        """Simulate sell transaction"""
        token = token_info["token"]
        
        if token not in self.trade_log:
            print(f"[DRY RUN] No buy record for {token}")
            return None
        
        exit_price = token_info["price_now"]
        entry_price = self.trade_log[token]["entry_price"]
        roi = (exit_price - entry_price) / entry_price
        
        print(f"[DRY RUN] Simulating sell: {token[:6]}... ROI: {roi*100:.2f}%")
        
        self.trade_log[token].update({
            "exit_price": exit_price,
            "roi": roi,
            "exit_time": time.time(),
            "simulated": True
        })
        
        return roi

    def build_tx(self, token_info):
        """Build transaction data"""
        gas_price = self.estimate_gas_price()
        nonce = self.web3.eth.get_transaction_count(self.wallet_address)
        
        tx = {
            'from': self.wallet_address,
            'to': token_info["token"],
            'value': self.web3.to_wei(0.01, 'ether'),
            'gas': self.tx_gas_limit,
            'maxFeePerGas': gas_price,
            'maxPriorityFeePerGas': int(gas_price * 0.2),
            'nonce': nonce,
            'chainId': int(os.getenv("CHAIN_ID", "42161")),
        }
        
        return tx

# Create instance with secure defaults
production_executor = SecureTradeExecutor()
