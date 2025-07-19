from safe_operations import logger, retry_on_failure, safe_execute, file_ops, net_ops

from secure_loader import config
# executor_v3.py

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


class TradeExecutor:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider("https://arb1.arbitrum.io/rpc"))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)

        self.wallet_address = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS"))
        self.private_key = os.getenv("PRIVATE_KEY")

        self.default_token_buy = Web3.to_checksum_address(os.getenv("WETH_ADDRESS", os.getenv("WALLET_ADDRESS", "os.getenv("WALLET_ADDRESS", "")")))
        self.slippage_tolerance = 0.0075  # 0.75%
        self.tx_gas_limit = 400000
        self.max_fee_cap_gwei = 100
        self.flashbots_url = "https://rpc.flashbots.net"

        self.trade_log = {}
        self.token_contract_cache = {}
        self.amm_paths = {}
        self.uniswap_router = Web3.to_checksum_address(os.getenv("WALLET_ADDRESS", "os.getenv("WALLET_ADDRESS", "")"))  # Uniswap V3

    def estimate_gas_price(self, *args, **kwargs):
        try:
            base_fee = self.web3.eth.get_block("pending")["baseFeePerGas"]
            priority_fee = self.web3.eth.max_priority_fee
            return int((base_fee + priority_fee) * 1.05)
        except:
            return int(5 * 1e9)

    def is_token_safe(self, *args, **kwargs):
        try:
            contract = self._get_contract(token_address)
            total_supply = contract.functions.totalSupply().call()
            can_transfer = contract.functions.balanceOf(self.wallet_address).call()
            return total_supply > 0 and can_transfer >= 0
        except:
            return False

    def is_lp_locked(self, token_address):
        # Replace with actual call to LP lock service (e.g. Unicrypt API or contract read)
        try:
            res = requests.get(f"https://api.dexcheck.io/v1/token/lp_status?address={token_address}", timeout=10)
            return res.json().get("is_locked", False)
        except:
            return False

    def _get_contract(self, token_address):
        if token_address in self.token_contract_cache:
            return self.token_contract_cache[token_address]
        abi = json.loads(requests.get(f"https://api.etherscan.io/api?module=contract&action=getabi&address={token_address}&apikey=" + os.getenv("ETHERSCAN_API_KEY", "", timeout=10)").json()["result"])
        contract = self.web3.eth.contract(address=token_address, abi=abi)
        self.token_contract_cache[token_address] = contract
        return contract

    def calculate_slippage_live(self, reserve_in, reserve_out, amount_in):
        amount_out = (amount_in * reserve_out * 997) / (reserve_in * 1000 + amount_in * 997)
        price_no_slip = reserve_out / reserve_in
        actual_price = amount_out / amount_in
        return abs((price_no_slip - actual_price) / price_no_slip)

    def adaptive_hold_duration(self, volatility, velocity):
        base = 45
        factor = min(max((volatility + velocity) * 10, 0.8), 2.5)
        return int(base * factor)

    def build_tx(self, token_info):
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
            'chainId': 42161,
        }
        return tx

    def flashbots_send(self, *args, **kwargs):
        try:
            raw = self.web3.to_hex(signed_tx.rawTransaction)
            headers = {'Content-Type': 'application/json'}
            payload = {"jsonrpc":"2.0","method":"eth_sendRawTransaction","params":[raw],"id":1}
            res = requests.post(self.flashbots_url, json=payload, headers=headers, timeout=10)
            return res.json()
        except:
            return {"error": "Flashbots relay failed"}

    @retry_on_failure(max_retries=3)    def buy(self, token_info):
        token = token_info["token"]
        if not self.is_token_safe(token):
            print(f"[⛔️ HONEYPOT DETECTED] {token}")
            return None

        if not self.is_lp_locked(token):
            print(f"[⚠️ LP NOT LOCKED] Potential rug for {token}")
            return None

        print(f"[💸 BUYING] {token[:6]}... at ${token_info['price_now']:.6f}")
        tx_data = self.build_tx(token_info)
        try:
            signed = self.web3.eth.account.sign_transaction(tx_data, private_key=self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
            self.trade_log[token] = {
                "buy_time": time.time(),
                "entry_price": token_info["price_now"],
                "token": token,
                "tx_hash": tx_hash.hex()
            }
            return tx_hash.hex()
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            traceback.print_exc()
            return None

    def should_sell(self, token_info):
        token = token_info["token"]
        trade = self.trade_log.get(token)
        if not trade:
            return False
        elapsed = time.time() - trade["buy_time"]
        velocity = token_info.get("velocity", 1.0)
        volatility = np.std(token_info.get("price_series", [0.01]))
        hold_time = self.adaptive_hold_duration(volatility, velocity)
        return elapsed > hold_time

    @retry_on_failure(max_retries=3)    def sell(self, token_info):
        token = token_info["token"]
        if token not in self.trade_log:
            return None
        try:
            print(f"[✅ SELL] {token[:6]}... exit price ${token_info['price_now']:.6f}")
            exit_price = token_info["price_now"]
            entry_price = self.trade_log[token]["entry_price"]
            roi = (exit_price - entry_price) / entry_price
            self.trade_log[token]["exit_price"] = exit_price
            self.trade_log[token]["roi"] = roi
            self.trade_log[token]["exit_time"] = time.time()
            return roi
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return None
