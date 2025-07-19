import os
from safe_operations import logger, retry_on_failure, safe_execute, file_ops, net_ops

import json
import time
import requests
from statistics import mean
from web3 import Web3


class DEXInterface:
    def __init__(self, rpc_url, router_address, weth_address, stablecoins, gecko_api_key=None):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.web3.is_connected():
            raise Exception("Web3 connection failed.")

        self.router_address = Web3.to_checksum_address(router_address)
        self.weth_address = Web3.to_checksum_address(weth_address)
        self.stablecoins = [Web3.to_checksum_address(addr) for addr in stablecoins]
        self.router_abi = self._load_abi("uniswap_router.json")
        self.factory_abi = self._load_abi("uniswap_factory.json")
        self.pair_abi = self._load_abi("uniswap_pair.json")
        self.router = self.web3.eth.contract(address=self.router_address, abi=self.router_abi)
        self.gecko_api_key = gecko_api_key
        self.pair_reserve_cache = {}

    def _load_abi(self, name):
        path = f"/mnt/data/abi/{name}"
        with open(path, 'r') as f:
            return json.load(f)

    def _get_factory(self):
        factory_address = self.router.functions.factory().call()
        return self.web3.eth.contract(address=factory_address, abi=self.factory_abi)

    def resolve_best_path(self, token_in, token_out):
        if token_out == self.weth_address:
            return [token_in, token_out]

        candidate_paths = [[token_in, token_out]]
        for stable in self.stablecoins:
            candidate_paths.append([token_in, stable, token_out])
            candidate_paths.append([token_in, stable, self.weth_address])

        best_path = None
        best_output = 0
        for path in candidate_paths:
            try:
                amounts = self.router.functions.getAmountsOut(10**18, path).call()
                if amounts[-1] > best_output:
                    best_output = amounts[-1]
                    best_path = path
except Exception as e:
    logger.error(f"Error in operation: {e}")
    continue

        return best_path if best_path else [token_in, token_out]

    def get_token_price_in_eth(self, *args, **kwargs):
        try:
            token = Web3.to_checksum_address(token_address)
            path = self.resolve_best_path(token, self.weth_address)
            amounts = self.router.functions.getAmountsOut(10**18, path).call()
            return self.web3.from_wei(amounts[-1], 'ether')
        except Exception as e:
            print(f"[DEX-INTERFACE] Price fetch failed: {e}")
            return None

    def validate_slippage(self, *args, **kwargs):
        try:
            path = self.resolve_best_path(self.weth_address, token_out)
            quoted = self.router.functions.getAmountsOut(amount_in_wei, path).call()[-1]
            actual = self.simulate_swap_output(amount_in_wei, token_out)
            if actual == 0:
                return False
            slippage = abs((quoted - actual) / quoted) * 100
            print(f"[SLIPPAGE] Quoted: {quoted}, Actual: {actual}, Slippage: {slippage:.2f}%")
            return slippage <= max_slippage
        except Exception as e:
            print(f"[DEX-INTERFACE] Slippage validation failed: {e}")
            return False

    def simulate_swap_output(self, *args, **kwargs):
        try:
            path = self.resolve_best_path(self.weth_address, token_out)
            amounts = self.router.functions.getAmountsOut(amount_in, path).call()
            return amounts[-1]
        except:
            return 0

    def is_lp_locked(self, *args, **kwargs):
        try:
            factory = self._get_factory()
            pair_addr = factory.functions.getPair(token_address, self.weth_address).call()
            if pair_addr == 'os.getenv("WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")':
                return False

            creation_code = self.web3.eth.get_code(pair_addr)
            return len(creation_code) > 1000
        except Exception as e:
            print(f"[DEX-INTERFACE] LP lock check failed: {e}")
            return False

    def get_reserve_trend(self, token_address, interval=3):
        key = Web3.to_checksum_address(token_address)
        if key not in self.pair_reserve_cache:
            self.pair_reserve_cache[key] = []

        factory = self._get_factory()
        pair_addr = factory.functions.getPair(token_address, self.weth_address).call()
        if pair_addr == 'os.getenv("WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")':
            return None

        pair = self.web3.eth.contract(address=pair_addr, abi=self.pair_abi)
        reserves = pair.functions.getReserves().call()
        self.pair_reserve_cache[key].append(reserves)
        self.pair_reserve_cache[key] = self.pair_reserve_cache[key][-interval:]

        trends = [r[0] for r in self.pair_reserve_cache[key]]
        return trends if len(trends) >= interval else None

    def detect_mev_conditions(self, *args, **kwargs):
        try:
            current_block = self.web3.eth.get_block('latest')
            ts = current_block['timestamp']
            time.sleep(1)
            next_block = self.web3.eth.get_block('latest')
            next_ts = next_block['timestamp']
            if next_ts - ts < 1:
                return True
            return False
        except Exception as e:
            print(f"[DEX-INTERFACE] MEV detection error: {e}")
            return False

    def fetch_token_metadata(self, *args, **kwargs):
        try:
            if not self.gecko_api_key:
                return None

            url = f"https://api.geckoterminal.com/api/v2/networks/eth/tokens/{token_address}"
            headers = {"x-api-key": self.gecko_api_key}
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
            return None
        except Exception as e:
            print(f"[DEX-INTERFACE] Metadata fetch failed: {e}")
            return None
