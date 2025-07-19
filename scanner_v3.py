
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

from safe_operations import logger, retry_on_failure, safe_execute, file_ops, net_ops
# scanner_v3.py

import time
import requests
import numpy as np
import hashlib
from collections import deque
from datetime import datetime, timezone
from web3 import Web3


class TokenScanner:
    def __init__(self):
        self.dex_endpoints = {
            "uniswap": "https://api.geckoterminal.com/api/v2/networks/eth/pools",
            "pancakeswap": "https://api.geckoterminal.com/api/v2/networks/bsc/pools",
            "camelot": "https://api.geckoterminal.com/api/v2/networks/arbitrum/pools",
        }
        self.last_prices = {}
        self.price_windows = {}
        self.price_velocities = {}
        self.liquidity_deltas = {}
        self.token_metadata_cache = {}
        self.exponential_decay = 0.92
        self.max_history_len = 12
        self.min_liquidity_usd = 5000
        self.max_token_age_minutes = 15
        self.rolling_tokens = deque(maxlen=5000)
        self.token_archetypes = {}

    def _normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val - min_val + 1e-9)

    def _dex_vector(self, dex):
        return {
            "uniswap": [1, 0, 0],
            "pancakeswap": [0, 1, 0],
            "camelot": [0, 0, 1],
        }.get(dex, [0, 0, 0])

    @retry_on_failure(max_retries=3)    def scan(self):
        breakout_tokens = []

        for dex_name, endpoint in self.dex_endpoints.items():
            try:
                response = requests.get(endpoint, timeout=3)
                pools = response.json().get("data", [])

                for pool in pools:
                    attr = pool.get("attributes", {})
                    token = attr.get("token", {})
                    token_address = token.get("address")

                    if not token_address:
                        continue

                    pool_id = pool.get("id")
                    now = time.time()
                    created = int(attr.get("created_at_timestamp", 0))
                    age = (now - created) / 60.0

                    price = float(attr.get("price_usd", 0))
                    liquidity = float(attr.get("reserve_in_usd", 0))

                    if liquidity < self.min_liquidity_usd or age > self.max_token_age_minutes:
                        continue

                    price_series = self.price_windows.get(token_address, deque(maxlen=self.max_history_len))
                    price_series.append(price)
                    self.price_windows[token_address] = price_series

                    if len(price_series) < self.max_history_len:
                        continue

                    # Calculate price velocity
                    velocity = self._calculate_velocity(price_series)
                    price_change_pct = ((price_series[-1] - price_series[0]) / price_series[0]) * 100

                    if 9.0 <= price_change_pct <= 13.0 and velocity > 1.0:
                        archetype = self.infer_archetype(token_address, price_series)
                        breakout_tokens.append({
                            "token": token_address,
                            "dex": dex_name,
                            "price_now": price,
                            "price_series": list(price_series),
                            "liquidity": liquidity,
                            "delta_pct": price_change_pct,
                            "velocity": velocity,
                            "age": age,
                            "archetype": archetype
                        })

            except Exception as e:
                logger.error(f"Error: {str(e)}")
                continue

        ranked = sorted(breakout_tokens, key=lambda x: (x["velocity"] * x["delta_pct"]), reverse=True)
        return ranked[:30]

    def _calculate_velocity(self, series):
        weights = np.exp(np.linspace(-1., 0., len(series)))
        weights /= weights.sum()
        smoothed = np.dot(series, weights)
        velocity = (series[-1] - smoothed) / (smoothed + 1e-8)
        return velocity

    def infer_archetype(self, token, price_series):
        hash_digest = hashlib.md5(token.encode()).hexdigest()
        entropy = np.std(price_series) / (np.mean(price_series) + 1e-9)
        if entropy > 0.1:
            if velocity := self._calculate_velocity(price_series) > 1.4:
                return "organic breakout"
            elif velocity < -0.5:
                return "early rug"
        elif entropy < 0.03:
            return "flatline"
        return "volatile / unknown"

    def extract_features(self, token_obj):
        p0 = token_obj["price_series"][0]
        p_now = token_obj["price_now"]
        v_now = token_obj["velocity"]
        l_now = token_obj["liquidity"]
        d_pct = token_obj["delta_pct"]
        age = token_obj["age"]
        archetype = token_obj["archetype"]

        volatility = np.std(token_obj["price_series"])
        price_diff = p_now - p0
        dex_vec = self._dex_vector(token_obj["dex"])

        archetype_flags = {
            "organic breakout": [1, 0, 0],
            "early rug": [0, 1, 0],
            "flatline": [0, 0, 1]
        }.get(archetype, [0, 0, 0])

        feature_vector = [
            p_now, price_diff, v_now,
            l_now, d_pct, age,
            volatility
        ] + dex_vec + archetype_flags

        return feature_vector
