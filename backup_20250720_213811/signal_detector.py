import aiohttp
import pandas as pd
from web3 import Web3
from abi import abi
import numpy as np
import logging
import redis
import pickle
from prometheus_client import Gauge, Histogram
import time
import sqlite3
from sklearn.cluster import KMeans
import asyncio
import json
import yaml
import os

from infrastructure.monitoring.error_handler import infrastructure.monitoring.error_handler, log_performance, CircuitBreaker, safe_execute
from infrastructure.monitoring.error_handler import infrastructure.monitoring.error_handler, NetworkError, ModelInferenceError

class SignalDetector:
    def __init__(self, chains, redis_client):
        self.chains = chains
        self.redis_client = redis_client
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)
        self.dex_endpoints = self.settings['dex_endpoints']
        self.conn = sqlite3.connect('token_cache.db')
        self.momentum_threshold = self.settings['trading']['momentum_threshold']
        self.decay_threshold = self.settings['trading']['decay_threshold']
        self.breakout_timeframe = self.settings['trading']['breakout_timeframe']
        self.min_volume_spike = self.settings['trading']['min_volume_spike']
        self.velocity_threshold = self.settings['trading']['velocity_threshold']
        self.liquidity_gauge = Gauge('pool_liquidity_usd', 'Pool liquidity in USD', ['chain', 'pool'])
        self.scan_latency = Histogram('scan_latency_seconds', 'Token scan latency', ['chain'])

    async def fetch_dex_data(self, endpoint, query, chain, pool_id):
        cache_key = f"{chain}:{pool_id}:subgraph"
        cached = self.redis_client.get(cache_key)
        if cached:
            return pickle.loads(cached)
        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                try:
                    async with session.post(endpoint, json={'query': query}, timeout=10) as resp:
                        if resp.status != 200:
                            raise Exception(f"GraphQL query failed: {resp.status}")
                        data = await resp.json()
                        ttl = self.calculate_ttl(chain, pool_id)
                        self.redis_client.setex(cache_key, ttl, pickle.dumps(data))
                        return data
                except Exception as e:
                    if attempt == 2:
                        logging.error(json.dumps({
                            'event': 'subgraph_query_error',
                            'chain': chain,
                            'pool': pool_id,
                            'error': str(e)
                        }))
                        return {}
                    await asyncio.sleep(2 ** attempt)

    def calculate_ttl(self, chain, pool_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT liquidity FROM tokens WHERE address = ?", (pool_id,))
        result = cursor.fetchone()
        volatility = self.get_volatility(chain, pool_id)
        if volatility < 0.1:
            return 300
        elif volatility < 0.5:
            return 60
        else:
            return 30

    def select_optimal_chains(self):
        chain_scores = {}
        for chain, w3 in self.chains.items():
            try:
                gas_price = w3.eth.gas_price / 1e9
                chain_scores[chain] = 1 / (gas_price + 0.01)
            except:
                chain_scores[chain] = 0
        return sorted(chain_scores, key=chain_scores.get, reverse=True)[:2]

    async def scan_tokens(self, chain):
        start_time = time.time()
        try:
            endpoints = self.dex_endpoints.get(chain, {})
            if not endpoints:
                return []
            query = """
            query($skip: Int!) {
              pools(first: 100, skip: $skip, where: {volumeUSD_gt: 1000000, liquidity_gt: 250000}) {
                id
                token0 { id symbol decimals }
                token1 { id symbol decimals }
                volumeUSD
                liquidity
                tick
                sqrtPriceX96
                feeTier
                swaps(first: 100, orderBy: timestamp, orderDirection: desc) {
                  amount0
                  amount1
                  timestamp
                  amountUSD
                }
              }
            }
            """
            tokens = []
            skip = 0
            endpoint_url = list(endpoints.values())[0]
            while len(tokens) < self.settings["scanning"]["max_tokens_per_scan"]:
                data = await self.fetch_dex_data(endpoint_url, query, chain, f"batch_{skip}")
                pools = data.get('data', {}).get('pools', [])
                if not pools:
                    break
                
                tasks = []
                for pool in pools:
                    tasks.append(self.process_pool(chain, pool))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, dict) and result.get('valid'):
                        tokens.append(result)
                
                skip += 100
                if len(tokens) >= self.settings["scanning"]["max_tokens_per_scan"]:
                    break
                    
            self.scan_latency.labels(chain=chain).observe(time.time() - start_time)
            return tokens
        except Exception as e:
            logging.error(json.dumps({
                'event': 'scan_tokens_error',
                'chain': chain,
                'error': str(e)
            }))
            return []

    async def process_pool(self, chain, pool):
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT blacklisted FROM tokens WHERE address = ?", (pool['id'],))
            result = cursor.fetchone()
            if result and result[0]:
                return {'valid': False}
            
            price_data = await self.get_price_movement(chain, pool['id'])
            if price_data.empty:
                return {'valid': False}
                
            features = self.engineer_features(price_data, pool)
            
            self.liquidity_gauge.labels(chain=chain, pool=pool['id']).set(float(pool['liquidity']))
            
            if self.is_breakout(features, pool):
                cursor.execute("""INSERT OR REPLACE INTO tokens 
                              (address, symbol, liquidity, blacklisted, last_updated) 
                              VALUES (?, ?, ?, ?, ?)""",
                              (pool['id'], pool['token0']['symbol'], 
                               float(pool['liquidity']), False, int(time.time())))
                self.conn.commit()
                
                return {
                    'valid': True,
                    'address': pool['id'],
                    'data': features,
                    'symbol': f"{pool['token0']['symbol']}/{pool['token1']['symbol']}",
                    'decimals': pool['token0']['decimals'],
                    'feeTier': pool['feeTier'],
                    'swap_volume': self.get_swap_volume(pool),
                    'momentum_score': self.calculate_momentum_score(features),
                    'velocity': self.calculate_velocity(features),
                    'volume_spike': self.detect_volume_spike(pool)
                }
            return {'valid': False}
        except Exception as e:
            logging.error(json.dumps({
                'event': 'process_pool_error',
                'chain': chain,
                'pool': pool.get('id', 'unknown'),
                'error': str(e)
            }))
            return {'valid': False}

    def is_breakout(self, features, pool):
        if features.empty or len(features) < 10:
            return False
        
        recent_returns = features['returns'].tail(self.breakout_timeframe)
        cumulative_return = recent_returns.sum()
        velocity = self.calculate_velocity(features)
        volume_spike = self.detect_volume_spike(pool)
        momentum_score = self.calculate_momentum_score(features)
        
        breakout_conditions = [
            cumulative_return >= self.momentum_threshold,
            velocity >= self.velocity_threshold,
            volume_spike >= self.min_volume_spike,
            momentum_score > 0.7,
            features['volatility'].iloc[-1] > features['volatility'].mean() * 1.5
        ]
        
        return sum(breakout_conditions) >= 3

    def calculate_momentum_score(self, features):
        if features.empty:
            return 0.0
        
        short_ma = features['returns'].rolling(5).mean().iloc[-1]
        long_ma = features['returns'].rolling(20).mean().iloc[-1]
        volatility_ratio = features['volatility'].iloc[-1] / features['volatility'].mean()
        volume_momentum = features['swap_volume'].iloc[-1] / features['swap_volume'].mean() if 'swap_volume' in features else 1.0
        
        score = (short_ma - long_ma) * volatility_ratio * np.log(volume_momentum + 1)
        return max(0, min(1, score))

    def calculate_velocity(self, features):
        if features.empty or len(features) < 5:
            return 0.0
        
        price_changes = features['returns'].tail(5)
        time_weights = np.exp(np.linspace(-1, 0, len(price_changes)))
        weighted_velocity = np.sum(price_changes * time_weights) / np.sum(time_weights)
        
        return abs(weighted_velocity)

    def detect_volume_spike(self, pool):
        try:
            recent_swaps = pool.get('swaps', [])[:10]
            older_swaps = pool.get('swaps', [])[10:50]
            
            if not recent_swaps or not older_swaps:
                return 1.0
            
            recent_volume = sum(float(swap.get('amountUSD', 0)) for swap in recent_swaps)
            baseline_volume = sum(float(swap.get('amountUSD', 0)) for swap in older_swaps) / 4
            
            return recent_volume / (baseline_volume + 1) if baseline_volume > 0 else 1.0
        except:
            return 1.0

    async def get_price_movement(self, chain, pool_address):
        try:
            contract = self.chains[chain].eth.contract(address=pool_address, abi=UNISWAP_V3_POOL_ABI)
            slot0 = contract.functions.slot0().call()
            token0 = contract.functions.token0().call()
            
            token0_contract = self.chains[chain].eth.contract(address=token0, abi=[{
                'constant': True,
                'inputs': [],
                'name': 'decimals',
                'outputs': [{'name': '', 'type': 'uint8'}],
                'stateMutability': 'view',
                'type': 'function'
            }])
            decimals = token0_contract.functions.decimals().call()
            
            current_price = (slot0[0] / 2**96)**2 / 10**(18 - decimals)
            historical_prices = await self.fetch_historical_prices(chain, pool_address)
            
            all_prices = historical_prices + [current_price]
            price_series = pd.Series(all_prices).pct_change().dropna()
            
            return price_series
        except Exception as e:
            logging.error(json.dumps({
                'event': 'price_movement_error',
                'chain': chain,
                'pool': pool_address,
                'error': str(e)
            }))
            return pd.Series()

    async def fetch_historical_prices(self, chain, pool_address):
        try:
            cache_key = f"{chain}:{pool_address}:historical_prices"
            cached = self.redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)
            
            query = """
            query($pool: ID!, $count: Int!) {
              pool(id: $pool) {
                poolHourData(first: $count, orderBy: periodStartUnix, orderDirection: desc) {
                  high
                  low
                  close
                  periodStartUnix
                }
              }
            }
            """
            variables = {'pool': pool_address.lower(), 'count': 120}
            endpoint = list(self.dex_endpoints[chain].values())[0]
            data = await self.fetch_dex_data(endpoint, query, chain, pool_address)
            
            if 'data' in data and 'pool' in data['data'] and data['data']['pool']:
                price_data = data['data']['pool']['poolHourData']
                prices = [float(item['close']) for item in price_data if item['close']]
            else:
                prices = [100.0 + np.random.normal(0, 5) for _ in range(120)]
            
            ttl = self.calculate_ttl(chain, pool_address)
            self.redis_client.setex(cache_key, ttl, pickle.dumps(prices))
            return prices
        except Exception as e:
            logging.error(json.dumps({
                'event': 'historical_prices_error',
                'chain': chain,
                'pool': pool_address,
                'error': str(e)
            }))
            return [100.0 + np.random.normal(0, 2) for _ in range(120)]

    def get_swap_volume(self, pool):
        try:
            swaps = pool.get('swaps', [])
            return sum(float(swap.get('amountUSD', 0)) for swap in swaps)
        except:
            return 1000.0

    def get_volatility(self, chain, pool_address):
        try:
            prices = asyncio.run(self.fetch_historical_prices(chain, pool_address))
            returns = pd.Series(prices).pct_change().dropna()
            return returns.std() * np.sqrt(252)
        except:
            return 0.3

    def detect_whale_activity(self, pool_address, swaps):
        try:
            if not swaps:
                return 0.0
            
            total_volume = sum(float(swap.get('amountUSD', 0)) for swap in swaps)
            avg_volume = total_volume / len(swaps) if swaps else 1
            large_tx_threshold = avg_volume * 10
            
            whale_txs = sum(1 for swap in swaps 
                           if float(swap.get('amountUSD', 0)) > large_tx_threshold)
            
            return whale_txs / len(swaps) if swaps else 0.0
        except:
            return 0.0

    def engineer_features(self, price_data, pool):
        if price_data.empty:
            return pd.DataFrame()
        
        features = pd.DataFrame()
        features['returns'] = price_data
        features['volatility'] = price_data.rolling(10, min_periods=1).std().fillna(0)
        features['momentum'] = (price_data.rolling(5, min_periods=1).mean() - 
                               price_data.rolling(20, min_periods=1).mean()).fillna(0)
        features['rsi'] = self.calculate_rsi(price_data)
        features['bb_position'] = self.calculate_bollinger_position(price_data)
        features['volume_ma'] = pd.Series([self.get_swap_volume(pool)] * len(price_data))
        features['whale_activity'] = pd.Series([self.detect_whale_activity(pool['id'], pool.get('swaps', []))] * len(price_data))
        features['price_acceleration'] = price_data.diff().rolling(3, min_periods=1).mean().fillna(0)
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(20, min_periods=1).mean().fillna(1)
        features['momentum_strength'] = abs(features['momentum']) * features['volatility_ratio']
        features['swap_volume'] = features['volume_ma']
        
        return features.dropna()

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_bollinger_position(self, prices, window=20):
        ma = prices.rolling(window=window, min_periods=1).mean()
        std = prices.rolling(window=window, min_periods=1).std()
        upper = ma + (std * 2)
        lower = ma - (std * 2)
        bb_position = (prices - lower) / (upper - lower).replace(0, 1)
        return bb_position.fillna(0.5)

    async def detect_market_regime(self):
        try:
            all_returns = []
            for chain in self.chains:
                query = """
                query {
                  pools(first: 10, orderBy: volumeUSD, orderDirection: desc) {
                    id
                    poolHourData(first: 24, orderBy: periodStartUnix, orderDirection: desc) {
                      close
                    }
                  }
                }
                """
                endpoint = list(self.dex_endpoints[chain].values())[0]
                data = await self.fetch_dex_data(endpoint, query, chain, 'market_regime')
                
                for pool in data.get('data', {}).get('pools', []):
                    prices = [float(hour['close']) for hour in pool.get('poolHourData', []) if hour.get('close')]
                    if len(prices) >= 2:
                        returns = pd.Series(prices).pct_change().dropna()
                        all_returns.extend(returns.tolist())
            
            if not all_returns:
                return 'normal'
            
            returns_series = pd.Series(all_returns)
            volatility = returns_series.std() * np.sqrt(24)
            mean_return = returns_series.mean()
            
            if volatility > 0.4:
                return 'extreme_volatility'
            elif mean_return > 0.02:
                return 'bull'
            elif mean_return < -0.02:
                return 'bear'
            else:
                return 'normal'
                
        except Exception as e:
            logging.error(json.dumps({
                'event': 'market_regime_error',
                'error': str(e)
            }))
            return 'normal'

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass