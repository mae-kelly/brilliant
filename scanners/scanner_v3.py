import asyncio
import aiohttp
import websockets
import json
import time
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
from web3 import Web3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config"))

try:
    from config.optimizer import get_dynamic_config
except ImportError:
    def get_dynamic_config():
        return {
            "momentum_threshold": 0.65,
            "confidence_threshold": 0.75,
            "min_liquidity_threshold": 10000,
            "min_price_change": 9,
            "max_price_change": 15,
            "volatility_threshold": 0.10
        }

@dataclass
class TokenSignal:
    address: str
    chain: str
    symbol: str
    name: str
    price: float
    volume_24h: float
    price_change_24h: float
    momentum_score: float
    liquidity_usd: float
    market_cap: float
    detected_at: float
    confidence: float
    velocity: float
    volatility: float
    dex_source: str

class UltraScanner:
    def __init__(self):
        self.dex_endpoints = {
            'uniswap_v3_ethereum': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
            'uniswap_v2_ethereum': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2',
            'sushiswap_ethereum': 'https://api.thegraph.com/subgraphs/name/sushi-v2/sushiswap-ethereum',
            'uniswap_v3_arbitrum': 'https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-minimal',
            'camelot_arbitrum': 'https://api.thegraph.com/subgraphs/name/camelotlabs/camelot-amm',
            'quickswap_polygon': 'https://api.thegraph.com/subgraphs/name/sameepsi/quickswap06',
            'uniswap_v3_polygon': 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon'
        }
        
        self.chain_configs = {
            'ethereum': {
                'rpc': f"https://eth-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'dexes': ['uniswap_v3_ethereum', 'uniswap_v2_ethereum', 'sushiswap_ethereum'],
                'chain_id': 1
            },
            'arbitrum': {
                'rpc': f"https://arb-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'dexes': ['uniswap_v3_arbitrum', 'camelot_arbitrum'],
                'chain_id': 42161
            },
            'polygon': {
                'rpc': f"https://polygon-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                'dexes': ['quickswap_polygon', 'uniswap_v3_polygon'],
                'chain_id': 137
            }
        }
        
        self.session = None
        self.workers = []
        self.discovered_tokens = set()
        self.signal_queue = asyncio.Queue(maxsize=10000)
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        self.volume_history = defaultdict(lambda: deque(maxlen=100))
        self.running = False
        
        self.stats = {
            'tokens_scanned': 0,
            'signals_generated': 0,
            'api_calls': 0,
            'start_time': time.time(),
            'errors': 0
        }
        
        self.cache = {}
        self.cache_ttl = 30
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
        self.running = True
        
        for i in range(200):
            chain = list(self.chain_configs.keys())[i % len(self.chain_configs)]
            worker = asyncio.create_task(self.worker_loop(i, chain))
            self.workers.append(worker)
        
        asyncio.create_task(self.performance_monitor())
        
        self.logger.info(f"Initialized {len(self.workers)} scanner workers")

    async def worker_loop(self, worker_id: int, chain: str):
        while self.running:
            try:
                await self.scan_chain_tokens(worker_id, chain)
                await asyncio.sleep(0.5)
            except Exception as e:
                self.stats['errors'] += 1
                await asyncio.sleep(2)

    async def scan_chain_tokens(self, worker_id: int, chain: str):
        dexes = self.chain_configs[chain]['dexes']
        
        for dex in dexes:
            try:
                tokens = await self.fetch_tokens_from_dex(dex, chain)
                
                for token in tokens:
                    token_key = f"{chain}_{token['id']}"
                    
                    if token_key not in self.discovered_tokens:
                        self.discovered_tokens.add(token_key)
                        await self.analyze_token_momentum(token, chain, dex)
                        self.stats['tokens_scanned'] += 1
                        
            except Exception as e:
                self.stats['errors'] += 1

    async def fetch_tokens_from_dex(self, dex: str, chain: str) -> List[Dict]:
        endpoint = self.dex_endpoints.get(dex)
        if not endpoint:
            return []
        
        cache_key = f"{dex}_{int(time.time() // self.cache_ttl)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        query = self.build_graphql_query(dex)
        
        try:
            async with self.session.post(
                endpoint,
                json={'query': query},
                headers={'Content-Type': 'application/json'}
            ) as response:
                self.stats['api_calls'] += 1
                
                if response.status == 200:
                    data = await response.json()
                    tokens = self.parse_token_data(data, dex)
                    self.cache[cache_key] = tokens
                    return tokens
                    
        except Exception as e:
            self.stats['errors'] += 1
        
        return []

    def build_graphql_query(self, dex: str) -> str:
        if 'uniswap_v3' in dex:
            return '''
            {
              tokens(
                first: 50
                orderBy: totalValueLockedUSD
                orderDirection: desc
                where: {
                  totalValueLockedUSD_gt: "10000"
                  txCount_gt: "100"
                }
              ) {
                id
                symbol
                name
                totalValueLockedUSD
                volumeUSD
                txCount
                derivedETH
                tokenDayData(first: 7, orderBy: date, orderDirection: desc) {
                  priceUSD
                  volumeUSD
                  totalValueLockedUSD
                }
              }
            }
            '''
        else:
            return '''
            {
              tokens(
                first: 50
                orderBy: totalLiquidity
                orderDirection: desc
                where: {
                  totalLiquidity_gt: "5000"
                }
              ) {
                id
                symbol
                name
                totalLiquidity
                tradeVolumeUSD
                txCount
                derivedETH
                tokenDayData(first: 7, orderBy: date, orderDirection: desc) {
                  priceUSD
                  dailyVolumeUSD
                  totalLiquidityUSD
                }
              }
            }
            '''

    def parse_token_data(self, data: Dict, dex: str) -> List[Dict]:
        try:
            tokens = data.get('data', {}).get('tokens', [])
            parsed = []
            
            for token in tokens:
                try:
                    day_data = token.get('tokenDayData', [])
                    if not day_data:
                        continue
                    
                    current_price = float(day_data[0].get('priceUSD', 0))
                    if current_price <= 0:
                        continue
                    
                    if 'uniswap_v3' in dex:
                        liquidity_usd = float(token.get('totalValueLockedUSD', 0))
                        volume_usd = float(token.get('volumeUSD', 0))
                    else:
                        liquidity_usd = float(token.get('totalLiquidity', 0))
                        volume_usd = float(token.get('tradeVolumeUSD', 0))
                    
                    if liquidity_usd < get_dynamic_config()['min_liquidity_threshold']:
                        continue
                    
                    parsed_token = {
                        'id': token['id'],
                        'symbol': token.get('symbol', 'UNKNOWN'),
                        'name': token.get('name', 'Unknown'),
                        'current_price': current_price,
                        'liquidity_usd': liquidity_usd,
                        'volume_usd': volume_usd,
                        'tx_count': int(token.get('txCount', 0)),
                        'day_data': day_data
                    }
                    
                    parsed.append(parsed_token)
                    
                except (ValueError, KeyError):
                    continue
            
            return parsed
            
        except Exception:
            return []

    async def analyze_token_momentum(self, token: Dict, chain: str, dex: str):
        try:
            token_key = f"{chain}_{token['id']}"
            current_price = token['current_price']
            
            self.price_history[token_key].append(current_price)
            self.volume_history[token_key].append(token.get('volume_usd', 0))
            
            if len(self.price_history[token_key]) < 3:
                return
            
            price_change_24h = self.calculate_price_change(token)
            momentum_score = self.calculate_momentum_score(token_key, token)
            velocity = self.calculate_velocity(self.price_history[token_key])
            volatility = self.calculate_volatility(self.price_history[token_key])
            confidence = self.calculate_confidence_score(token, momentum_score)
            
            config = get_dynamic_config()
            
            if (config['min_price_change'] <= abs(price_change_24h) <= config['max_price_change'] and
                momentum_score >= config['momentum_threshold'] and
                confidence >= config['confidence_threshold']):
                
                signal = TokenSignal(
                    address=token['id'],
                    chain=chain,
                    symbol=token['symbol'],
                    name=token['name'],
                    price=current_price,
                    volume_24h=token.get('volume_usd', 0),
                    price_change_24h=price_change_24h,
                    momentum_score=momentum_score,
                    liquidity_usd=token['liquidity_usd'],
                    market_cap=current_price * 1000000,
                    detected_at=time.time(),
                    confidence=confidence,
                    velocity=velocity,
                    volatility=volatility,
                    dex_source=dex
                )
                
                try:
                    self.signal_queue.put_nowait(signal)
                    self.stats['signals_generated'] += 1
                except asyncio.QueueFull:
                    pass
                    
        except Exception as e:
            self.stats['errors'] += 1

    def calculate_price_change(self, token: Dict) -> float:
        day_data = token.get('day_data', [])
        if len(day_data) < 2:
            return 0.0
        
        current_price = float(day_data[0].get('priceUSD', 0))
        yesterday_price = float(day_data[1].get('priceUSD', 0))
        
        if yesterday_price <= 0:
            return 0.0
        
        return ((current_price - yesterday_price) / yesterday_price) * 100

    def calculate_momentum_score(self, token_key: str, token: Dict) -> float:
        price_hist = np.array(list(self.price_history[token_key]))
        volume_hist = np.array(list(self.volume_history[token_key]))
        
        if len(price_hist) < 3:
            return 0.0
        
        price_momentum = 0.0
        if len(price_hist) >= 2:
            recent_avg = np.mean(price_hist[-2:])
            older_avg = np.mean(price_hist[:-2]) if len(price_hist) > 2 else price_hist[0]
            if older_avg > 0:
                price_momentum = (recent_avg - older_avg) / older_avg
        
        volume_momentum = 0.0
        if len(volume_hist) >= 2:
            recent_vol = volume_hist[-1]
            older_vol = np.mean(volume_hist[:-1])
            if older_vol > 0:
                volume_momentum = (recent_vol - older_vol) / older_vol
        
        combined_momentum = price_momentum * 0.6 + min(volume_momentum, 1.0) * 0.4
        
        return np.clip(combined_momentum * 2 + 0.5, 0.0, 1.0)

    def calculate_velocity(self, price_history: deque) -> float:
        if len(price_history) < 2:
            return 0.0
        
        prices = np.array(list(price_history))
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        velocity = np.mean(returns)
        
        return np.clip(velocity * 10 + 0.5, 0.0, 1.0)

    def calculate_volatility(self, price_history: deque) -> float:
        if len(price_history) < 2:
            return 0.0
        
        prices = np.array(list(price_history))
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        volatility = np.std(returns)
        
        return np.clip(volatility * 20, 0.0, 1.0)

    def calculate_confidence_score(self, token: Dict, momentum_score: float) -> float:
        liquidity_factor = min(token['liquidity_usd'] / 50000, 1.0)
        volume_factor = min(token.get('volume_usd', 0) / 10000, 1.0)
        tx_factor = min(token.get('tx_count', 0) / 500, 1.0)
        
        confidence = (
            momentum_score * 0.4 +
            liquidity_factor * 0.3 +
            volume_factor * 0.2 +
            tx_factor * 0.1
        )
        
        return np.clip(confidence, 0.0, 1.0)

    async def performance_monitor(self):
        while self.running:
            try:
                runtime = time.time() - self.stats['start_time']
                tokens_per_hour = (self.stats['tokens_scanned'] / runtime) * 3600 if runtime > 0 else 0
                daily_projection = tokens_per_hour * 24
                
                self.logger.info("=" * 70)
                self.logger.info("📊 ULTRA SCANNER PERFORMANCE")
                self.logger.info("=" * 70)
                self.logger.info(f"⏱️  Runtime: {runtime/60:.1f} minutes")
                self.logger.info(f"🔍 Tokens scanned: {self.stats['tokens_scanned']:,}")
                self.logger.info(f"📊 Signals generated: {self.stats['signals_generated']:,}")
                self.logger.info(f"🌐 API calls: {self.stats['api_calls']:,}")
                self.logger.info(f"❌ Errors: {self.stats['errors']:,}")
                self.logger.info(f"🚀 Scan rate: {tokens_per_hour:.0f} tokens/hour")
                self.logger.info(f"🎯 Daily projection: {daily_projection:.0f}/day")
                self.logger.info(f"🏆 Target (10k): {min(daily_projection/10000*100, 100):.1f}%")
                self.logger.info(f"⚙️  Workers: {len(self.workers)}")
                self.logger.info(f"💾 Queue: {self.signal_queue.qsize()}")
                self.logger.info(f"🔗 Discovered: {len(self.discovered_tokens):,}")
                self.logger.info("=" * 70)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                await asyncio.sleep(60)

    async def get_signals(self, max_signals: int = 20) -> List[TokenSignal]:
        signals = []
        
        for _ in range(max_signals):
            try:
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=0.1)
                signals.append(signal)
            except asyncio.TimeoutError:
                break
        
        return sorted(signals, key=lambda x: x.momentum_score * x.confidence, reverse=True)

    async def get_recent_tokens(self, limit: int = 100) -> List[TokenSignal]:
        return await self.get_signals(limit)

    async def shutdown(self):
        self.running = False
        
        for worker in self.workers:
            worker.cancel()
        
        if self.session:
            await self.session.close()

ultra_scanner = UltraScanner()