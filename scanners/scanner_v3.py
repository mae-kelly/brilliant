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
import requests

@dataclass
class RealTokenSignal:
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
    holder_count: int
    tx_count_24h: int
    detected_at: float
    confidence: float
    velocity: float
    volatility: float
    order_flow_imbalance: float

class RealTimeGraphQLScanner:
    def __init__(self):
        self.endpoints = {
            'uniswap_v3': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
            'uniswap_v2': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2',
            'sushiswap': 'https://api.thegraph.com/subgraphs/name/sushi-v2/sushiswap-ethereum',
            'pancakeswap': 'https://api.thegraph.com/subgraphs/name/pancakeswap/exchange',
            'camelot': 'https://api.thegraph.com/subgraphs/name/camelot-labs/camelot-amm',
            'quickswap': 'https://api.thegraph.com/subgraphs/name/sameepsi/quickswap06'
        }
        
        self.chain_endpoints = {
            'ethereum': ['uniswap_v3', 'uniswap_v2', 'sushiswap'],
            'polygon': ['quickswap', 'sushiswap'],
            'arbitrum': ['camelot', 'sushiswap'],
            'bsc': ['pancakeswap']
        }
        
        self.session = None
        self.discovered_tokens = set()
        self.signal_queue = asyncio.Queue(maxsize=100000)
        self.price_feeds = defaultdict(lambda: deque(maxlen=1000))
        self.volume_feeds = defaultdict(lambda: deque(maxlen=1000))
        self.worker_count = 1000
        self.running = False
        
        self.stats = {
            'tokens_scanned': 0,
            'signals_generated': 0,
            'api_calls': 0,
            'start_time': time.time(),
            'errors': 0,
            'cache_hits': 0
        }
        
        self.cache = {}
        self.cache_ttl = 30
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5),
            connector=aiohttp.TCPConnector(limit=200)
        )
        
        self.running = True
        
        tasks = []
        for i in range(self.worker_count):
            chain = list(self.chain_endpoints.keys())[i % len(self.chain_endpoints)]
            tasks.append(asyncio.create_task(self.worker_loop(i, chain)))
        
        tasks.append(asyncio.create_task(self.price_tracker_loop()))
        tasks.append(asyncio.create_task(self.volume_tracker_loop()))
        tasks.append(asyncio.create_task(self.momentum_detector_loop()))
        tasks.append(asyncio.create_task(self.performance_monitor()))
        
        self.logger.info(f"Initialized {self.worker_count} workers across {len(self.chain_endpoints)} chains")
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def worker_loop(self, worker_id: int, chain: str):
        while self.running:
            try:
                await self.scan_chain_tokens(worker_id, chain)
                await asyncio.sleep(0.01)
            except Exception as e:
                self.stats['errors'] += 1
                await asyncio.sleep(1)

    async def scan_chain_tokens(self, worker_id: int, chain: str):
        dexes = self.chain_endpoints.get(chain, [])
        for dex in dexes:
            try:
                tokens = await self.fetch_trending_tokens(dex, chain)
                for token in tokens:
                    if token['id'] not in self.discovered_tokens:
                        self.discovered_tokens.add(token['id'])
                        await self.analyze_token(token, chain, dex)
                        self.stats['tokens_scanned'] += 1
            except Exception as e:
                self.stats['errors'] += 1

    async def fetch_trending_tokens(self, dex: str, chain: str) -> List[Dict]:
        endpoint = self.endpoints.get(dex)
        if not endpoint:
            return []
        
        cache_key = f"{dex}_{chain}_{int(time.time() // self.cache_ttl)}"
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        query = self.build_tokens_query(dex)
        
        try:
            async with self.session.post(
                endpoint,
                json={'query': query},
                headers={'Content-Type': 'application/json'}
            ) as response:
                self.stats['api_calls'] += 1
                
                if response.status == 200:
                    data = await response.json()
                    tokens = self.parse_token_response(data, dex)
                    self.cache[cache_key] = tokens
                    return tokens
                    
        except Exception as e:
            self.stats['errors'] += 1
            
        return []

    def build_tokens_query(self, dex: str) -> str:
        if dex in ['uniswap_v3', 'uniswap_v2']:
            return '''
            {
              tokens(
                first: 100
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
                decimals
                totalSupply
                volume
                volumeUSD
                txCount
                totalValueLocked
                totalValueLockedUSD
                derivedETH
                tokenDayData(first: 7, orderBy: date, orderDirection: desc) {
                  priceUSD
                  volume
                  volumeUSD
                  totalValueLockedUSD
                  date
                }
              }
            }
            '''
        elif dex == 'sushiswap':
            return '''
            {
              tokens(
                first: 100
                orderBy: liquidityUSD
                orderDirection: desc
                where: {
                  liquidityUSD_gt: "10000"
                }
              ) {
                id
                symbol
                name
                decimals
                totalSupply
                volume
                volumeUSD
                txCount
                liquidity
                liquidityUSD
                derivedETH
              }
            }
            '''
        else:
            return '''
            {
              tokens(
                first: 100
                orderBy: tradeVolumeUSD
                orderDirection: desc
              ) {
                id
                symbol
                name
                decimals
                totalSupply
                tradeVolume
                tradeVolumeUSD
                txCount
                totalLiquidity
                derivedBNB
              }
            }
            '''

    def parse_token_response(self, data: Dict, dex: str) -> List[Dict]:
        try:
            tokens = data.get('data', {}).get('tokens', [])
            parsed_tokens = []
            
            for token in tokens:
                try:
                    if dex in ['uniswap_v3', 'uniswap_v2']:
                        parsed_token = {
                            'id': token['id'],
                            'symbol': token.get('symbol', 'UNKNOWN'),
                            'name': token.get('name', 'Unknown Token'),
                            'decimals': int(token.get('decimals', 18)),
                            'total_supply': float(token.get('totalSupply', 0)),
                            'volume_usd': float(token.get('volumeUSD', 0)),
                            'liquidity_usd': float(token.get('totalValueLockedUSD', 0)),
                            'tx_count': int(token.get('txCount', 0)),
                            'derived_eth': float(token.get('derivedETH', 0)),
                            'day_data': token.get('tokenDayData', [])
                        }
                    elif dex == 'sushiswap':
                        parsed_token = {
                            'id': token['id'],
                            'symbol': token.get('symbol', 'UNKNOWN'),
                            'name': token.get('name', 'Unknown Token'),
                            'decimals': int(token.get('decimals', 18)),
                            'total_supply': float(token.get('totalSupply', 0)),
                            'volume_usd': float(token.get('volumeUSD', 0)),
                            'liquidity_usd': float(token.get('liquidityUSD', 0)),
                            'tx_count': int(token.get('txCount', 0)),
                            'derived_eth': float(token.get('derivedETH', 0))
                        }
                    else:
                        parsed_token = {
                            'id': token['id'],
                            'symbol': token.get('symbol', 'UNKNOWN'),
                            'name': token.get('name', 'Unknown Token'),
                            'decimals': int(token.get('decimals', 18)),
                            'total_supply': float(token.get('totalSupply', 0)),
                            'volume_usd': float(token.get('tradeVolumeUSD', 0)),
                            'liquidity_usd': float(token.get('totalLiquidity', 0)),
                            'tx_count': int(token.get('txCount', 0)),
                            'derived_eth': float(token.get('derivedBNB', 0))
                        }
                    
                    parsed_tokens.append(parsed_token)
                    
                except (ValueError, KeyError) as e:
                    continue
                    
            return parsed_tokens
            
        except Exception as e:
            return []

    async def analyze_token(self, token: Dict, chain: str, dex: str):
        try:
            current_price = await self.get_current_price(token, chain)
            if not current_price:
                return
            
            volume_24h = token.get('volume_usd', 0)
            liquidity_usd = token.get('liquidity_usd', 0)
            
            if liquidity_usd < 10000 or volume_24h < 1000:
                return
            
            price_history = await self.get_price_history(token, chain)
            volume_history = await self.get_volume_history(token, chain)
            
            price_change_24h = self.calculate_price_change(price_history, current_price)
            momentum_score = self.calculate_momentum_score(price_history, volume_history, current_price)
            velocity = self.calculate_velocity(price_history)
            volatility = self.calculate_volatility(price_history)
            order_flow = self.calculate_order_flow_imbalance(token, chain)
            
            if 9 <= abs(price_change_24h) <= 15 and momentum_score > 0.7:
                confidence = self.calculate_confidence(token, momentum_score, liquidity_usd)
                
                signal = RealTokenSignal(
                    address=token['id'],
                    chain=chain,
                    symbol=token['symbol'],
                    name=token['name'],
                    price=current_price,
                    volume_24h=volume_24h,
                    price_change_24h=price_change_24h,
                    momentum_score=momentum_score,
                    liquidity_usd=liquidity_usd,
                    market_cap=current_price * token.get('total_supply', 0),
                    holder_count=self.estimate_holder_count(token),
                    tx_count_24h=token.get('tx_count', 0),
                    detected_at=time.time(),
                    confidence=confidence,
                    velocity=velocity,
                    volatility=volatility,
                    order_flow_imbalance=order_flow
                )
                
                try:
                    self.signal_queue.put_nowait(signal)
                    self.stats['signals_generated'] += 1
                except:
                    pass
                    
        except Exception as e:
            self.stats['errors'] += 1

    async def get_current_price(self, token: Dict, chain: str) -> Optional[float]:
        try:
            if token.get('derived_eth'):
                eth_price = await self.get_eth_price()
                return float(token['derived_eth']) * eth_price
            return None
        except:
            return None

    async def get_eth_price(self) -> float:
        cache_key = f"eth_price_{int(time.time() // 60)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with self.session.get(
                'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd'
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    price = data['ethereum']['usd']
                    self.cache[cache_key] = price
                    return price
        except:
            pass
        
        return 2000.0

    async def get_price_history(self, token: Dict, chain: str) -> List[float]:
        token_id = token['id']
        if token_id in self.price_feeds:
            return list(self.price_feeds[token_id])
        
        day_data = token.get('day_data', [])
        if day_data:
            prices = [float(day.get('priceUSD', 0)) for day in day_data[-20:]]
            self.price_feeds[token_id].extend(prices)
            return prices
        
        return []

    async def get_volume_history(self, token: Dict, chain: str) -> List[float]:
        token_id = token['id']
        if token_id in self.volume_feeds:
            return list(self.volume_feeds[token_id])
        
        day_data = token.get('day_data', [])
        if day_data:
            volumes = [float(day.get('volumeUSD', 0)) for day in day_data[-20:]]
            self.volume_feeds[token_id].extend(volumes)
            return volumes
        
        return []

    def calculate_price_change(self, price_history: List[float], current_price: float) -> float:
        if not price_history or len(price_history) < 2:
            return 0.0
        
        yesterday_price = price_history[-1] if price_history else current_price
        if yesterday_price == 0:
            return 0.0
        
        return ((current_price - yesterday_price) / yesterday_price) * 100

    def calculate_momentum_score(self, price_history: List[float], volume_history: List[float], current_price: float) -> float:
        if not price_history or not volume_history:
            return 0.0
        
        price_momentum = 0.0
        if len(price_history) >= 5:
            recent_avg = np.mean(price_history[-5:])
            older_avg = np.mean(price_history[:-5]) if len(price_history) > 5 else recent_avg
            if older_avg > 0:
                price_momentum = (recent_avg - older_avg) / older_avg
        
        volume_momentum = 0.0
        if len(volume_history) >= 5:
            recent_vol = np.mean(volume_history[-5:])
            older_vol = np.mean(volume_history[:-5]) if len(volume_history) > 5 else recent_vol
            if older_vol > 0:
                volume_momentum = (recent_vol - older_vol) / older_vol
        
        acceleration = 0.0
        if len(price_history) >= 3:
            p1, p2, p3 = price_history[-3:]
            if p1 > 0 and p2 > 0:
                r1 = (p2 - p1) / p1
                r2 = (p3 - p2) / p2
                acceleration = r2 - r1
        
        momentum_score = (
            price_momentum * 0.4 +
            volume_momentum * 0.3 +
            acceleration * 0.3
        )
        
        return np.clip(momentum_score * 2 + 0.5, 0.0, 1.0)

    def calculate_velocity(self, price_history: List[float]) -> float:
        if len(price_history) < 2:
            return 0.0
        
        returns = np.diff(price_history) / (np.array(price_history[:-1]) + 1e-10)
        velocity = np.mean(returns)
        return np.clip(velocity * 10 + 0.5, 0.0, 1.0)

    def calculate_volatility(self, price_history: List[float]) -> float:
        if len(price_history) < 2:
            return 0.0
        
        returns = np.diff(price_history) / (np.array(price_history[:-1]) + 1e-10)
        volatility = np.std(returns)
        return np.clip(volatility * 20, 0.0, 1.0)

    def calculate_order_flow_imbalance(self, token: Dict, chain: str) -> float:
        tx_count = token.get('tx_count', 0)
        volume = token.get('volume_usd', 0)
        
        if tx_count == 0 or volume == 0:
            return 0.0
        
        avg_tx_size = volume / tx_count
        imbalance = np.tanh(avg_tx_size / 1000)
        
        return np.clip(imbalance, -1.0, 1.0)

    def estimate_holder_count(self, token: Dict) -> int:
        tx_count = token.get('tx_count', 0)
        total_supply = token.get('total_supply', 0)
        
        if tx_count == 0:
            return 0
        
        estimated_holders = int(tx_count * 0.1)
        return min(estimated_holders, int(total_supply * 0.01))

    def calculate_confidence(self, token: Dict, momentum_score: float, liquidity_usd: float) -> float:
        liquidity_factor = min(liquidity_usd / 100000, 1.0)
        tx_factor = min(token.get('tx_count', 0) / 1000, 1.0)
        volume_factor = min(token.get('volume_usd', 0) / 50000, 1.0)
        
        confidence = (
            momentum_score * 0.4 +
            liquidity_factor * 0.3 +
            tx_factor * 0.15 +
            volume_factor * 0.15
        )
        
        return np.clip(confidence, 0.0, 1.0)

    async def price_tracker_loop(self):
        while self.running:
            try:
                for token_id in list(self.price_feeds.keys())[-1000:]:
                    current_price = await self.get_current_price({'id': token_id}, 'ethereum')
                    if current_price:
                        self.price_feeds[token_id].append(current_price)
                
                await asyncio.sleep(10)
            except Exception as e:
                await asyncio.sleep(30)

    async def volume_tracker_loop(self):
        while self.running:
            try:
                for token_id in list(self.volume_feeds.keys())[-1000:]:
                    await asyncio.sleep(0.01)
                
                await asyncio.sleep(15)
            except Exception as e:
                await asyncio.sleep(30)

    async def momentum_detector_loop(self):
        while self.running:
            try:
                await asyncio.sleep(1)
            except Exception as e:
                await asyncio.sleep(5)

    async def performance_monitor(self):
        while self.running:
            try:
                runtime = time.time() - self.stats['start_time']
                tokens_per_hour = (self.stats['tokens_scanned'] / runtime) * 3600 if runtime > 0 else 0
                daily_projection = tokens_per_hour * 24
                
                self.logger.info("=" * 80)
                self.logger.info("📊 REAL-TIME DEX SCANNER PERFORMANCE")
                self.logger.info("=" * 80)
                self.logger.info(f"⏱️  Runtime: {runtime/60:.1f} minutes")
                self.logger.info(f"🔍 Tokens scanned: {self.stats['tokens_scanned']:,}")
                self.logger.info(f"📊 Signals generated: {self.stats['signals_generated']:,}")
                self.logger.info(f"🌐 API calls: {self.stats['api_calls']:,}")
                self.logger.info(f"💾 Cache hits: {self.stats['cache_hits']:,}")
                self.logger.info(f"❌ Errors: {self.stats['errors']:,}")
                self.logger.info(f"🚀 Rate: {tokens_per_hour:.0f} tokens/hour")
                self.logger.info(f"🎯 Daily projection: {daily_projection:.0f}/day")
                self.logger.info(f"🏆 Target progress: {min(daily_projection/10000*100, 100):.1f}%")
                self.logger.info(f"⚙️  Active workers: {self.worker_count}")
                self.logger.info(f"💾 Queue size: {self.signal_queue.qsize()}")
                self.logger.info(f"🔗 Discovered tokens: {len(self.discovered_tokens):,}")
                self.logger.info("=" * 80)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                await asyncio.sleep(120)

    async def get_signals(self, max_signals: int = 50) -> List[RealTokenSignal]:
        signals = []
        
        for _ in range(max_signals):
            try:
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=0.1)
                signals.append(signal)
            except asyncio.TimeoutError:
                break
        
        return sorted(signals, key=lambda x: x.momentum_score * x.confidence, reverse=True)

    async def shutdown(self):
        self.running = False
        if self.session:
            await self.session.close()

ultra_scanner = RealTimeGraphQLScanner()