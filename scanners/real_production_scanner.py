import asyncio
import aiohttp
import json
import time
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
import websockets
from core.web3_manager import web3_manager

@dataclass
class RealTokenSignal:
    address: str
    chain: str
    symbol: str
    name: str
    price: float
    volume_24h: float
    price_change_1m: float
    price_change_5m: float
    momentum_score: float
    liquidity_usd: float
    market_cap: float
    detected_at: float
    confidence: float
    velocity: float
    volatility: float
    breakout_strength: float
    pair_address: str
    reserve0: int
    reserve1: int

class RealProductionScanner:
    def __init__(self):
        self.dexscreener_base = "https://api.dexscreener.com/latest"
        self.moralis_base = "https://deep-index.moralis.io/api/v2"
        self.session = None
        
        self.price_history = defaultdict(lambda: deque(maxlen=300))
        self.volume_history = defaultdict(lambda: deque(maxlen=300))
        self.discovered_tokens = set()
        self.signal_queue = asyncio.Queue(maxsize=10000)
        
        self.chains = ['ethereum', 'arbitrum', 'polygon']
        self.running = False
        
        self.stats = {
            'tokens_scanned': 0,
            'signals_generated': 0,
            'api_calls': 0,
            'start_time': time.time()
        }
        
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=100)
        )
        
        await web3_manager.initialize()
        self.running = True
        
        tasks = []
        for chain in self.chains:
            tasks.append(asyncio.create_task(self.scan_new_pairs(chain)))
            tasks.append(asyncio.create_task(self.monitor_price_movements(chain)))
            tasks.append(asyncio.create_task(self.websocket_monitor(chain)))
        
        tasks.append(asyncio.create_task(self.momentum_detector()))
        tasks.append(asyncio.create_task(self.performance_tracker()))
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def scan_new_pairs(self, chain: str):
        while self.running:
            try:
                new_pairs = await self.fetch_recent_pairs(chain)
                
                for pair in new_pairs:
                    if pair['baseToken']['address'] not in self.discovered_tokens:
                        self.discovered_tokens.add(pair['baseToken']['address'])
                        await self.analyze_new_token(pair, chain)
                        self.stats['tokens_scanned'] += 1
                        
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error scanning {chain}: {e}")
                await asyncio.sleep(30)

    async def fetch_recent_pairs(self, chain: str) -> List[Dict]:
        url = f"{self.dexscreener_base}/dex/tokens/{chain}"
        
        try:
            async with self.session.get(url) as response:
                self.stats['api_calls'] += 1
                
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    recent_pairs = []
                    current_time = time.time()
                    
                    for pair in pairs:
                        pair_created = pair.get('pairCreatedAt')
                        if pair_created:
                            created_timestamp = int(pair_created) / 1000
                            if current_time - created_timestamp < 3600:
                                volume = float(pair.get('volume', {}).get('h24', 0))
                                if volume > 10000:
                                    recent_pairs.append(pair)
                    
                    return recent_pairs[:50]
                    
        except Exception as e:
            self.logger.error(f"Error fetching pairs from {chain}: {e}")
            
        return []

    async def analyze_new_token(self, pair: Dict, chain: str):
        try:
            base_token = pair['baseToken']
            token_address = base_token['address']
            
            token_info = await web3_manager.get_token_info(token_address, chain)
            if not token_info:
                return
                
            security_analysis = await web3_manager.analyze_contract_security(token_address, chain)
            if not security_analysis['safe']:
                return
            
            price = float(pair.get('priceUsd', 0))
            volume_24h = float(pair.get('volume', {}).get('h24', 0))
            liquidity = float(pair.get('liquidity', {}).get('usd', 0))
            
            if price <= 0 or volume_24h < 10000 or liquidity < 50000:
                return
                
            token_key = f"{chain}_{token_address}"
            self.price_history[token_key].append((time.time(), price))
            self.volume_history[token_key].append((time.time(), volume_24h))
            
            if len(self.price_history[token_key]) >= 10:
                await self.check_momentum_breakout(token_key, pair, chain)
                
        except Exception as e:
            self.logger.error(f"Error analyzing token: {e}")

    async def monitor_price_movements(self, chain: str):
        while self.running:
            try:
                for token_key in list(self.price_history.keys()):
                    if chain in token_key:
                        await self.update_token_price(token_key, chain)
                        
                await asyncio.sleep(2)
                
            except Exception as e:
                await asyncio.sleep(10)

    async def update_token_price(self, token_key: str, chain: str):
        try:
            _, token_address = token_key.split('_', 1)
            
            current_price = await web3_manager.get_token_price(token_address, chain)
            if current_price:
                self.price_history[token_key].append((time.time(), current_price))
                
                if len(self.price_history[token_key]) >= 20:
                    await self.detect_price_momentum(token_key, chain)
                    
        except Exception as e:
            self.logger.debug(f"Error updating price for {token_key}: {e}")

    async def detect_price_momentum(self, token_key: str, chain: str):
        price_data = list(self.price_history[token_key])
        
        if len(price_data) < 20:
            return
            
        current_time = time.time()
        
        prices_1m = [p[1] for p in price_data if current_time - p[0] <= 60]
        prices_5m = [p[1] for p in price_data if current_time - p[0] <= 300]
        
        if len(prices_1m) < 5 or len(prices_5m) < 10:
            return
            
        price_change_1m = ((prices_1m[-1] - prices_1m[0]) / prices_1m[0]) * 100
        price_change_5m = ((prices_5m[-1] - prices_5m[0]) / prices_5m[0]) * 100
        
        if 9 <= price_change_1m <= 15:
            momentum_score = self.calculate_momentum_score(prices_1m, prices_5m)
            velocity = self.calculate_velocity(prices_1m)
            volatility = self.calculate_volatility(prices_1m)
            breakout_strength = self.calculate_breakout_strength(price_data)
            
            if momentum_score >= 0.7 and breakout_strength >= 0.6:
                await self.create_signal(token_key, chain, price_change_1m, price_change_5m, 
                                       momentum_score, velocity, volatility, breakout_strength)

    def calculate_momentum_score(self, prices_1m: List[float], prices_5m: List[float]) -> float:
        if len(prices_1m) < 3 or len(prices_5m) < 5:
            return 0.0
            
        recent_trend = np.polyfit(range(len(prices_1m)), prices_1m, 1)[0]
        baseline_trend = np.polyfit(range(len(prices_5m)), prices_5m, 1)[0]
        
        trend_acceleration = recent_trend / (baseline_trend + 1e-10)
        
        price_ratio = prices_1m[-1] / np.mean(prices_5m)
        
        volume_factor = 1.0
        
        momentum = (trend_acceleration * 0.5 + price_ratio * 0.3 + volume_factor * 0.2)
        
        return np.clip(momentum / 2, 0.0, 1.0)

    def calculate_velocity(self, prices: List[float]) -> float:
        if len(prices) < 3:
            return 0.0
            
        returns = np.diff(prices) / (np.array(prices[:-1]) + 1e-10)
        velocity = np.mean(returns[-3:])
        
        return np.clip(velocity * 20 + 0.5, 0.0, 1.0)

    def calculate_volatility(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0
            
        returns = np.diff(prices) / (np.array(prices[:-1]) + 1e-10)
        volatility = np.std(returns)
        
        return np.clip(volatility * 50, 0.0, 1.0)

    def calculate_breakout_strength(self, price_data: List[Tuple]) -> float:
        if len(price_data) < 30:
            return 0.0
            
        recent_prices = [p[1] for p in price_data[-15:]]
        older_prices = [p[1] for p in price_data[-30:-15]]
        
        recent_vol = np.std(recent_prices) / (np.mean(recent_prices) + 1e-10)
        older_vol = np.std(older_prices) / (np.mean(older_prices) + 1e-10)
        
        volatility_spike = recent_vol / (older_vol + 1e-10)
        
        price_jump = recent_prices[-1] / (np.mean(older_prices) + 1e-10)
        
        strength = min(volatility_spike * price_jump, 3.0) / 3.0
        
        return np.clip(strength, 0.0, 1.0)

    async def create_signal(self, token_key: str, chain: str, price_change_1m: float, 
                          price_change_5m: float, momentum_score: float, velocity: float, 
                          volatility: float, breakout_strength: float):
        try:
            _, token_address = token_key.split('_', 1)
            
            token_info = await web3_manager.get_token_info(token_address, chain)
            if not token_info:
                return
                
            current_price = self.price_history[token_key][-1][1]
            
            weth_address = web3_manager.chains[chain]['weth']
            pair_info = await web3_manager.get_pair_info(token_address, weth_address, chain)
            
            confidence = min(momentum_score * breakout_strength * (1 - volatility * 0.5), 1.0)
            
            if confidence >= 0.75:
                signal = RealTokenSignal(
                    address=token_address,
                    chain=chain,
                    symbol=token_info.symbol,
                    name=token_info.name,
                    price=current_price,
                    volume_24h=np.mean([v[1] for v in list(self.volume_history[token_key])[-10:]]),
                    price_change_1m=price_change_1m,
                    price_change_5m=price_change_5m,
                    momentum_score=momentum_score,
                    liquidity_usd=0.0,
                    market_cap=current_price * token_info.total_supply / (10 ** token_info.decimals),
                    detected_at=time.time(),
                    confidence=confidence,
                    velocity=velocity,
                    volatility=volatility,
                    breakout_strength=breakout_strength,
                    pair_address=pair_info.address if pair_info else '',
                    reserve0=pair_info.reserve0 if pair_info else 0,
                    reserve1=pair_info.reserve1 if pair_info else 0
                )
                
                try:
                    self.signal_queue.put_nowait(signal)
                    self.stats['signals_generated'] += 1
                    
                    self.logger.info(f"SIGNAL: {token_info.symbol} {price_change_1m:+.1f}% momentum={momentum_score:.2f} confidence={confidence:.2f}")
                    
                except asyncio.QueueFull:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error creating signal: {e}")

    async def websocket_monitor(self, chain: str):
        while self.running:
            try:
                await asyncio.sleep(10)
            except Exception as e:
                await asyncio.sleep(30)

    async def momentum_detector(self):
        while self.running:
            try:
                await asyncio.sleep(1)
                
                for token_key in list(self.price_history.keys()):
                    if len(self.price_history[token_key]) >= 30:
                        chain = token_key.split('_')[0]
                        await self.detect_price_momentum(token_key, chain)
                        
            except Exception as e:
                await asyncio.sleep(5)

    async def performance_tracker(self):
        while self.running:
            try:
                runtime = time.time() - self.stats['start_time']
                tokens_per_hour = (self.stats['tokens_scanned'] / runtime) * 3600 if runtime > 0 else 0
                
                self.logger.info(f"Scanner: {self.stats['tokens_scanned']} tokens, {self.stats['signals_generated']} signals, {tokens_per_hour:.0f}/hour")
                
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
                
        return sorted(signals, key=lambda x: x.confidence * x.momentum_score, reverse=True)

    async def check_momentum_breakout(self, token_key: str, pair: Dict, chain: str):
        price_data = list(self.price_history[token_key])
        
        if len(price_data) < 10:
            return
            
        current_price = price_data[-1][1]
        baseline_price = np.mean([p[1] for p in price_data[:-5]])
        
        price_change = ((current_price - baseline_price) / baseline_price) * 100
        
        if 9 <= price_change <= 15:
            await self.detect_price_momentum(token_key, chain)

    async def shutdown(self):
        self.running = False
        if self.session:
            await self.session.close()
        await web3_manager.close()

production_scanner = RealProductionScanner()