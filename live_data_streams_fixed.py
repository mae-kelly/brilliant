
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import aiohttp
import json
import time
import os
from typing import Dict, List, Optional
from collections import deque, defaultdict
import requests

class LiveDataStreams:
    def __init__(self):
        self.price_feeds = defaultdict(lambda: deque(maxlen=1000))
        self.volume_feeds = defaultdict(lambda: deque(maxlen=1000))
        self.discovered_tokens = set()
        
        self.api_endpoints = {
            'ethereum': [
                "https://api.geckoterminal.com/api/v2/networks/eth/trending_pools",
                "https://api.geckoterminal.com/api/v2/networks/eth/new_pools"
            ],
            'arbitrum': [
                "https://api.geckoterminal.com/api/v2/networks/arbitrum/trending_pools",
                "https://api.geckoterminal.com/api/v2/networks/arbitrum/new_pools"
            ],
            'polygon': [
                "https://api.geckoterminal.com/api/v2/networks/polygon_pos/trending_pools",
                "https://api.geckoterminal.com/api/v2/networks/polygon_pos/new_pools"
            ]
        }
        
        alchemy_key = os.getenv('ALCHEMY_API_KEY')
        if alchemy_key and not alchemy_key.startswith('your_'):
            self.websocket_endpoints = {
                'ethereum': f"wss://eth-mainnet.g.alchemy.com/v2/{alchemy_key}",
                'arbitrum': f"wss://arb-mainnet.g.alchemy.com/v2/{alchemy_key}",
                'polygon': f"wss://polygon-mainnet.g.alchemy.com/v2/{alchemy_key}"
            }
        else:
            self.websocket_endpoints = {}
        
        self.session = None
        self.running = False
        self.momentum_signals = asyncio.Queue(maxsize=1000)

    async def initialize(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={'User-Agent': 'DeFi-Scanner/1.0'}
        )
        self.running = True
        
        tasks = []
        
        for chain in self.api_endpoints.keys():
            task = asyncio.create_task(self.api_token_discovery(chain))
            tasks.append(task)
        
        if self.websocket_endpoints:
            for chain, endpoint in self.websocket_endpoints.items():
                task = asyncio.create_task(self.websocket_connection(chain, endpoint))
                tasks.append(task)
        
        task = asyncio.create_task(self.momentum_analyzer())
        tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def api_token_discovery(self, chain: str):
        endpoints = self.api_endpoints[chain]
        
        while self.running:
            try:
                for endpoint in endpoints:
                    async with self.session.get(endpoint) as response:
                        if response.status == 200:
                            data = await response.json()
                            await self.process_api_data(data, chain)
                        
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"API discovery error for {chain}: {e}")
                await asyncio.sleep(10)

    async def process_api_data(self, data: dict, chain: str):
        pools = data.get('data', [])
        
        for pool in pools[:20]:
            try:
                attributes = pool.get('attributes', {})
                
                base_token = attributes.get('base_token_price_usd')
                quote_token = attributes.get('quote_token_price_usd')
                
                if base_token:
                    price = float(base_token)
                    volume = float(attributes.get('volume_usd', {}).get('h24', 0))
                    
                    token_address = pool.get('relationships', {}).get('base_token', {}).get('data', {}).get('id', '')
                    
                    if token_address and price > 0:
                        await self.store_token_data(token_address, price, volume, chain)
                        
            except Exception as e:
                continue

    async def store_token_data(self, token_address: str, price: float, volume: float, chain: str):
        token_key = f"{chain}_{token_address}"
        timestamp = time.time()
        
        self.price_feeds[token_key].append({
            'price': price,
            'timestamp': timestamp,
            'volume': volume
        })
        
        self.discovered_tokens.add(token_address)

    async def websocket_connection(self, chain: str, endpoint: str):
        while self.running:
            try:
                import websockets
                
                async with websockets.connect(endpoint) as websocket:
                    subscribe_msg = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "eth_subscribe",
                        "params": ["newHeads"]
                    }
                    
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    async for message in websocket:
                        data = json.loads(message)
                        await self.process_websocket_data(data, chain)
                        
            except Exception as e:
                print(f"WebSocket error for {chain}: {e}")
                await asyncio.sleep(5)

    async def process_websocket_data(self, data: dict, chain: str):
        pass

    async def momentum_analyzer(self):
        while self.running:
            try:
                for token_key, price_data in self.price_feeds.items():
                    if len(price_data) >= 10:
                        recent_prices = [p['price'] for p in list(price_data)[-10:]]
                        recent_volumes = [p['volume'] for p in list(price_data)[-10:]]
                        
                        if len(recent_prices) >= 2:
                            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                            
                            if abs(price_change) > 0.05:
                                avg_volume = sum(recent_volumes) / len(recent_volumes)
                                
                                signal = {
                                    'token_address': token_key.split('_', 1)[1],
                                    'chain': token_key.split('_')[0],
                                    'price_change': price_change,
                                    'current_price': recent_prices[-1],
                                    'volume': avg_volume,
                                    'timestamp': time.time()
                                }
                                
                                try:
                                    self.momentum_signals.put_nowait(signal)
                                except asyncio.QueueFull:
                                    pass
                
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Momentum analyzer error: {e}")
                await asyncio.sleep(5)

    async def get_signals(self, max_signals: int = 10) -> List[Dict]:
        signals = []
        
        for _ in range(max_signals):
            try:
                signal = await asyncio.wait_for(
                    self.momentum_signals.get(), 
                    timeout=0.1
                )
                signals.append(signal)
            except asyncio.TimeoutError:
                break
                
        return signals

    async def shutdown(self):
        self.running = False
        if self.session:
            await self.session.close()

live_streams = LiveDataStreams()
