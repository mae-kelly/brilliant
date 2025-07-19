import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import websockets
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np

@dataclass
class OrderBookLevel:
    price: float
    size: float
    orders: int

@dataclass
class OrderBookSnapshot:
    token_address: str
    chain: str
    dex: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: float
    mid_price: float
    spread: float
    depth_usd: float

class OrderBookMonitor:
    def __init__(self):
        self.dex_orderbook_feeds = {
            'uniswap_v3': 'wss://api.uniswap.org/v1/graphql',
            'camelot': 'wss://api.camelot.exchange/v1/orderbook',
            'quickswap': 'wss://api.quickswap.exchange/v1/orderbook',
            'curve': 'wss://api.curve.fi/v1/orderbook'
        }
        
        self.orderbooks = defaultdict(lambda: deque(maxlen=100))
        self.connections = {}
        self.running = False
        
        self.stats = {
            'orderbook_updates': 0,
            'liquidity_alerts': 0,
            'spread_alerts': 0
        }

    async def initialize(self):
        self.running = True
        
        tasks = []
        for dex, endpoint in self.dex_orderbook_feeds.items():
            task = asyncio.create_task(self.monitor_orderbook(dex, endpoint))
            tasks.append(task)
        
        task = asyncio.create_task(self.analyze_orderbook_patterns())
        tasks.append(task)

    async def monitor_orderbook(self, dex: str, endpoint: str):
        while self.running:
            try:
                async with websockets.connect(endpoint) as websocket:
                    self.connections[dex] = websocket
                    
                    subscription = {
                        "type": "subscribe",
                        "channel": "orderbook",
                        "symbols": ["*"]
                    }
                    
                    await websocket.send(json.dumps(subscription))
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self.process_orderbook_update(data, dex)
                        except Exception as e:
                            continue
                        
            except Exception as e:
                await asyncio.sleep(5)

    async def process_orderbook_update(self, data: dict, dex: str):
        try:
            if 'data' not in data:
                return
            
            update_data = data['data']
            token_address = update_data.get('symbol', '').split('/')[0]
            
            if not token_address:
                return
            
            bids = []
            asks = []
            
            for bid_data in update_data.get('bids', [])[:20]:
                bids.append(OrderBookLevel(
                    price=float(bid_data[0]),
                    size=float(bid_data[1]),
                    orders=int(bid_data[2]) if len(bid_data) > 2 else 1
                ))
            
            for ask_data in update_data.get('asks', [])[:20]:
                asks.append(OrderBookLevel(
                    price=float(ask_data[0]),
                    size=float(ask_data[1]),
                    orders=int(ask_data[2]) if len(ask_data) > 2 else 1
                ))
            
            if bids and asks:
                mid_price = (bids[0].price + asks[0].price) / 2
                spread = (asks[0].price - bids[0].price) / mid_price
                depth_usd = sum(level.price * level.size for level in bids[:5] + asks[:5])
                
                snapshot = OrderBookSnapshot(
                    token_address=token_address,
                    chain=self.get_chain_from_dex(dex),
                    dex=dex,
                    bids=bids,
                    asks=asks,
                    timestamp=time.time(),
                    mid_price=mid_price,
                    spread=spread,
                    depth_usd=depth_usd
                )
                
                cache_key = f"{dex}_{token_address}"
                self.orderbooks[cache_key].append(snapshot)
                self.stats['orderbook_updates'] += 1
                
        except Exception as e:
            pass

    def get_chain_from_dex(self, dex: str) -> str:
        dex_chains = {
            'uniswap_v3': 'ethereum',
            'camelot': 'arbitrum',
            'quickswap': 'polygon',
            'curve': 'ethereum'
        }
        return dex_chains.get(dex, 'ethereum')

    async def analyze_orderbook_patterns(self):
        while self.running:
            try:
                for cache_key, snapshots in self.orderbooks.items():
                    if len(snapshots) >= 5:
                        await self.detect_liquidity_patterns(cache_key, list(snapshots))
                
                await asyncio.sleep(10)
                
            except Exception as e:
                await asyncio.sleep(30)

    async def detect_liquidity_patterns(self, cache_key: str, snapshots: List[OrderBookSnapshot]):
        try:
            latest = snapshots[-1]
            
            spreads = [s.spread for s in snapshots[-10:]]
            depths = [s.depth_usd for s in snapshots[-10:]]
            
            avg_spread = np.mean(spreads)
            avg_depth = np.mean(depths)
            
            if latest.spread > avg_spread * 2:
                self.stats['spread_alerts'] += 1
                print(f"🚨 Wide spread alert: {cache_key} - {latest.spread:.4f}")
            
            if latest.depth_usd < avg_depth * 0.5:
                self.stats['liquidity_alerts'] += 1
                print(f"💧 Low liquidity alert: {cache_key} - ${latest.depth_usd:.0f}")
            
            imbalance = self.calculate_order_imbalance(latest)
            if abs(imbalance) > 0.7:
                print(f"⚖️ Order imbalance: {cache_key} - {imbalance:.2f}")
                
        except Exception as e:
            pass

    def calculate_order_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        try:
            bid_volume = sum(level.size for level in snapshot.bids[:5])
            ask_volume = sum(level.size for level in snapshot.asks[:5])
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.0
            
            return (bid_volume - ask_volume) / total_volume
            
        except Exception as e:
            return 0.0

    def get_orderbook_snapshot(self, token_address: str, dex: str) -> Optional[OrderBookSnapshot]:
        cache_key = f"{dex}_{token_address}"
        snapshots = self.orderbooks.get(cache_key)
        
        if snapshots:
            return snapshots[-1]
        
        return None

    def get_spread_history(self, token_address: str, dex: str, limit: int = 50) -> List[float]:
        cache_key = f"{dex}_{token_address}"
        snapshots = self.orderbooks.get(cache_key, [])
        
        return [s.spread for s in list(snapshots)[-limit:]]

    async def shutdown(self):
        self.running = False
        
        for connection in self.connections.values():
            try:
                await connection.close()
            except:
                pass

orderbook_monitor = OrderBookMonitor()
