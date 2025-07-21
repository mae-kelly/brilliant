#!/usr/bin/env python3
"""
Real-time WebSocket data feeds for instant market data
Connects to multiple DEX streams simultaneously
"""

import asyncio
import websockets
import json
import logging
import time
from typing import Dict, List, Callable
import aiohttp
import numpy as np
from dataclasses import dataclass
from collections import deque
import redis

@dataclass
class StreamData:
    symbol: str
    price: float
    volume: float
    timestamp: float
    source: str
    liquidity: float = 0
    bid: float = 0
    ask: float = 0

class RealTimeStreamer:
    """Ultra-fast real-time data streaming"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.streams = {}
        self.callbacks = []
        self.data_buffer = deque(maxlen=10000)
        self.stream_health = {}
        
        # DEX WebSocket endpoints
        self.endpoints = {
            'uniswap_v3': 'wss://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-arbitrum/ws',
            'camelot': 'wss://api.thegraph.com/subgraphs/name/camelotlabs/camelot-amm/ws',
            'pancakeswap': 'wss://api.thegraph.com/subgraphs/name/pancakeswap/exchange-v3-polygon/ws'
        }
    
    async def start_all_streams(self):
        """Start all WebSocket streams concurrently"""
        
        logging.info("🚀 Starting real-time data streams...")
        
        stream_tasks = []
        for source, endpoint in self.endpoints.items():
            task = asyncio.create_task(self.connect_stream(source, endpoint))
            stream_tasks.append(task)
        
        # Start health monitoring
        health_task = asyncio.create_task(self.monitor_stream_health())
        stream_tasks.append(health_task)
        
        await asyncio.gather(*stream_tasks, return_exceptions=True)
    
    async def connect_stream(self, source: str, endpoint: str):
        """Connect to individual WebSocket stream with auto-reconnect"""
        
        while True:
            try:
                logging.info(f"📡 Connecting to {source}...")
                
                async with websockets.connect(endpoint) as websocket:
                    self.stream_health[source] = time.time()
                    
                    # Subscribe to relevant channels
                    subscription = {
                        "id": "1",
                        "type": "start",
                        "payload": {
                            "query": """
                                subscription {
                                    swaps(first: 50, orderBy: timestamp, orderDirection: desc) {
                                        id
                                        timestamp
                                        amount0
                                        amount1
                                        amountUSD
                                        pool {
                                            id
                                            token0 { symbol }
                                            token1 { symbol }
                                            liquidity
                                            sqrtPriceX96
                                        }
                                    }
                                }
                            """
                        }
                    }
                    
                    await websocket.send(json.dumps(subscription))
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self.process_stream_data(source, data)
                            self.stream_health[source] = time.time()
                            
                        except Exception as e:
                            logging.error(f"Stream data processing error: {e}")
                            continue
                            
            except Exception as e:
                logging.error(f"Stream {source} disconnected: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting
    
    async def process_stream_data(self, source: str, raw_data: Dict):
        """Process incoming stream data"""
        
        try:
            swaps_data = raw_data.get('payload', {}).get('data', {}).get('swaps', [])
            
            for swap in swaps_data:
                pool = swap.get('pool', {})
                
                # Extract meaningful data
                stream_data = StreamData(
                    symbol=f"{pool.get('token0', {}).get('symbol', 'UNK')}/{pool.get('token1', {}).get('symbol', 'UNK')}",
                    price=self.calculate_price_from_swap(swap),
                    volume=float(swap.get('amountUSD', 0)),
                    timestamp=float(swap.get('timestamp', time.time())),
                    source=source,
                    liquidity=float(pool.get('liquidity', 0))
                )
                
                # Buffer data
                self.data_buffer.append(stream_data)
                
                # Cache in Redis
                if self.redis_client:
                    cache_key = f"stream:{source}:{pool.get('id', 'unknown')}"
                    self.redis_client.setex(
                        cache_key, 
                        60, 
                        json.dumps({
                            'price': stream_data.price,
                            'volume': stream_data.volume,
                            'timestamp': stream_data.timestamp,
                            'liquidity': stream_data.liquidity
                        })
                    )
                
                # Trigger callbacks
                for callback in self.callbacks:
                    try:
                        await callback(stream_data)
                    except Exception as e:
                        logging.error(f"Callback error: {e}")
                        
        except Exception as e:
            logging.error(f"Stream processing error: {e}")
    
    def calculate_price_from_swap(self, swap: Dict) -> float:
        """Calculate price from swap data"""
        try:
            amount0 = float(swap.get('amount0', 0))
            amount1 = float(swap.get('amount1', 0))
            
            if amount0 != 0 and amount1 != 0:
                return abs(amount1 / amount0)
            
            # Fallback to sqrtPriceX96
            pool = swap.get('pool', {})
            sqrt_price = float(pool.get('sqrtPriceX96', 0))
            if sqrt_price > 0:
                return (sqrt_price / 2**96) ** 2
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def add_callback(self, callback: Callable):
        """Add callback for real-time data"""
        self.callbacks.append(callback)
    
    async def monitor_stream_health(self):
        """Monitor stream health and restart if needed"""
        
        while True:
            try:
                current_time = time.time()
                
                for source, last_update in self.stream_health.items():
                    if current_time - last_update > 30:  # 30 seconds timeout
                        logging.warning(f"⚠️ Stream {source} appears stale")
                        # Could implement restart logic here
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
    
    def get_recent_data(self, symbol: str = None, seconds: int = 60) -> List[StreamData]:
        """Get recent streaming data"""
        
        cutoff_time = time.time() - seconds
        
        if symbol:
            return [data for data in self.data_buffer 
                   if data.timestamp > cutoff_time and data.symbol == symbol]
        else:
            return [data for data in self.data_buffer 
                   if data.timestamp > cutoff_time]
    
    def get_stream_statistics(self) -> Dict:
        """Get streaming statistics"""
        
        current_time = time.time()
        
        return {
            'active_streams': len([s for s, t in self.stream_health.items() 
                                 if current_time - t < 30]),
            'total_streams': len(self.endpoints),
            'buffer_size': len(self.data_buffer),
            'data_rate_per_second': len([d for d in self.data_buffer 
                                       if current_time - d.timestamp < 1]),
            'stream_health': {source: current_time - last_update 
                            for source, last_update in self.stream_health.items()}
        }

class PriceVelocityDetector:
    """Detect rapid price movements in real-time"""
    
    def __init__(self, streamer: RealTimeStreamer):
        self.streamer = streamer
        self.price_history = {}
        self.velocity_thresholds = {
            'breakout': 0.09,      # 9% in short timeframe
            'acceleration': 0.13,   # 13% acceleration
            'momentum_surge': 0.15  # 15% momentum surge
        }
        
        # Register callback
        streamer.add_callback(self.process_price_update)
    
    async def process_price_update(self, data: StreamData):
        """Process real-time price updates"""
        
        symbol = data.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)
        
        self.price_history[symbol].append({
            'price': data.price,
            'volume': data.volume,
            'timestamp': data.timestamp
        })
        
        # Detect velocity patterns
        velocity_signals = self.detect_velocity_patterns(symbol)
        
        if any(velocity_signals.values()):
            logging.info(f"🚀 Velocity signal detected for {symbol}: {velocity_signals}")
            
            # Could trigger trading logic here
            await self.handle_velocity_signal(symbol, velocity_signals, data)
    
    def detect_velocity_patterns(self, symbol: str) -> Dict[str, bool]:
        """Detect various velocity patterns"""
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return {pattern: False for pattern in self.velocity_thresholds.keys()}
        
        prices = [p['price'] for p in self.price_history[symbol]]
        times = [p['timestamp'] for p in self.price_history[symbol]]
        
        # Calculate returns over different timeframes
        returns_1min = self.calculate_returns(prices[-5:])   # Last 5 data points
        returns_5min = self.calculate_returns(prices[-20:])  # Last 20 data points
        returns_total = self.calculate_returns(prices)       # All data points
        
        # Detect patterns
        patterns = {
            'breakout': abs(returns_1min) > self.velocity_thresholds['breakout'],
            'acceleration': abs(returns_5min) > self.velocity_thresholds['acceleration'],
            'momentum_surge': abs(returns_total) > self.velocity_thresholds['momentum_surge']
        }
        
        # Additional pattern: sustained acceleration
        if len(prices) >= 30:
            recent_acceleration = self.calculate_acceleration(prices[-30:])
            patterns['sustained_acceleration'] = recent_acceleration > 0.05
        
        return patterns
    
    def calculate_returns(self, prices: List[float]) -> float:
        """Calculate total returns"""
        if len(prices) < 2:
            return 0.0
        
        return (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0.0
    
    def calculate_acceleration(self, prices: List[float]) -> float:
        """Calculate price acceleration"""
        if len(prices) < 10:
            return 0.0
        
        # Split into three segments and calculate acceleration
        segment_size = len(prices) // 3
        segment1 = prices[:segment_size]
        segment2 = prices[segment_size:2*segment_size]
        segment3 = prices[2*segment_size:]
        
        return1 = self.calculate_returns(segment1)
        return2 = self.calculate_returns(segment2)
        return3 = self.calculate_returns(segment3)
        
        # Calculate acceleration as change in velocity
        velocity1 = return2 - return1
        velocity2 = return3 - return2
        
        return velocity2 - velocity1
    
    async def handle_velocity_signal(self, symbol: str, signals: Dict, data: StreamData):
        """Handle detected velocity signal"""
        
        # This would integrate with the main trading pipeline
        signal_strength = sum(signals.values()) / len(signals)
        
        alert = {
            'symbol': symbol,
            'signal_strength': signal_strength,
            'signals': signals,
            'price': data.price,
            'volume': data.volume,
            'timestamp': data.timestamp,
            'source': data.source
        }
        
        # Could publish to Redis channel for main pipeline
        if self.streamer.redis_client:
            channel = "velocity_signals"
            self.streamer.redis_client.publish(
                channel, 
                json.dumps(alert)
            )

# Integration with existing scanner
async def integrate_streaming_with_scanner():
    """Integrate real-time streaming with existing scanner"""
    
    import redis
    from core.execution.scanner_v3 import backup_20250720_213811.scanner_v3
    
    # Initialize components
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
    streamer = RealTimeStreamer(redis_client)
    velocity_detector = PriceVelocityDetector(streamer)
    
    # Start streaming
    logging.info("🚀 Starting integrated real-time pipeline...")
    await streamer.start_all_streams()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(integrate_streaming_with_scanner())
