#!/usr/bin/env python3
"""
OPTIMIZED WEBSOCKET SCANNER - Fixed Dependencies
High-performance token discovery without problematic packages
"""

import asyncio
import websockets
import aiohttp
import json
import time
import os
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
from web3 import Web3
import uvloop

# Set high-performance event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@dataclass
class TokenSignal:
    address: str
    chain: str
    dex: str
    price: float
    volume_24h: float
    liquidity_usd: float
    price_change_1h: float
    momentum_score: float
    detected_at: float
    confidence: float

class OptimizedWebSocketScanner:
    def __init__(self):
        # WebSocket endpoints (using free/public endpoints to avoid API key issues)
        self.endpoints = {
            'ethereum': [
                "wss://eth.llamarpc.com",
                "wss://ethereum.publicnode.com",
            ],
            'arbitrum': [
                "wss://arb1.arbitrum.io/ws",
                "wss://arbitrum.llamarpc.com",
            ],
            'polygon': [
                "wss://polygon.llamarpc.com",
                "wss://polygon-bor.publicnode.com",
            ]
        }
        
        # Data structures
        self.token_data = defaultdict(lambda: {
            'prices': deque(maxlen=100),
            'volumes': deque(maxlen=100),
            'last_update': 0
        })
        
        # Queues and workers
        self.momentum_signals = asyncio.Queue(maxsize=10000)
        self.connections = {}
        self.workers = []
        
        # Performance tracking
        self.tokens_processed = 0
        self.signals_generated = 0
        self.start_time = time.time()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize WebSocket connections"""
        self.logger.info("🚀 Initializing Optimized WebSocket Scanner...")
        
        # Start WebSocket connections
        for chain, endpoints in self.endpoints.items():
            for i, endpoint in enumerate(endpoints):
                task = asyncio.create_task(
                    self.maintain_connection(chain, endpoint, i)
                )
                self.workers.append(task)
        
        # Start analysis workers
        for i in range(10):  # Reduced worker count for stability
            task = asyncio.create_task(self.momentum_analyzer(i))
            self.workers.append(task)
            
        # Start performance monitor
        task = asyncio.create_task(self.performance_monitor())
        self.workers.append(task)
        
        self.logger.info(f"✅ Started {len(self.workers)} workers")
        
    async def maintain_connection(self, chain: str, endpoint: str, connection_id: int):
        """Maintain WebSocket connection with auto-reconnection"""
        connection_key = f"{chain}_{connection_id}"
        reconnect_delay = 1
        
        while True:
            try:
                self.logger.info(f"🌐 Connecting to {chain} WebSocket {connection_id}...")
                
                async with websockets.connect(
                    endpoint,
                    ping_interval=20,
                    ping_timeout=10
                ) as websocket:
                    
                    # Subscribe to new blocks for token discovery
                    subscription = {
                        "jsonrpc": "2.0",
                        "method": "eth_subscribe",
                        "params": ["newHeads"],
                        "id": 1
                    }
                    await websocket.send(json.dumps(subscription))
                    
                    self.connections[connection_key] = websocket
                    reconnect_delay = 1
                    
                    self.logger.info(f"✅ Connected to {chain}")
                    
                    # Process messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self.process_message(chain, data)
                        except:
                            continue
                            
            except Exception as e:
                self.logger.error(f"WebSocket {connection_key} error: {e}")
                self.connections.pop(connection_key, None)
                
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)
                
    async def process_message(self, chain: str, data: Dict):
        """Process WebSocket messages"""
        if 'params' in data and 'result' in data['params']:
            # New block detected - simulate token discovery
            await self.discover_tokens(chain)
            
    async def discover_tokens(self, chain: str):
        """Simulate token discovery (replace with real DEX API calls)"""
        try:
            # For demo purposes, simulate discovering tokens
            # In production, this would query DEX APIs for new pairs
            
            mock_tokens = [
                {
                    'address': f'0x{i:040x}',
                    'price': 1.0 + np.random.uniform(-0.1, 0.1),
                    'volume_24h': np.random.uniform(10000, 1000000),
                    'liquidity': np.random.uniform(50000, 500000),
                    'price_change_1h': np.random.uniform(-0.2, 0.2)
                }
                for i in range(10)
            ]
            
            for token in mock_tokens:
                await self.update_token_data(chain, token)
                self.tokens_processed += 1
                
        except Exception as e:
            self.logger.error(f"Token discovery error: {e}")
            
    async def update_token_data(self, chain: str, token_data: Dict):
        """Update token data and trigger analysis"""
        token_key = f"{chain}_{token_data['address']}"
        
        # Update data structures
        data = self.token_data[token_key]
        data['prices'].append(token_data['price'])
        data['volumes'].append(token_data['volume_24h'])
        data['last_update'] = time.time()
        
        # Trigger momentum analysis if enough data
        if len(data['prices']) >= 10:
            await self.analyze_momentum(chain, token_data['address'], token_data)
            
    async def analyze_momentum(self, chain: str, address: str, token_data: Dict):
        """Analyze token momentum"""
        try:
            token_key = f"{chain}_{address}"
            data = self.token_data[token_key]
            
            if len(data['prices']) < 10:
                return
                
            # Calculate momentum metrics
            prices = np.array(list(data['prices']))
            volumes = np.array(list(data['volumes']))
            
            # Price momentum
            price_change = (prices[-1] - prices[0]) / prices[0]
            price_velocity = np.mean(np.diff(prices[-5:]))
            
            # Volume analysis
            volume_avg = np.mean(volumes)
            volume_recent = np.mean(volumes[-3:])
            volume_spike = volume_recent / (volume_avg + 1e-6)
            
            # Calculate momentum score
            momentum_score = self.calculate_momentum_score(
                price_change, price_velocity, volume_spike
            )
            
            # Calculate confidence
            confidence = self.calculate_confidence(prices, volumes)
            
            # Generate signal if criteria met
            if momentum_score > 0.7 and confidence > 0.6:
                signal = TokenSignal(
                    address=address,
                    chain=chain,
                    dex='uniswap',  # Would determine from source
                    price=float(prices[-1]),
                    volume_24h=float(token_data.get('volume_24h', 0)),
                    liquidity_usd=float(token_data.get('liquidity', 0)),
                    price_change_1h=float(token_data.get('price_change_1h', 0)),
                    momentum_score=float(momentum_score),
                    detected_at=time.time(),
                    confidence=float(confidence)
                )
                
                await self.momentum_signals.put(signal)
                self.signals_generated += 1
                
        except Exception as e:
            self.logger.error(f"Momentum analysis error: {e}")
            
    def calculate_momentum_score(self, price_change: float, velocity: float, volume_spike: float) -> float:
        """Calculate momentum score 0-1"""
        # Price momentum component (30%)
        price_momentum = np.tanh(price_change * 10) * 0.3
        
        # Velocity component (30%)
        velocity_momentum = np.tanh(velocity * 100) * 0.3
        
        # Volume component (40%)
        volume_momentum = min(volume_spike / 3.0, 1.0) * 0.4
        
        # Combine components
        momentum = price_momentum + velocity_momentum + volume_momentum
        
        return float(np.clip(momentum, 0, 1))
        
    def calculate_confidence(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate confidence in signal"""
        factors = []
        
        # Data quality
        if len(prices) >= 20:
            factors.append(0.9)
        else:
            factors.append(0.6)
            
        # Price consistency
        price_std = np.std(prices) / (np.mean(prices) + 1e-6)
        consistency = 1.0 - min(price_std, 1.0)
        factors.append(consistency)
        
        # Volume consistency
        if len(volumes) > 0:
            volume_consistency = 1.0 - np.std(volumes) / (np.mean(volumes) + 1e-6)
            factors.append(min(volume_consistency, 1.0))
        
        return float(np.mean(factors))
        
    async def momentum_analyzer(self, worker_id: int):
        """Background momentum analysis worker"""
        while True:
            try:
                # Process existing token data
                current_time = time.time()
                
                for token_key, data in list(self.token_data.items()):
                    if current_time - data['last_update'] < 60:  # Recent activity
                        if len(data['prices']) >= 10:
                            # Re-analyze for momentum changes
                            chain, address = token_key.split('_', 1)
                            await self.analyze_momentum(chain, address, {
                                'volume_24h': data['volumes'][-1] if data['volumes'] else 0,
                                'liquidity': 100000,  # Mock data
                                'price_change_1h': 0.05  # Mock data
                            })
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Momentum analyzer {worker_id} error: {e}")
                await asyncio.sleep(5)
                
    async def performance_monitor(self):
        """Monitor and report performance"""
        while True:
            try:
                uptime = time.time() - self.start_time
                tokens_per_sec = self.tokens_processed / (uptime + 1)
                signals_per_min = self.signals_generated / (uptime / 60 + 1)
                
                self.logger.info(
                    f"📊 Performance: "
                    f"Tokens/sec: {tokens_per_sec:.1f} | "
                    f"Signals/min: {signals_per_min:.1f} | "
                    f"Active tokens: {len(self.token_data)} | "
                    f"Connections: {len(self.connections)}"
                )
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(60)
                
    async def get_signals(self, max_signals: int = 10) -> List[TokenSignal]:
        """Get momentum signals"""
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
        """Gracefully shutdown"""
        self.logger.info("🛑 Shutting down scanner...")
        
        for worker in self.workers:
            worker.cancel()
            
        for connection in self.connections.values():
            await connection.close()
            
        self.logger.info("✅ Shutdown complete")

# Global scanner instance
scanner = OptimizedWebSocketScanner()

async def initialize_scanner():
    """Initialize global scanner"""
    await scanner.initialize()
    return scanner

if __name__ == "__main__":
    async def main():
        await initialize_scanner()
        
        print("🔍 Starting optimized momentum detection...")
        
        try:
            while True:
                signals = await scanner.get_signals()
                
                if signals:
                    print(f"📊 Found {len(signals)} momentum signals:")
                    for signal in signals:
                        print(f"  🎯 {signal.address[:8]}... "
                              f"Chain: {signal.chain} "
                              f"Score: {signal.momentum_score:.3f} "
                              f"Confidence: {signal.confidence:.3f}")
                else:
                    print("⏳ Scanning for momentum signals...")
                    
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            await scanner.shutdown()
            
    asyncio.run(main())
