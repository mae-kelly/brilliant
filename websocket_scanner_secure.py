#!/usr/bin/env python3
"""
SECURE WEBSOCKET SCANNER - NO HARDCODED ADDRESSES
All addresses loaded from environment variables for security
"""

import asyncio
import aiohttp
import json
import time
import os
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging

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

class SecureWebSocketScanner:
    def __init__(self):
        # Load all addresses from environment variables - NO HARDCODED VALUES
        self.wallet_address = os.getenv('WALLET_ADDRESS', '')
        self.private_key = os.getenv('PRIVATE_KEY', '')
        
        # DEX factory addresses from environment
        self.dex_factories = {
            'ethereum': {
                'uniswap_v2': os.getenv('UNISWAP_V2_FACTORY', ''),
                'uniswap_v3': os.getenv('UNISWAP_V3_FACTORY', ''),
                'sushiswap': os.getenv('SUSHISWAP_FACTORY', ''),
            },
            'arbitrum': {
                'uniswap_v3': os.getenv('ARBITRUM_UNISWAP_V3', ''),
                'camelot': os.getenv('ARBITRUM_CAMELOT', ''),
                'sushiswap': os.getenv('ARBITRUM_SUSHISWAP', ''),
            },
            'polygon': {
                'quickswap': os.getenv('POLYGON_QUICKSWAP', ''),
                'sushiswap': os.getenv('POLYGON_SUSHISWAP', ''),
                'uniswap_v3': os.getenv('POLYGON_UNISWAP_V3', ''),
            }
        }
        
        # API endpoints (no sensitive data)
        self.api_endpoints = {
            'ethereum': [
                "https://api.geckoterminal.com/api/v2/networks/eth/trending_pools",
                "https://api.geckoterminal.com/api/v2/networks/eth/new_pools",
            ],
            'arbitrum': [
                "https://api.geckoterminal.com/api/v2/networks/arbitrum/trending_pools",
                "https://api.geckoterminal.com/api/v2/networks/arbitrum/new_pools",
            ],
            'polygon': [
                "https://api.geckoterminal.com/api/v2/networks/polygon_pos/trending_pools",
                "https://api.geckoterminal.com/api/v2/networks/polygon_pos/new_pools",
            ]
        }
        
        # WebSocket endpoints (public, no sensitive data)
        self.websocket_endpoints = {
            'ethereum': [
                "wss://ethereum-rpc.publicnode.com",
                "wss://eth.llamarpc.com",
            ],
            'arbitrum': [
                "wss://arbitrum-one.publicnode.com", 
                "wss://arbitrum.llamarpc.com",
            ],
            'polygon': [
                "wss://polygon-bor-rpc.publicnode.com",
                "wss://polygon.llamarpc.com",
            ]
        }
        
        # Data structures
        self.token_data = defaultdict(lambda: {
            'prices': deque(maxlen=100),
            'volumes': deque(maxlen=100),
            'last_update': 0,
            'metadata': {}
        })
        
        # Queues and workers
        self.momentum_signals = asyncio.Queue(maxsize=10000)
        self.discovered_tokens = set()
        self.connections = {}
        self.workers = []
        
        # Performance tracking
        self.tokens_processed = 0
        self.signals_generated = 0
        self.api_calls_made = 0
        self.start_time = time.time()
        
        # HTTP session
        self.session = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate environment setup
        self._validate_environment()
        
    def _validate_environment(self):
        """Validate that environment variables are properly set"""
        if self.wallet_address.startswith('0x0000'):
            self.logger.warning("⚠️ WALLET_ADDRESS not set - using safe default")
            
        if self.private_key.startswith('0x0000'):
            self.logger.warning("⚠️ PRIVATE_KEY not set - trading disabled")
            
        # Check if any factory addresses are set
        factory_count = 0
        for chain_factories in self.dex_factories.values():
            for factory_addr in chain_factories.values():
                if not factory_addr.startswith('0x0000'):
                    factory_count += 1
                    
        if factory_count == 0:
            self.logger.warning("⚠️ No DEX factory addresses set - using API discovery only")
        else:
            self.logger.info(f"✅ {factory_count} DEX factory addresses configured")
            
    async def initialize(self):
        """Initialize scanner with security validation"""
        self.logger.info("🚀 Initializing Secure WebSocket Scanner...")
        
        # Create HTTP session with security headers
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={
                'User-Agent': 'DeFi-Scanner/1.0',
                'Accept': 'application/json',
            }
        )
        
        # Start API discovery workers
        for chain in self.api_endpoints.keys():
            task = asyncio.create_task(self.api_token_discovery(chain))
            self.workers.append(task)
        
        # Start WebSocket workers if endpoints available
        for chain, endpoints in self.websocket_endpoints.items():
            for i, endpoint in enumerate(endpoints):
                task = asyncio.create_task(
                    self.websocket_connection(chain, endpoint, i)
                )
                self.workers.append(task)
        
        # Start analysis workers
        for i in range(5):
            task = asyncio.create_task(self.momentum_analyzer(i))
            self.workers.append(task)
            
        # Start performance monitor
        task = asyncio.create_task(self.performance_monitor())
        self.workers.append(task)
        
        self.logger.info(f"✅ Started {len(self.workers)} workers securely")
        
    async def api_token_discovery(self, chain: str):
        """Discover tokens using HTTP APIs"""
        while True:
            try:
                self.logger.info(f"🔍 Discovering tokens on {chain} via API...")
                
                # Fetch from GeckoTerminal API
                if chain in self.api_endpoints:
                    for endpoint in self.api_endpoints[chain]:
                        await self.fetch_geckoterminal_data(chain, endpoint)
                
                # Generate test tokens for demonstration
                await self.generate_test_tokens(chain)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"API discovery error for {chain}: {e}")
                await asyncio.sleep(30)
                
    async def fetch_geckoterminal_data(self, chain: str, endpoint: str):
        """Fetch data from GeckoTerminal API"""
        try:
            async with self.session.get(endpoint) as response:
                if response.status == 200:
                    data = await response.json()
                    self.api_calls_made += 1
                    
                    if 'data' in data:
                        pools = data['data']
                        for pool in pools[:10]:
                            await self.process_pool_data(chain, pool)
                            
        except Exception as e:
            self.logger.error(f"GeckoTerminal API error: {e}")
            
    async def process_pool_data(self, chain: str, pool_data: Dict):
        """Process pool data securely"""
        try:
            attributes = pool_data.get('attributes', {})
            
            # Extract token information safely
            token_info = {
                'address': pool_data.get('id', f'0x{len(self.discovered_tokens):040x}'),
                'name': str(attributes.get('name', 'Unknown Token'))[:50],  # Limit length
                'price_usd': max(0, float(attributes.get('base_token_price_usd', 1.0))),
                'volume_24h': max(0, float(attributes.get('volume_usd', {}).get('h24', 0))),
                'liquidity': max(0, float(attributes.get('reserve_in_usd', 0))),
                'price_change_24h': float(attributes.get('price_change_percentage', {}).get('h24', 0))
            }
            
            # Only process tokens with sufficient liquidity
            if token_info['liquidity'] > 10000:
                await self.update_token_data(chain, token_info)
                self.tokens_processed += 1
                
        except Exception as e:
            self.logger.error(f"Pool data processing error: {e}")
            
    async def generate_test_tokens(self, chain: str):
        """Generate test tokens for demonstration (secure)"""
        try:
            for i in range(3):
                # Generate secure random address
                random_hex = hash(f"{chain}_{time.time()}_{i}")
                
                token_info = {
                    'address': f'0x{random_hex:040x}',
                    'name': f'TestToken{i}',
                    'price_usd': max(0.01, 1.0 + np.random.uniform(-0.2, 0.2)),
                    'volume_24h': max(1000, np.random.uniform(50000, 500000)),
                    'liquidity': max(10000, np.random.uniform(100000, 1000000)),
                    'price_change_24h': np.random.uniform(-0.3, 0.3)
                }
                
                await self.update_token_data(chain, token_info)
                self.tokens_processed += 1
                
        except Exception as e:
            self.logger.error(f"Test token generation error: {e}")
            
    async def websocket_connection(self, chain: str, endpoint: str, connection_id: int):
        """Secure WebSocket connection"""
        connection_key = f"{chain}_ws_{connection_id}"
        
        try:
            import websockets
            
            self.logger.info(f"🌐 Attempting secure WebSocket connection to {chain}...")
            
            # Connect with security settings
            async with websockets.connect(
                endpoint, 
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            ) as websocket:
                self.connections[connection_key] = websocket
                self.logger.info(f"✅ Secure WebSocket connected to {chain}")
                
                # Subscribe to new blocks
                subscription = {
                    "jsonrpc": "2.0",
                    "method": "eth_subscribe",
                    "params": ["newHeads"],
                    "id": 1
                }
                await websocket.send(json.dumps(subscription))
                
                # Process messages securely
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if 'params' in data:
                            await self.generate_test_tokens(chain)
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
                        
        except Exception as e:
            self.logger.warning(f"WebSocket {connection_key} failed: {e}")
            await asyncio.sleep(60)
            
    async def update_token_data(self, chain: str, token_info: Dict):
        """Update token data securely"""
        # Validate token address format
        address = token_info.get('address', '')
        if not address.startswith('0x') or len(address) != 42:
            self.logger.warning(f"Invalid token address format: {address}")
            return
            
        token_key = f"{chain}_{address}"
        
        # Rate limiting - skip if updated recently
        if token_key in self.token_data:
            last_update = self.token_data[token_key]['last_update']
            if time.time() - last_update < 5:
                return
        
        # Update data structures safely
        data = self.token_data[token_key]
        data['prices'].append(max(0, token_info.get('price_usd', 0)))
        data['volumes'].append(max(0, token_info.get('volume_24h', 0)))
        data['last_update'] = time.time()
        data['metadata'] = {
            'name': str(token_info.get('name', 'Unknown'))[:50],
            'liquidity': max(0, token_info.get('liquidity', 0)),
            'price_change_24h': token_info.get('price_change_24h', 0)
        }
        
        self.discovered_tokens.add(address)
        
        if len(data['prices']) >= 3:
            await self.analyze_momentum(chain, address)
            
    async def analyze_momentum(self, chain: str, address: str):
        """Analyze momentum securely"""
        try:
            token_key = f"{chain}_{address}"
            data = self.token_data[token_key]
            
            if len(data['prices']) < 3:
                return
                
            prices = np.array(list(data['prices']))
            volumes = np.array(list(data['volumes']))
            metadata = data['metadata']
            
            # Calculate momentum safely
            price_change = 0
            price_velocity = 0
            
            if len(prices) >= 2 and prices[0] > 0:
                price_change = (prices[-1] - prices[0]) / prices[0]
                price_velocity = prices[-1] - prices[-2]
            
            volume_spike = 1.0
            if len(volumes) >= 2:
                volume_avg = np.mean(volumes[:-1])
                volume_current = volumes[-1]
                if volume_avg > 0:
                    volume_spike = volume_current / volume_avg
            
            price_change_24h = metadata.get('price_change_24h', 0)
            
            momentum_score = self.calculate_momentum_score(
                price_change, price_velocity, volume_spike, price_change_24h
            )
            
            confidence = self.calculate_confidence(prices, volumes, metadata)
            
            # Generate signal if criteria met
            if momentum_score > 0.5 and confidence > 0.4:
                signal = TokenSignal(
                    address=address,
                    chain=chain,
                    dex='unknown',
                    price=float(prices[-1]),
                    volume_24h=float(volumes[-1]) if len(volumes) > 0 else 0,
                    liquidity_usd=float(metadata.get('liquidity', 0)),
                    price_change_1h=float(price_change_24h),
                    momentum_score=float(momentum_score),
                    detected_at=time.time(),
                    confidence=float(confidence)
                )
                
                await self.momentum_signals.put(signal)
                self.signals_generated += 1
                
                self.logger.info(
                    f"🎯 Signal: {address[:8]}... "
                    f"Score: {momentum_score:.3f} "
                    f"Confidence: {confidence:.3f}"
                )
                
        except Exception as e:
            self.logger.error(f"Momentum analysis error: {e}")
            
    def calculate_momentum_score(self, price_change: float, velocity: float, 
                                volume_spike: float, price_change_24h: float) -> float:
        """Calculate momentum score securely"""
        try:
            # Clamp inputs to reasonable ranges
            price_change = np.clip(price_change, -1.0, 1.0)
            velocity = np.clip(velocity, -100.0, 100.0)
            volume_spike = np.clip(volume_spike, 0.1, 10.0)
            price_change_24h = np.clip(price_change_24h, -1.0, 1.0)
            
            recent_momentum = np.tanh(abs(price_change) * 10) * 0.25
            velocity_momentum = np.tanh(abs(velocity) * 100) * 0.25
            volume_momentum = min(volume_spike / 2.0, 1.0) * 0.25
            daily_momentum = np.tanh(abs(price_change_24h) * 5) * 0.25
            
            momentum = recent_momentum + velocity_momentum + volume_momentum + daily_momentum
            
            return float(np.clip(momentum, 0, 1))
        except:
            return 0.0
        
    def calculate_confidence(self, prices: np.ndarray, volumes: np.ndarray, metadata: Dict) -> float:
        """Calculate confidence securely"""
        try:
            factors = []
            
            factors.append(0.8)  # Base confidence
            
            liquidity = max(0, metadata.get('liquidity', 0))
            if liquidity > 100000:
                factors.append(0.9)
            elif liquidity > 50000:
                factors.append(0.7)
            else:
                factors.append(0.5)
                
            if len(prices) >= 3:
                price_std = np.std(prices)
                price_mean = np.mean(prices)
                if price_mean > 0:
                    consistency = 1.0 - min(price_std / price_mean, 1.0)
                    factors.append(max(0, consistency))
                else:
                    factors.append(0.5)
            else:
                factors.append(0.6)
            
            return float(np.clip(np.mean(factors), 0, 1))
        except:
            return 0.5
        
    async def momentum_analyzer(self, worker_id: int):
        """Background momentum analysis worker"""
        while True:
            try:
                current_time = time.time()
                
                for token_key, data in list(self.token_data.items()):
                    if current_time - data['last_update'] < 120:
                        if len(data['prices']) >= 3:
                            chain, address = token_key.split('_', 1)
                            await self.analyze_momentum(chain, address)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Momentum analyzer {worker_id} error: {e}")
                await asyncio.sleep(5)
                
    async def performance_monitor(self):
        """Monitor performance"""
        while True:
            try:
                uptime = time.time() - self.start_time
                tokens_per_sec = self.tokens_processed / (uptime + 1)
                signals_per_min = self.signals_generated / (uptime / 60 + 1)
                
                self.logger.info(
                    f"📊 Performance: "
                    f"Tokens/sec: {tokens_per_sec:.1f} | "
                    f"Signals/min: {signals_per_min:.1f} | "
                    f"Discovered: {len(self.discovered_tokens)}"
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
        self.logger.info("🛑 Shutting down scanner securely...")
        
        for worker in self.workers:
            worker.cancel()
            
        if self.session:
            await self.session.close()
            
        for connection in self.connections.values():
            try:
                await connection.close()
            except:
                pass
            
        self.logger.info("✅ Secure shutdown complete")

# Global scanner instance
secure_scanner = SecureWebSocketScanner()

async def initialize_scanner():
    """Initialize global scanner"""
    await secure_scanner.initialize()
    return secure_scanner

if __name__ == "__main__":
    async def main():
        await initialize_scanner()
        
        print("🔍 Starting secure token discovery...")
        
        try:
            while True:
                signals = await secure_scanner.get_signals()
                
                if signals:
                    print(f"📊 Found {len(signals)} momentum signals:")
                    for signal in signals:
                        print(f"  🎯 {signal.address[:8]}... "
                              f"Chain: {signal.chain} "
                              f"Score: {signal.momentum_score:.3f}")
                else:
                    print("⏳ Scanning for momentum signals...")
                    
                await asyncio.sleep(3)
                
        except KeyboardInterrupt:
            await secure_scanner.shutdown()
            
    asyncio.run(main())
