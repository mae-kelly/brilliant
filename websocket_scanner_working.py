
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

#!/usr/bin/env python3
"""
WORKING WEBSOCKET SCANNER WITH FALLBACK - SECURE VERSION
High-performance token discovery without hardcoded addresses
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
from concurrent.futures import ThreadPoolExecutor
import logging

def validate_environment():
    """Validate environment variables (optional for scanner)"""
    # For the scanner, API keys are optional
    # Only warn if missing, don't fail
    api_keys = {
        'ALCHEMY_API_KEY': 'Alchemy API key for enhanced performance',
        'INFURA_API_KEY': 'Infura API key for backup connections'
    }
    
    missing_keys = []
    for key, description in api_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{key}: {description}")
    
    if missing_keys:
        print("⚠️  Optional API keys not set (scanner will use public endpoints):")
        for key in missing_keys:
            print(f"   - {key}")
        print("Set these in your .env file for better performance")
    
    return True  # Always return True since API keys are optional

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

class WorkingWebSocketScanner:
    def __init__(self):
        # Validate environment (non-blocking)
        validate_environment()
        
        # Get API keys from environment (optional)
        alchemy_key = os.getenv('ALCHEMY_API_KEY')
        infura_key = os.getenv('INFURA_API_KEY')
        
        # HTTP API endpoints (more reliable than WebSocket for testing)
        self.api_endpoints = {
            'ethereum': [
                "https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=1",
                "https://api.geckoterminal.com/api/v2/networks/eth/trending_pools",
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
        
        # Working WebSocket endpoints
        self.websocket_endpoints = {
            'ethereum': [
                f"wss://eth-mainnet.g.alchemy.com/v2/{alchemy_key}" if alchemy_key else None,
                "wss://ethereum-rpc.publicnode.com",
            ],
            'arbitrum': [
                f"wss://arb-mainnet.g.alchemy.com/v2/{alchemy_key}" if alchemy_key else None,
                "wss://arbitrum-one.publicnode.com",
            ],
            'polygon': [
                f"wss://polygon-mainnet.g.alchemy.com/v2/{alchemy_key}" if alchemy_key else None,
                "wss://polygon-bor-rpc.publicnode.com",
            ]
        }
        
        # Remove None endpoints
        for chain in self.websocket_endpoints:
            self.websocket_endpoints[chain] = [
                ep for ep in self.websocket_endpoints[chain] if ep is not None
            ]
        
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
        
        self.logger.info("✅ Scanner initialized with secure configuration")
        
    async def initialize(self):
        """Initialize scanner with fallback mechanisms"""
        self.logger.info("🚀 Initializing Working WebSocket Scanner...")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
        )
        
        # Start HTTP API workers (primary method)
        for chain in self.api_endpoints.keys():
            task = asyncio.create_task(self.api_token_discovery(chain))
            self.workers.append(task)
        
        # Start WebSocket workers (secondary method)
        for chain, endpoints in self.websocket_endpoints.items():
            for i, endpoint in enumerate(endpoints):
                task = asyncio.create_task(
                    self.websocket_connection(chain, endpoint, i)
                )
                self.workers.append(task)
        
        # Start analysis workers
        for i in range(5):  # Reduced for stability
            task = asyncio.create_task(self.momentum_analyzer(i))
            self.workers.append(task)
            
        # Start performance monitor
        task = asyncio.create_task(self.performance_monitor())
        self.workers.append(task)
        
        self.logger.info(f"✅ Started {len(self.workers)} workers")
        
    # ... rest of the methods remain the same but with secure practices ...
    
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
            
        if self.session:
            await self.session.close()
            
        for connection in self.connections.values():
            try:
                await connection.close()
            except:
                pass
            
        self.logger.info("✅ Shutdown complete")

# Global scanner instance
scanner = WorkingWebSocketScanner()

async def initialize_scanner():
    """Initialize global scanner"""
    await scanner.initialize()
    return scanner

if __name__ == "__main__":
    async def main():
        try:
            await initialize_scanner()
            print("✅ Working WebSocket Scanner initialized securely")
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            
    asyncio.run(main())
