import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from optimizer import get_dynamic_config, update_performance
except ImportError:
    def get_dynamic_config(): return {"volatility_threshold": 0.1, "confidence_threshold": 0.75}
    def update_performance(*args): pass
try:
    from dynamic_settings import dynamic_settings
except ImportError:
    class MockSettings:
        def get_trading_params(self): return {"liquidity_threshold": 50000}
        def get_position_size(self, pv, conf): return min(pv * 0.1, 1.0)
    dynamic_settings = MockSettings()
import aiosqlite
import json
import time
from typing import Dict, Optional, List
import os

class AsyncTokenCache:
    def __init__(self, db_path='cache/async_cache.db'):
        self.db_path = db_path
        self.connection = None
    
    async def initialize(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.connection = await aiosqlite.connect(self.db_path)
        
        await self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS token_cache (
                key TEXT PRIMARY KEY,
                data TEXT,
                timestamp REAL,
                ttl INTEGER DEFAULT 300
            );
            
            CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON token_cache(timestamp);
        ''')
        await self.connection.commit()
    
    async def get(self, key: str) -> Optional[Dict]:
        if not self.connection:
            await self.initialize()
        
        cursor = await self.connection.execute('''
            SELECT data, timestamp, ttl FROM token_cache 
            WHERE key = ? AND timestamp + ttl > ?
        ''', (key, time.time()))
        
        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    async def set(self, key: str, data: Dict, ttl: int = 300):
        if not self.connection:
            await self.initialize()
        
        await self.connection.execute('''
            INSERT OR REPLACE INTO token_cache (key, data, timestamp, ttl)
            VALUES (?, ?, ?, ?)
        ''', (key, json.dumps(data), time.time(), ttl))
        await self.connection.commit()
    
    async def cleanup_expired(self):
        if not self.connection:
            return
        
        await self.connection.execute('''
            DELETE FROM token_cache WHERE timestamp + ttl < ?
        ''', (time.time(),))
        await self.connection.commit()
    
    async def close(self):
        if self.connection:
            await self.connection.close()

async_token_cache = AsyncTokenCache()
