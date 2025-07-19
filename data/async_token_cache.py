import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import aiosqlite
import asyncio
import time
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import numpy as np

@dataclass
class TokenCacheEntry:
    token_address: str
    chain: str
    name: str
    symbol: str
    decimals: int
    price: float
    volume_24h: float
    liquidity_usd: float
    market_cap: float
    features: Dict[str, float]
    safety_score: float
    momentum_score: float
    created_at: float
    updated_at: float
    ttl: float

class AsyncTokenCache:
    def __init__(self, db_path: str = './cache/async_token_cache.db'):
        self.db_path = db_path
        self.connection_pool = []
        self.pool_size = 20
        self.lock = asyncio.Lock()
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'writes': 0,
            'reads': 0
        }
        
    async def initialize(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA synchronous=NORMAL')
            await conn.execute('PRAGMA cache_size=10000')
            await conn.execute('PRAGMA temp_store=memory')
            self.connection_pool.append(conn)
        
        await self.create_tables()
        
        asyncio.create_task(self.cleanup_expired_entries())

    async def get_connection(self):
        while not self.connection_pool:
            await asyncio.sleep(0.01)
        return self.connection_pool.pop()

    async def return_connection(self, conn):
        self.connection_pool.append(conn)

    async def create_tables(self):
        conn = await self.get_connection()
        try:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS token_cache (
                    token_address TEXT,
                    chain TEXT,
                    name TEXT,
                    symbol TEXT,
                    decimals INTEGER,
                    price REAL,
                    volume_24h REAL,
                    liquidity_usd REAL,
                    market_cap REAL,
                    features TEXT,
                    safety_score REAL,
                    momentum_score REAL,
                    created_at REAL,
                    updated_at REAL,
                    ttl REAL,
                    PRIMARY KEY (token_address, chain)
                )
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_token_chain ON token_cache(token_address, chain)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_updated_at ON token_cache(updated_at)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_momentum_score ON token_cache(momentum_score)
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT,
                    chain TEXT,
                    price REAL,
                    volume REAL,
                    timestamp REAL,
                    block_number INTEGER
                )
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_price_token_time ON price_history(token_address, chain, timestamp)
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS signal_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT,
                    chain TEXT,
                    signal_type TEXT,
                    signal_data TEXT,
                    confidence REAL,
                    created_at REAL,
                    expires_at REAL
                )
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_signal_expires ON signal_cache(expires_at)
            ''')
            
            await conn.commit()
            
        finally:
            await self.return_connection(conn)

    async def cache_token(self, entry: TokenCacheEntry):
        conn = await self.get_connection()
        try:
            await conn.execute('''
                INSERT OR REPLACE INTO token_cache (
                    token_address, chain, name, symbol, decimals, price,
                    volume_24h, liquidity_usd, market_cap, features,
                    safety_score, momentum_score, created_at, updated_at, ttl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.token_address, entry.chain, entry.name, entry.symbol,
                entry.decimals, entry.price, entry.volume_24h, entry.liquidity_usd,
                entry.market_cap, json.dumps(entry.features), entry.safety_score,
                entry.momentum_score, entry.created_at, entry.updated_at, entry.ttl
            ))
            
            await conn.commit()
            self.stats['writes'] += 1
            
        finally:
            await self.return_connection(conn)

    async def get_token(self, token_address: str, chain: str) -> Optional[TokenCacheEntry]:
        conn = await self.get_connection()
        try:
            cursor = await conn.execute('''
                SELECT * FROM token_cache 
                WHERE token_address = ? AND chain = ? AND updated_at + ttl > ?
            ''', (token_address, chain, time.time()))
            
            row = await cursor.fetchone()
            self.stats['reads'] += 1
            
            if row:
                self.stats['cache_hits'] += 1
                return TokenCacheEntry(
                    token_address=row[0],
                    chain=row[1],
                    name=row[2],
                    symbol=row[3],
                    decimals=row[4],
                    price=row[5],
                    volume_24h=row[6],
                    liquidity_usd=row[7],
                    market_cap=row[8],
                    features=json.loads(row[9]) if row[9] else {},
                    safety_score=row[10],
                    momentum_score=row[11],
                    created_at=row[12],
                    updated_at=row[13],
                    ttl=row[14]
                )
            else:
                self.stats['cache_misses'] += 1
                return None
                
        finally:
            await self.return_connection(conn)

    async def get_top_momentum_tokens(self, chain: str = None, limit: int = 100) -> List[TokenCacheEntry]:
        conn = await self.get_connection()
        try:
            if chain:
                cursor = await conn.execute('''
                    SELECT * FROM token_cache 
                    WHERE chain = ? AND updated_at + ttl > ?
                    ORDER BY momentum_score DESC LIMIT ?
                ''', (chain, time.time(), limit))
            else:
                cursor = await conn.execute('''
                    SELECT * FROM token_cache 
                    WHERE updated_at + ttl > ?
                    ORDER BY momentum_score DESC LIMIT ?
                ''', (time.time(), limit))
            
            rows = await cursor.fetchall()
            
            return [
                TokenCacheEntry(
                    token_address=row[0],
                    chain=row[1],
                    name=row[2],
                    symbol=row[3],
                    decimals=row[4],
                    price=row[5],
                    volume_24h=row[6],
                    liquidity_usd=row[7],
                    market_cap=row[8],
                    features=json.loads(row[9]) if row[9] else {},
                    safety_score=row[10],
                    momentum_score=row[11],
                    created_at=row[12],
                    updated_at=row[13],
                    ttl=row[14]
                ) for row in rows
            ]
            
        finally:
            await self.return_connection(conn)

    async def cache_price_point(self, token_address: str, chain: str, price: float, 
                               volume: float, block_number: int = 0):
        conn = await self.get_connection()
        try:
            await conn.execute('''
                INSERT INTO price_history (token_address, chain, price, volume, timestamp, block_number)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (token_address, chain, price, volume, time.time(), block_number))
            
            await conn.commit()
            
        finally:
            await self.return_connection(conn)

    async def get_price_history(self, token_address: str, chain: str, 
                               hours: int = 24) -> List[Dict]:
        conn = await self.get_connection()
        try:
            since = time.time() - (hours * 3600)
            cursor = await conn.execute('''
                SELECT price, volume, timestamp, block_number FROM price_history
                WHERE token_address = ? AND chain = ? AND timestamp > ?
                ORDER BY timestamp ASC
            ''', (token_address, chain, since))
            
            rows = await cursor.fetchall()
            
            return [
                {
                    'price': row[0],
                    'volume': row[1],
                    'timestamp': row[2],
                    'block_number': row[3]
                } for row in rows
            ]
            
        finally:
            await self.return_connection(conn)

    async def cache_signal(self, token_address: str, chain: str, signal_type: str, 
                          signal_data: Dict, confidence: float, ttl: int = 300):
        conn = await self.get_connection()
        try:
            expires_at = time.time() + ttl
            
            await conn.execute('''
                INSERT INTO signal_cache (token_address, chain, signal_type, signal_data, 
                                        confidence, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (token_address, chain, signal_type, json.dumps(signal_data), 
                  confidence, time.time(), expires_at))
            
            await conn.commit()
            
        finally:
            await self.return_connection(conn)

    async def get_cached_signal(self, token_address: str, chain: str, 
                               signal_type: str) -> Optional[Dict]:
        conn = await self.get_connection()
        try:
            cursor = await conn.execute('''
                SELECT signal_data, confidence, created_at FROM signal_cache
                WHERE token_address = ? AND chain = ? AND signal_type = ? AND expires_at > ?
                ORDER BY created_at DESC LIMIT 1
            ''', (token_address, chain, signal_type, time.time()))
            
            row = await cursor.fetchone()
            
            if row:
                return {
                    'signal_data': json.loads(row[0]),
                    'confidence': row[1],
                    'created_at': row[2]
                }
            
            return None
            
        finally:
            await self.return_connection(conn)

    async def batch_cache_tokens(self, entries: List[TokenCacheEntry]):
        conn = await self.get_connection()
        try:
            data = [
                (
                    entry.token_address, entry.chain, entry.name, entry.symbol,
                    entry.decimals, entry.price, entry.volume_24h, entry.liquidity_usd,
                    entry.market_cap, json.dumps(entry.features), entry.safety_score,
                    entry.momentum_score, entry.created_at, entry.updated_at, entry.ttl
                ) for entry in entries
            ]
            
            await conn.executemany('''
                INSERT OR REPLACE INTO token_cache (
                    token_address, chain, name, symbol, decimals, price,
                    volume_24h, liquidity_usd, market_cap, features,
                    safety_score, momentum_score, created_at, updated_at, ttl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)
            
            await conn.commit()
            self.stats['writes'] += len(entries)
            
        finally:
            await self.return_connection(conn)

    async def search_tokens(self, query: str, chain: str = None, limit: int = 50) -> List[TokenCacheEntry]:
        conn = await self.get_connection()
        try:
            query_lower = query.lower()
            
            if chain:
                cursor = await conn.execute('''
                    SELECT * FROM token_cache 
                    WHERE chain = ? AND updated_at + ttl > ? AND (
                        LOWER(name) LIKE ? OR LOWER(symbol) LIKE ? OR token_address LIKE ?
                    )
                    ORDER BY momentum_score DESC LIMIT ?
                ''', (chain, time.time(), f'%{query_lower}%', f'%{query_lower}%', f'%{query}%', limit))
            else:
                cursor = await conn.execute('''
                    SELECT * FROM token_cache 
                    WHERE updated_at + ttl > ? AND (
                        LOWER(name) LIKE ? OR LOWER(symbol) LIKE ? OR token_address LIKE ?
                    )
                    ORDER BY momentum_score DESC LIMIT ?
                ''', (time.time(), f'%{query_lower}%', f'%{query_lower}%', f'%{query}%', limit))
            
            rows = await cursor.fetchall()
            
            return [
                TokenCacheEntry(
                    token_address=row[0],
                    chain=row[1],
                    name=row[2],
                    symbol=row[3],
                    decimals=row[4],
                    price=row[5],
                    volume_24h=row[6],
                    liquidity_usd=row[7],
                    market_cap=row[8],
                    features=json.loads(row[9]) if row[9] else {},
                    safety_score=row[10],
                    momentum_score=row[11],
                    created_at=row[12],
                    updated_at=row[13],
                    ttl=row[14]
                ) for row in rows
            ]
            
        finally:
            await self.return_connection(conn)

    async def cleanup_expired_entries(self):
        while True:
            try:
                await asyncio.sleep(300)
                
                conn = await self.get_connection()
                try:
                    current_time = time.time()
                    
                    await conn.execute('''
                        DELETE FROM token_cache WHERE updated_at + ttl < ?
                    ''', (current_time,))
                    
                    await conn.execute('''
                        DELETE FROM signal_cache WHERE expires_at < ?
                    ''', (current_time,))
                    
                    await conn.execute('''
                        DELETE FROM price_history WHERE timestamp < ?
                    ''', (current_time - 86400 * 7,))
                    
                    await conn.commit()
                    
                finally:
                    await self.return_connection(conn)
                    
            except Exception as e:
                await asyncio.sleep(60)

    async def get_cache_stats(self) -> Dict:
        conn = await self.get_connection()
        try:
            cursor = await conn.execute('SELECT COUNT(*) FROM token_cache')
            token_count = (await cursor.fetchone())[0]
            
            cursor = await conn.execute('SELECT COUNT(*) FROM price_history')
            price_count = (await cursor.fetchone())[0]
            
            cursor = await conn.execute('SELECT COUNT(*) FROM signal_cache')
            signal_count = (await cursor.fetchone())[0]
            
            hit_rate = self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
            
            return {
                'token_entries': token_count,
                'price_entries': price_count,
                'signal_entries': signal_count,
                'cache_hit_rate': hit_rate,
                'total_reads': self.stats['reads'],
                'total_writes': self.stats['writes'],
                'pool_size': len(self.connection_pool)
            }
            
        finally:
            await self.return_connection(conn)

    async def close(self):
        for conn in self.connection_pool:
            await conn.close()
        self.connection_pool.clear()

async_token_cache = AsyncTokenCache()
