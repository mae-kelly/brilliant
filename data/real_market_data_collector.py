import asyncio
import aiohttp
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import json
import os
from decimal import Decimal
import sqlite3

@dataclass
class TokenMarketData:
    address: str
    chain: str
    symbol: str
    name: str
    price_usd: float
    volume_24h: float
    liquidity_usd: float
    price_change_24h: float
    market_cap: float
    holder_count: int
    tx_count_24h: int
    created_at: float
    dex_screener_data: Optional[Dict] = None
    coingecko_data: Optional[Dict] = None

class RealMarketDataCollector:
    def __init__(self):
        self.session = None
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY', '')
        self.dexscreener_base = "https://api.dexscreener.com/latest"
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.rate_limits = {
            'coingecko': {'calls': 0, 'reset_time': 0, 'max_per_minute': 30},
            'dexscreener': {'calls': 0, 'reset_time': 0, 'max_per_minute': 300}
        }
        self.cache = {}
        self.cache_ttl = 30
        
    async def initialize(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    async def close(self):
        if self.session:
            await self.session.close()

    async def check_rate_limit(self, service: str) -> bool:
        current_time = time.time()
        rate_info = self.rate_limits[service]
        
        if current_time > rate_info['reset_time']:
            rate_info['calls'] = 0
            rate_info['reset_time'] = current_time + 60
        
        if rate_info['calls'] >= rate_info['max_per_minute']:
            return False
        
        rate_info['calls'] += 1
        return True

    async def get_dexscreener_pairs(self, chain: str, limit: int = 100) -> List[Dict]:
        if not await self.check_rate_limit('dexscreener'):
            await asyncio.sleep(1)
            return []

        cache_key = f"dexscreener_pairs_{chain}_{int(time.time() // self.cache_ttl)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = f"{self.dexscreener_base}/dex/tokens/{chain}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])[:limit]
                    self.cache[cache_key] = pairs
                    return pairs
                elif response.status == 429:
                    await asyncio.sleep(2)
                    return []
        except Exception as e:
            return []
        
        return []

    async def get_coingecko_token_data(self, token_id: str) -> Optional[Dict]:
        if not await self.check_rate_limit('coingecko'):
            await asyncio.sleep(2)
            return None

        cache_key = f"coingecko_{token_id}_{int(time.time() // self.cache_ttl)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        headers = {}
        if self.coingecko_api_key:
            headers['X-CG-Pro-API-Key'] = self.coingecko_api_key

        url = f"{self.coingecko_base}/coins/{token_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'false',
            'developer_data': 'false'
        }

        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.cache[cache_key] = data
                    return data
                elif response.status == 429:
                    await asyncio.sleep(3)
                    return None
        except Exception as e:
            return None
        
        return None

    async def get_coingecko_trending(self) -> List[Dict]:
        if not await self.check_rate_limit('coingecko'):
            await asyncio.sleep(2)
            return []

        cache_key = f"coingecko_trending_{int(time.time() // 300)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        headers = {}
        if self.coingecko_api_key:
            headers['X-CG-Pro-API-Key'] = self.coingecko_api_key

        url = f"{self.coingecko_base}/search/trending"

        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    trending = data.get('coins', [])
                    self.cache[cache_key] = trending
                    return trending
        except Exception as e:
            return []
        
        return []

    async def get_token_historical_data(self, token_address: str, chain: str, days: int = 7) -> List[Dict]:
        historical_data = []
        
        dex_data = await self.get_dexscreener_token_history(token_address, chain, days)
        if dex_data:
            historical_data.extend(dex_data)
        
        return historical_data

    async def get_dexscreener_token_history(self, token_address: str, chain: str, days: int) -> List[Dict]:
        if not await self.check_rate_limit('dexscreener'):
            await asyncio.sleep(1)
            return []

        url = f"{self.dexscreener_base}/dex/pairs/{chain}/{token_address}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    history = []
                    for pair in pairs[:5]:
                        if 'priceHistory' in pair:
                            for point in pair['priceHistory']:
                                history.append({
                                    'timestamp': point.get('timestamp'),
                                    'price': float(point.get('price', 0)),
                                    'volume': float(pair.get('volume', {}).get('h24', 0)),
                                    'liquidity': float(pair.get('liquidity', {}).get('usd', 0))
                                })
                    
                    return sorted(history, key=lambda x: x['timestamp'])
        except Exception as e:
            return []
        
        return []

    async def collect_token_data(self, token_address: str, chain: str) -> Optional[TokenMarketData]:
        dex_data = await self.get_dexscreener_pairs(chain)
        coingecko_data = None
        
        token_pair = None
        for pair in dex_data:
            if (pair.get('baseToken', {}).get('address', '').lower() == token_address.lower() or
                pair.get('quoteToken', {}).get('address', '').lower() == token_address.lower()):
                token_pair = pair
                break
        
        if not token_pair:
            return None

        base_token = token_pair.get('baseToken', {})
        quote_token = token_pair.get('quoteToken', {})
        
        target_token = base_token if base_token.get('address', '').lower() == token_address.lower() else quote_token
        
        symbol = target_token.get('symbol', 'UNKNOWN')
        name = target_token.get('name', 'Unknown Token')
        
        try:
            coingecko_data = await self.get_coingecko_token_data(symbol.lower())
        except:
            pass

        price_usd = float(token_pair.get('priceUsd', 0))
        volume_24h = float(token_pair.get('volume', {}).get('h24', 0))
        liquidity_usd = float(token_pair.get('liquidity', {}).get('usd', 0))
        
        price_change_24h = 0
        if 'priceChange' in token_pair:
            price_change_24h = float(token_pair['priceChange'].get('h24', 0))

        market_cap = 0
        if coingecko_data:
            market_data = coingecko_data.get('market_data', {})
            if market_data.get('market_cap', {}).get('usd'):
                market_cap = float(market_data['market_cap']['usd'])

        holder_count = 0
        tx_count_24h = int(token_pair.get('txns', {}).get('h24', {}).get('buys', 0) + 
                          token_pair.get('txns', {}).get('h24', {}).get('sells', 0))

        return TokenMarketData(
            address=token_address,
            chain=chain,
            symbol=symbol,
            name=name,
            price_usd=price_usd,
            volume_24h=volume_24h,
            liquidity_usd=liquidity_usd,
            price_change_24h=price_change_24h,
            market_cap=market_cap,
            holder_count=holder_count,
            tx_count_24h=tx_count_24h,
            created_at=time.time(),
            dex_screener_data=token_pair,
            coingecko_data=coingecko_data
        )

    async def batch_collect_tokens(self, token_addresses: List[str], chain: str) -> List[TokenMarketData]:
        semaphore = asyncio.Semaphore(10)
        
        async def collect_single(address):
            async with semaphore:
                return await self.collect_token_data(address, chain)
        
        tasks = [collect_single(addr) for addr in token_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, TokenMarketData):
                valid_results.append(result)
        
        return valid_results

    async def get_trending_tokens(self, chains: List[str] = ['ethereum', 'arbitrum', 'polygon']) -> List[TokenMarketData]:
        all_tokens = []
        
        for chain in chains:
            pairs = await self.get_dexscreener_pairs(chain, 50)
            
            for pair in pairs:
                base_token = pair.get('baseToken', {})
                quote_token = pair.get('quoteToken', {})
                
                for token in [base_token, quote_token]:
                    if token.get('address') and token.get('symbol'):
                        token_data = await self.collect_token_data(token['address'], chain)
                        if token_data and token_data.volume_24h > 10000:
                            all_tokens.append(token_data)
        
        return sorted(all_tokens, key=lambda x: x.volume_24h, reverse=True)[:100]

    def save_to_database(self, tokens: List[TokenMarketData], db_path: str = 'data/market_data.db'):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS token_market_data (
                address TEXT,
                chain TEXT,
                symbol TEXT,
                name TEXT,
                price_usd REAL,
                volume_24h REAL,
                liquidity_usd REAL,
                price_change_24h REAL,
                market_cap REAL,
                holder_count INTEGER,
                tx_count_24h INTEGER,
                created_at REAL,
                dex_screener_data TEXT,
                coingecko_data TEXT,
                PRIMARY KEY (address, chain, created_at)
            )
        ''')
        
        conn.execute('CREATE INDEX IF NOT EXISTS idx_chain_volume ON token_market_data (chain, volume_24h DESC)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_price_change ON token_market_data (price_change_24h DESC)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON token_market_data (created_at DESC)')
        
        for token in tokens:
            conn.execute('''
                INSERT OR REPLACE INTO token_market_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                token.address, token.chain, token.symbol, token.name,
                token.price_usd, token.volume_24h, token.liquidity_usd,
                token.price_change_24h, token.market_cap, token.holder_count,
                token.tx_count_24h, token.created_at,
                json.dumps(token.dex_screener_data) if token.dex_screener_data else None,
                json.dumps(token.coingecko_data) if token.coingecko_data else None
            ))
        
        conn.commit()
        conn.close()

real_market_collector = RealMarketDataCollector()