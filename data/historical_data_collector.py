import asyncio
import aiohttp
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import logging
from dataclasses import dataclass
import json

@dataclass
class HistoricalTokenData:
    token_address: str
    timestamp: int
    price_usd: float
    volume_24h: float
    market_cap: float
    tvl: float
    holders_count: int
    price_change_1h: float
    price_change_24h: float
    price_change_7d: float
    momentum_score: float
    outcome_24h: int

class HistoricalDataCollector:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        self.apis = {
            'coingecko': 'https://pro-api.coingecko.com/api/v3',
            'defillama': 'https://api.llama.fi',
            'moralis': 'https://deep-index.moralis.io/api/v2',
            'graph': 'https://api.thegraph.com/subgraphs/name'
        }
        
        self.api_keys = {
            'coingecko': os.getenv('COINGECKO_PRO_KEY'),
            'moralis': os.getenv('MORALIS_API_KEY')
        }

    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            self.db_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        connector = aiohttp.TCPConnector(limit=100)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        await self.create_tables()

    async def create_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS historical_tokens (
                    id SERIAL PRIMARY KEY,
                    token_address VARCHAR(42) NOT NULL,
                    timestamp BIGINT NOT NULL,
                    price_usd DECIMAL(20,10),
                    volume_24h DECIMAL(20,2),
                    market_cap DECIMAL(20,2),
                    tvl DECIMAL(20,2),
                    holders_count INTEGER,
                    price_change_1h DECIMAL(10,4),
                    price_change_24h DECIMAL(10,4),
                    price_change_7d DECIMAL(10,4),
                    momentum_score DECIMAL(5,4),
                    outcome_24h INTEGER,
                    features JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_historical_tokens_address_timestamp 
                ON historical_tokens(token_address, timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_historical_tokens_outcome 
                ON historical_tokens(outcome_24h);
                
                CREATE TABLE IF NOT EXISTS whale_wallets (
                    id SERIAL PRIMARY KEY,
                    wallet_address VARCHAR(42) UNIQUE NOT NULL,
                    total_value_usd DECIMAL(20,2),
                    token_count INTEGER,
                    cluster_id INTEGER,
                    first_seen TIMESTAMP,
                    last_active TIMESTAMP,
                    risk_score DECIMAL(5,4),
                    tags JSONB
                );
                
                CREATE TABLE IF NOT EXISTS token_correlations (
                    id SERIAL PRIMARY KEY,
                    token_a VARCHAR(42) NOT NULL,
                    token_b VARCHAR(42) NOT NULL,
                    correlation_1h DECIMAL(8,6),
                    correlation_24h DECIMAL(8,6),
                    correlation_7d DECIMAL(8,6),
                    timestamp BIGINT NOT NULL,
                    UNIQUE(token_a, token_b, timestamp)
                );
                
                CREATE TABLE IF NOT EXISTS liquidity_metrics (
                    id SERIAL PRIMARY KEY,
                    token_address VARCHAR(42) NOT NULL,
                    dex VARCHAR(50) NOT NULL,
                    pool_address VARCHAR(42),
                    liquidity_usd DECIMAL(20,2),
                    volume_24h DECIMAL(20,2),
                    fees_24h DECIMAL(20,2),
                    fragmentation_score DECIMAL(5,4),
                    depth_1pct DECIMAL(20,2),
                    depth_5pct DECIMAL(20,2),
                    timestamp BIGINT NOT NULL
                );
            ''')

    async def collect_historical_data(self, lookback_days: int = 90) -> int:
        end_time = int(time.time())
        start_time = end_time - (lookback_days * 24 * 3600)
        
        tokens = await self.get_top_tokens(limit=5000)
        collected_count = 0
        
        for token in tokens:
            try:
                historical_data = await self.fetch_token_historical_data(
                    token['address'], start_time, end_time
                )
                
                if historical_data:
                    await self.store_historical_data(historical_data)
                    collected_count += len(historical_data)
                    
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error collecting data for {token['address']}: {e}")
                continue
        
        self.logger.info(f"Collected {collected_count} historical data points")
        return collected_count

    async def get_top_tokens(self, limit: int = 5000) -> List[Dict]:
        headers = {'X-CG-Pro-API-Key': self.api_keys['coingecko']}
        
        all_tokens = []
        page = 1
        per_page = 250
        
        while len(all_tokens) < limit:
            url = f"{self.apis['coingecko']}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': per_page,
                'page': page,
                'sparkline': 'false'
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if not data:
                        break
                    
                    tokens = [
                        {
                            'address': token.get('contract_address', ''),
                            'symbol': token['symbol'],
                            'name': token['name'],
                            'market_cap': token.get('market_cap', 0)
                        }
                        for token in data
                        if token.get('contract_address')
                    ]
                    
                    all_tokens.extend(tokens)
                    page += 1
                    
                    await asyncio.sleep(0.2)
                else:
                    break
        
        return all_tokens[:limit]

    async def fetch_token_historical_data(self, token_address: str, 
                                        start_time: int, end_time: int) -> List[HistoricalTokenData]:
        try:
            price_data = await self.fetch_price_history(token_address, start_time, end_time)
            volume_data = await self.fetch_volume_history(token_address, start_time, end_time)
            tvl_data = await self.fetch_tvl_history(token_address, start_time, end_time)
            
            combined_data = []
            
            for i, price_point in enumerate(price_data):
                timestamp = price_point['timestamp']
                
                volume_point = next((v for v in volume_data if abs(v['timestamp'] - timestamp) < 3600), {})
                tvl_point = next((t for t in tvl_data if abs(t['timestamp'] - timestamp) < 3600), {})
                
                price_24h_ago = self.get_price_at_time(price_data, timestamp - 86400)
                outcome = self.calculate_outcome(price_point['price'], price_24h_ago)
                
                momentum_score = self.calculate_momentum_score(
                    price_data[max(0, i-24):i+1],
                    volume_point.get('volume', 0)
                )
                
                data_point = HistoricalTokenData(
                    token_address=token_address,
                    timestamp=timestamp,
                    price_usd=price_point['price'],
                    volume_24h=volume_point.get('volume', 0),
                    market_cap=price_point.get('market_cap', 0),
                    tvl=tvl_point.get('tvl', 0),
                    holders_count=0,
                    price_change_1h=self.calculate_price_change(price_data, timestamp, 3600),
                    price_change_24h=self.calculate_price_change(price_data, timestamp, 86400),
                    price_change_7d=self.calculate_price_change(price_data, timestamp, 604800),
                    momentum_score=momentum_score,
                    outcome_24h=outcome
                )
                
                combined_data.append(data_point)
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {token_address}: {e}")
            return []

    async def fetch_price_history(self, token_address: str, start_time: int, end_time: int) -> List[Dict]:
        url = f"{self.apis['coingecko']}/coins/{token_address}/market_chart/range"
        headers = {'X-CG-Pro-API-Key': self.api_keys['coingecko']}
        params = {
            'vs_currency': 'usd',
            'from': start_time,
            'to': end_time
        }
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                prices = data.get('prices', [])
                market_caps = data.get('market_caps', [])
                
                return [
                    {
                        'timestamp': int(price[0] / 1000),
                        'price': price[1],
                        'market_cap': next((mc[1] for mc in market_caps if mc[0] == price[0]), 0)
                    }
                    for price in prices
                ]
            return []

    async def fetch_volume_history(self, token_address: str, start_time: int, end_time: int) -> List[Dict]:
        url = f"{self.apis['graph']}/uniswap/uniswap-v3"
        
        query = '''
        {
          tokenDayDatas(
            where: {
              token: "%s",
              date_gte: %d,
              date_lte: %d
            },
            orderBy: date,
            orderDirection: asc
          ) {
            date
            volumeUSD
            totalValueLockedUSD
          }
        }
        ''' % (token_address.lower(), start_time, end_time)
        
        async with self.session.post(url, json={'query': query}) as response:
            if response.status == 200:
                data = await response.json()
                day_data = data.get('data', {}).get('tokenDayDatas', [])
                
                return [
                    {
                        'timestamp': int(item['date']),
                        'volume': float(item['volumeUSD']),
                        'tvl': float(item['totalValueLockedUSD'])
                    }
                    for item in day_data
                ]
            return []

    async def fetch_tvl_history(self, token_address: str, start_time: int, end_time: int) -> List[Dict]:
        url = f"{self.apis['defillama']}/protocol/{token_address}"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                tvl_data = data.get('chainTvls', {}).get('Ethereum', {}).get('tvl', [])
                
                return [
                    {
                        'timestamp': int(point['date']),
                        'tvl': point['totalLiquidityUSD']
                    }
                    for point in tvl_data
                    if start_time <= point['date'] <= end_time
                ]
            return []

    def get_price_at_time(self, price_data: List[Dict], timestamp: int) -> float:
        closest_price = min(
            price_data,
            key=lambda x: abs(x['timestamp'] - timestamp),
            default={'price': 0}
        )
        return closest_price['price']

    def calculate_price_change(self, price_data: List[Dict], timestamp: int, period: int) -> float:
        current_price = self.get_price_at_time(price_data, timestamp)
        past_price = self.get_price_at_time(price_data, timestamp - period)
        
        if past_price > 0:
            return ((current_price - past_price) / past_price) * 100
        return 0.0

    def calculate_momentum_score(self, price_history: List[Dict], volume: float) -> float:
        if len(price_history) < 2:
            return 0.0
        
        prices = [p['price'] for p in price_history]
        returns = np.diff(prices) / prices[:-1]
        
        momentum = np.mean(returns[-12:]) if len(returns) >= 12 else np.mean(returns)
        volatility = np.std(returns)
        volume_factor = min(volume / 100000, 1.0)
        
        score = momentum * volume_factor / (volatility + 0.001)
        return float(np.clip(score, -1, 1))

    def calculate_outcome(self, current_price: float, future_price: float) -> int:
        if future_price == 0:
            return 0
        
        change = ((future_price - current_price) / current_price) * 100
        
        if change >= 10:
            return 2
        elif change >= 5:
            return 1
        elif change <= -10:
            return -2
        elif change <= -5:
            return -1
        else:
            return 0

    async def store_historical_data(self, data_points: List[HistoricalTokenData]):
        async with self.pool.acquire() as conn:
            await conn.executemany('''
                INSERT INTO historical_tokens (
                    token_address, timestamp, price_usd, volume_24h, market_cap,
                    tvl, holders_count, price_change_1h, price_change_24h,
                    price_change_7d, momentum_score, outcome_24h
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (token_address, timestamp) DO NOTHING
            ''', [
                (
                    point.token_address,
                    point.timestamp,
                    point.price_usd,
                    point.volume_24h,
                    point.market_cap,
                    point.tvl,
                    point.holders_count,
                    point.price_change_1h,
                    point.price_change_24h,
                    point.price_change_7d,
                    point.momentum_score,
                    point.outcome_24h
                )
                for point in data_points
            ])

    async def get_training_data(self, min_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT 
                    price_change_1h, price_change_24h, price_change_7d,
                    volume_24h, market_cap, tvl, momentum_score,
                    outcome_24h
                FROM historical_tokens 
                WHERE outcome_24h IS NOT NULL
                AND price_usd > 0
                ORDER BY RANDOM()
                LIMIT $1
            ''', min_samples * 2)
        
        if len(rows) < min_samples:
            raise ValueError(f"Insufficient training data: {len(rows)} < {min_samples}")
        
        features = np.array([
            [
                float(row['price_change_1h']),
                float(row['price_change_24h']),
                float(row['price_change_7d']),
                float(row['volume_24h']),
                float(row['market_cap']),
                float(row['tvl']),
                float(row['momentum_score'])
            ]
            for row in rows
        ])
        
        outcomes = np.array([row['outcome_24h'] for row in rows])
        
        valid_mask = ~np.isnan(features).any(axis=1) & ~np.isnan(outcomes)
        features = features[valid_mask]
        outcomes = outcomes[valid_mask]
        
        self.logger.info(f"Training data: {len(features)} samples")
        return features, outcomes

    async def close(self):
        if self.session:
            await self.session.close()
        if self.pool:
            await self.pool.close()