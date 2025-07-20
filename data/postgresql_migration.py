import asyncio
import asyncpg
import aioredis
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time
import logging
import json
import os
from contextlib import asynccontextmanager
import psutil

@dataclass
class TokenData:
    address: str
    chain: str
    symbol: str
    name: str
    price: float
    volume_24h: float
    liquidity_usd: float
    momentum_score: float
    velocity: float
    volatility: float
    market_cap: float
    holders_count: int
    created_at: datetime = None

@dataclass
class TradeRecord:
    trade_id: str
    token_address: str
    chain: str
    side: str
    amount_usd: float
    amount_tokens: float
    entry_price: float
    exit_price: Optional[float]
    profit_loss: Optional[float]
    roi: Optional[float]
    confidence_score: float
    momentum_score: float
    execution_time: float
    slippage: float
    gas_cost: float
    tx_hash: str
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime]
    exit_reason: Optional[str]
    created_at: datetime = None

class PostgreSQLManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=10,
                max_size=50,
                command_timeout=30,
                server_settings={
                    'jit': 'off',
                    'application_name': 'renaissance_trading'
                }
            )
            
            await self.create_extensions()
            await self.create_tables()
            await self.create_indexes()
            await self.create_partitions()
            
            self.logger.info("PostgreSQL database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def create_extensions(self):
        async with self.pool.acquire() as conn:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "btree_gin"')
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "pg_stat_statements"')