import asyncio
import aiosqlite
import sqlite3
import json
import time
import numpy as np
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

@dataclass
class TokenData:
    address: str
    chain: str
    symbol: str
    name: str
    price: float
    volume_24h: float
    liquidity_usd: float
    market_cap: float
    momentum_score: float
    velocity: float
    volatility: float
    confidence: float
    discovered_at: float
    last_updated: float

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
    entry_time: float
    exit_time: Optional[float]
    profit_loss: Optional[float]
    roi: Optional[float]
    confidence_score: float
    momentum_score: float
    hold_time: Optional[float]
    exit_reason: Optional[str]
    tx_hash: str
    gas_used: Optional[int]
    created_at: float

@dataclass
class PerformanceSnapshot:
    timestamp: float
    portfolio_value: float
    total_trades: int
    profitable_trades: int
    total_pnl: float
    win_rate: float
    avg_hold_time: float
    max_drawdown: float
    sharpe_ratio: float
    tokens_scanned: int
    signals_generated: int

class ProductionDatabaseManager:
    def __init__(self, db_path: str = 'data/trading_production.db'):
        self.db_path = db_path
        self.connection_pool = asyncio.Queue(maxsize=10)
        self.lock = asyncio.Lock()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        for _ in range(5):
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA synchronous=NORMAL')
            await conn.execute('PRAGMA cache_size=10000')
            await conn.execute('PRAGMA temp_store=memory')
            await self.connection_pool.put(conn)
        
        async with self.get_connection() as conn:
            await self.create_tables(conn)
            await self.create_indexes(conn)
        
        self.logger.info("Production database initialized")

    async def get_connection(self):
        return await self.connection_pool.get()

    async def return_connection(self, conn):
        await self.connection_pool.put(conn)

    async def create_tables(self, conn: aiosqlite.Connection):
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS tokens (
                address TEXT,
                chain TEXT,
                symbol TEXT,
                name TEXT,
                price REAL,
                volume_24h REAL,
                liquidity_usd REAL,
                market_cap REAL,
                momentum_score REAL,
                velocity REAL,
                volatility REAL,
                confidence REAL,
                discovered_at REAL,
                last_updated REAL,
                PRIMARY KEY (address, chain)
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                token_address TEXT,
                chain TEXT,
                side TEXT,
                amount_usd REAL,
                amount_tokens REAL,
                entry_price REAL,
                exit_price REAL,
                entry_time REAL,
                exit_time REAL,
                profit_loss REAL,
                roi REAL,
                confidence_score REAL,
                momentum_score REAL,
                hold_time REAL,
                exit_reason TEXT,
                tx_hash TEXT,
                gas_used INTEGER,
                created_at REAL
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                portfolio_value REAL,
                total_trades INTEGER,
                profitable_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                avg_hold_time REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                tokens_scanned INTEGER,
                signals_generated INTEGER
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS token_features (
                token_address TEXT,
                chain TEXT,
                timestamp REAL,
                feature_vector TEXT,
                prediction_score REAL,
                confidence REAL,
                PRIMARY KEY (token_address, chain, timestamp)
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                cpu_usage REAL,
                memory_usage REAL,
                active_workers INTEGER,
                queue_size INTEGER,
                api_calls_per_minute REAL,
                error_rate REAL
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                model_version TEXT,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                auc_score REAL,
                prediction_count INTEGER,
                correct_predictions INTEGER
            )
        ''')
        
        await conn.commit()

    async def create_indexes(self, conn: aiosqlite.Connection):
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_tokens_momentum ON tokens (momentum_score DESC, last_updated DESC)',
            'CREATE INDEX IF NOT EXISTS idx_tokens_chain ON tokens (chain, momentum_score DESC)',
            'CREATE INDEX IF NOT EXISTS idx_tokens_volume ON tokens (volume_24h DESC, last_updated DESC)',
            'CREATE INDEX IF NOT EXISTS idx_trades_time ON trades (entry_time DESC)',
            'CREATE INDEX IF NOT EXISTS idx_trades_token ON trades (token_address, chain)',
            'CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades (profit_loss DESC, entry_time DESC)',
            'CREATE INDEX IF NOT EXISTS idx_performance_time ON performance_snapshots (timestamp DESC)',
            'CREATE INDEX IF NOT EXISTS idx_features_time ON token_features (timestamp DESC)',
            'CREATE INDEX IF NOT EXISTS idx_features_score ON token_features (prediction_score DESC, timestamp DESC)',
            'CREATE INDEX IF NOT EXISTS idx_system_time ON system_metrics (timestamp DESC)',
            'CREATE INDEX IF NOT EXISTS idx_model_time ON model_performance (timestamp DESC)'
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
        
        await conn.commit()

    async def cache_token(self, token: TokenData):
        async with self.lock:
            conn = await self.get_connection()
            try:
                await conn.execute('''
                    INSERT OR REPLACE INTO tokens 
                    (address, chain, symbol, name, price, volume_24h, liquidity_usd, market_cap,
                     momentum_score, velocity, volatility, confidence, discovered_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    token.address, token.chain, token.symbol, token.name,
                    token.price, token.volume_24h, token.liquidity_usd, token.market_cap,
                    token.momentum_score, token.velocity, token.volatility, token.confidence,
                    token.discovered_at, token.last_updated
                ))
                await conn.commit()
            finally:
                await self.return_connection(conn)

    async def get_token(self, address: str, chain: str) -> Optional[TokenData]:
        conn = await self.get_connection()
        try:
            async with conn.execute('''
                SELECT * FROM tokens WHERE address = ? AND chain = ?
            ''', (address, chain)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    return TokenData(
                        address=row[0], chain=row[1], symbol=row[2], name=row[3],
                        price=row[4], volume_24h=row[5], liquidity_usd=row[6],
                        market_cap=row[7], momentum_score=row[8], velocity=row[9],
                        volatility=row[10], confidence=row[11], discovered_at=row[12],
                        last_updated=row[13]
                    )
                return None
        finally:
            await self.return_connection(conn)

    async def get_top_momentum_tokens(self, chain: Optional[str] = None, limit: int = 100) -> List[TokenData]:
        conn = await self.get_connection()
        try:
            if chain:
                query = '''
                    SELECT * FROM tokens 
                    WHERE chain = ? AND last_updated > ?
                    ORDER BY momentum_score DESC, volume_24h DESC 
                    LIMIT ?
                '''
                params = (chain, time.time() - 3600, limit)
            else:
                query = '''
                    SELECT * FROM tokens 
                    WHERE last_updated > ?
                    ORDER BY momentum_score DESC, volume_24h DESC 
                    LIMIT ?
                '''
                params = (time.time() - 3600, limit)
            
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                
                tokens = []
                for row in rows:
                    tokens.append(TokenData(
                        address=row[0], chain=row[1], symbol=row[2], name=row[3],
                        price=row[4], volume_24h=row[5], liquidity_usd=row[6],
                        market_cap=row[7], momentum_score=row[8], velocity=row[9],
                        volatility=row[10], confidence=row[11], discovered_at=row[12],
                        last_updated=row[13]
                    ))
                
                return tokens
        finally:
            await self.return_connection(conn)

    async def record_trade(self, trade: TradeRecord):
        async with self.lock:
            conn = await self.get_connection()
            try:
                await conn.execute('''
                    INSERT OR REPLACE INTO trades 
                    (trade_id, token_address, chain, side, amount_usd, amount_tokens,
                     entry_price, exit_price, entry_time, exit_time, profit_loss, roi,
                     confidence_score, momentum_score, hold_time, exit_reason, tx_hash,
                     gas_used, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.trade_id, trade.token_address, trade.chain, trade.side,
                    trade.amount_usd, trade.amount_tokens, trade.entry_price,
                    trade.exit_price, trade.entry_time, trade.exit_time,
                    trade.profit_loss, trade.roi, trade.confidence_score,
                    trade.momentum_score, trade.hold_time, trade.exit_reason,
                    trade.tx_hash, trade.gas_used, trade.created_at
                ))
                await conn.commit()
            finally:
                await self.return_connection(conn)

    async def update_trade_exit(self, trade_id: str, exit_price: float, profit_loss: float, 
                               roi: float, exit_reason: str):
        async with self.lock:
            conn = await self.get_connection()
            try:
                exit_time = time.time()
                
                async with conn.execute('SELECT entry_time FROM trades WHERE trade_id = ?', (trade_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        hold_time = exit_time - row[0]
                    else:
                        hold_time = 0
                
                await conn.execute('''
                    UPDATE trades 
                    SET exit_price = ?, exit_time = ?, profit_loss = ?, roi = ?, 
                        hold_time = ?, exit_reason = ?
                    WHERE trade_id = ?
                ''', (exit_price, exit_time, profit_loss, roi, hold_time, exit_reason, trade_id))
                
                await conn.commit()
            finally:
                await self.return_connection(conn)

    async def get_recent_trades(self, hours: int = 24, limit: int = 100) -> List[TradeRecord]:
        conn = await self.get_connection()
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            async with conn.execute('''
                SELECT * FROM trades 
                WHERE entry_time > ?
                ORDER BY entry_time DESC 
                LIMIT ?
            ''', (cutoff_time, limit)) as cursor:
                rows = await cursor.fetchall()
                
                trades = []
                for row in rows:
                    trades.append(TradeRecord(
                        trade_id=row[0], token_address=row[1], chain=row[2],
                        side=row[3], amount_usd=row[4], amount_tokens=row[5],
                        entry_price=row[6], exit_price=row[7], entry_time=row[8],
                        exit_time=row[9], profit_loss=row[10], roi=row[11],
                        confidence_score=row[12], momentum_score=row[13],
                        hold_time=row[14], exit_reason=row[15], tx_hash=row[16],
                        gas_used=row[17], created_at=row[18]
                    ))
                
                return trades
        finally:
            await self.return_connection(conn)

    async def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        conn = await self.get_connection()
        try:
            cutoff_time = time.time() - (days * 86400)
            
            async with conn.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as profitable_trades,
                    SUM(profit_loss) as total_pnl,
                    AVG(profit_loss) as avg_pnl,
                    AVG(hold_time) as avg_hold_time,
                    AVG(roi) as avg_roi,
                    MIN(profit_loss) as max_loss,
                    MAX(profit_loss) as max_profit
                FROM trades 
                WHERE entry_time > ? AND profit_loss IS NOT NULL
            ''', (cutoff_time,)) as cursor:
                row = await cursor.fetchone()
                
                if row and row[0] > 0:
                    total_trades = row[0]
                    profitable_trades = row[1]
                    total_pnl = row[2] or 0
                    avg_pnl = row[3] or 0
                    avg_hold_time = row[4] or 0
                    avg_roi = row[5] or 0
                    max_loss = row[6] or 0
                    max_profit = row[7] or 0
                    
                    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
                    
                    daily_returns = []
                    async with conn.execute('''
                        SELECT DATE(entry_time, 'unixepoch') as trade_date, SUM(profit_loss) as daily_pnl
                        FROM trades 
                        WHERE entry_time > ? AND profit_loss IS NOT NULL
                        GROUP BY DATE(entry_time, 'unixepoch')
                        ORDER BY trade_date
                    ''', (cutoff_time,)) as cursor2:
                        rows = await cursor2.fetchall()
                        daily_returns = [row[1] for row in rows if row[1] is not None]
                    
                    sharpe_ratio = 0
                    max_drawdown = 0
                    
                    if len(daily_returns) > 1:
                        daily_returns_np = np.array(daily_returns)
                        if np.std(daily_returns_np) > 0:
                            sharpe_ratio = np.mean(daily_returns_np) / np.std(daily_returns_np) * np.sqrt(365)
                        
                        cumulative_returns = np.cumsum(daily_returns_np)
                        running_max = np.maximum.accumulate(cumulative_returns)
                        drawdowns = (cumulative_returns - running_max) / (running_max + 1e-10)
                        max_drawdown = np.min(drawdowns)
                    
                    return {
                        'period_days': days,
                        'total_trades': total_trades,
                        'profitable_trades': profitable_trades,
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'avg_pnl_per_trade': avg_pnl,
                        'avg_hold_time_hours': avg_hold_time / 3600 if avg_hold_time else 0,
                        'avg_roi': avg_roi,
                        'max_profit': max_profit,
                        'max_loss': max_loss,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': abs(max_drawdown),
                        'profit_factor': abs(max_profit / max_loss) if max_loss < 0 else 0
                    }
                else:
                    return {
                        'period_days': days,
                        'total_trades': 0,
                        'profitable_trades': 0,
                        'win_rate': 0,
                        'total_pnl': 0,
                        'avg_pnl_per_trade': 0,
                        'avg_hold_time_hours': 0,
                        'avg_roi': 0,
                        'max_profit': 0,
                        'max_loss': 0,
                        'sharpe_ratio': 0,
                        'max_drawdown': 0,
                        'profit_factor': 0
                    }
        finally:
            await self.return_connection(conn)

    async def record_performance_snapshot(self, snapshot: PerformanceSnapshot):
        async with self.lock:
            conn = await self.get_connection()
            try:
                await conn.execute('''
                    INSERT INTO performance_snapshots 
                    (timestamp, portfolio_value, total_trades, profitable_trades, total_pnl,
                     win_rate, avg_hold_time, max_drawdown, sharpe_ratio, tokens_scanned, signals_generated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp, snapshot.portfolio_value, snapshot.total_trades,
                    snapshot.profitable_trades, snapshot.total_pnl, snapshot.win_rate,
                    snapshot.avg_hold_time, snapshot.max_drawdown, snapshot.sharpe_ratio,
                    snapshot.tokens_scanned, snapshot.signals_generated
                ))
                await conn.commit()
            finally:
                await self.return_connection(conn)

    async def record_system_performance(self, cpu_usage: float, memory_usage: float,
                                      active_workers: int, queue_size: int,
                                      api_calls_per_minute: float, error_rate: float):
        async with self.lock:
            conn = await self.get_connection()
            try:
                await conn.execute('''
                    INSERT INTO system_metrics 
                    (timestamp, cpu_usage, memory_usage, active_workers, queue_size,
                     api_calls_per_minute, error_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    time.time(), cpu_usage, memory_usage, active_workers,
                    queue_size, api_calls_per_minute, error_rate
                ))
                await conn.commit()
            finally:
                await self.return_connection(conn)

    async def save_token_features(self, token_address: str, chain: str, 
                                 feature_vector: List[float], prediction_score: float, 
                                 confidence: float):
        async with self.lock:
            conn = await self.get_connection()
            try:
                await conn.execute('''
                    INSERT OR REPLACE INTO token_features 
                    (token_address, chain, timestamp, feature_vector, prediction_score, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    token_address, chain, time.time(),
                    json.dumps(feature_vector), prediction_score, confidence
                ))
                await conn.commit()
            finally:
                await self.return_connection(conn)

    async def get_token_features(self, token_address: str, chain: str, 
                                hours_back: int = 24) -> List[Dict]:
        conn = await self.get_connection()
        try:
            cutoff_time = time.time() - (hours_back * 3600)
            
            async with conn.execute('''
                SELECT timestamp, feature_vector, prediction_score, confidence 
                FROM token_features 
                WHERE token_address = ? AND chain = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (token_address, chain, cutoff_time)) as cursor:
                rows = await cursor.fetchall()
                
                features = []
                for row in rows:
                    features.append({
                        'timestamp': row[0],
                        'feature_vector': json.loads(row[1]),
                        'prediction_score': row[2],
                        'confidence': row[3]
                    })
                
                return features
        finally:
            await self.return_connection(conn)

    async def cleanup_old_data(self, days_to_keep: int = 30):
        async with self.lock:
            conn = await self.get_connection()
            try:
                cutoff_time = time.time() - (days_to_keep * 86400)
                
                tables_to_clean = [
                    ('tokens', 'last_updated'),
                    ('token_features', 'timestamp'),
                    ('system_metrics', 'timestamp'),
                    ('model_performance', 'timestamp')
                ]
                
                for table, time_column in tables_to_clean:
                    await conn.execute(f'DELETE FROM {table} WHERE {time_column} < ?', (cutoff_time,))
                
                await conn.execute('VACUUM')
                await conn.commit()
                
                self.logger.info(f"Cleaned up data older than {days_to_keep} days")
                
            finally:
                await self.return_connection(conn)

    async def get_database_stats(self) -> Dict[str, Any]:
        conn = await self.get_connection()
        try:
            stats = {}
            
            tables = ['tokens', 'trades', 'performance_snapshots', 'token_features', 'system_metrics']
            
            for table in tables:
                async with conn.execute(f'SELECT COUNT(*) FROM {table}') as cursor:
                    row = await cursor.fetchone()
                    stats[f'{table}_count'] = row[0] if row else 0
            
            async with conn.execute('SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()') as cursor:
                row = await cursor.fetchone()
                stats['db_size_bytes'] = row[0] if row else 0
            
            stats['db_size_mb'] = stats['db_size_bytes'] / (1024 * 1024)
            
            return stats
        finally:
            await self.return_connection(conn)

    async def close(self):
        while not self.connection_pool.empty():
            conn = await self.connection_pool.get()
            await conn.close()

db_manager = ProductionDatabaseManager()