import sqlite3
import asyncio
import aiosqlite
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
from contextlib import asynccontextmanager

@dataclass
class TokenData:
    address: str
    chain: str
    symbol: str = ""
    name: str = ""
    price: float = 0.0
    volume_24h: float = 0.0
    liquidity_usd: float = 0.0
    momentum_score: float = 0.0
    velocity: float = 0.0
    volatility: float = 0.0

@dataclass
class TradeRecord:
    trade_id: str
    token_address: str
    chain: str
    side: str
    amount_usd: float
    amount_tokens: float
    entry_price: float
    exit_price: Optional[float] = None
    profit_loss_usd: float = 0.0
    roi_percent: float = 0.0
    hold_time_seconds: int = 0
    gas_cost_usd: float = 0.0
    slippage_percent: float = 0.0
    confidence_score: float = 0.0
    momentum_score: float = 0.0
    exit_reason: Optional[str] = None
    tx_hash: Optional[str] = None

class DatabaseManager:
    def __init__(self, db_path: str = 'cache/renaissance_trading.db'):
        self.db_path = db_path
        self.connection_pool = []
        self.pool_size = 10
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize database with schema"""
        # Create database directory
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Load and execute schema
        with open('data/db_schema.sql', 'r') as f:
            schema = f.read()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(schema)
            await db.commit()
        
        # Initialize connection pool
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute('PRAGMA journal_mode=WAL')  # Enable WAL mode for performance
            await conn.execute('PRAGMA synchronous=NORMAL')
            await conn.execute('PRAGMA cache_size=10000')
            await conn.execute('PRAGMA temp_store=MEMORY')
            self.connection_pool.append(conn)
        
        self.logger.info(f"Database initialized with {self.pool_size} connections")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        with self.lock:
            if self.connection_pool:
                conn = self.connection_pool.pop()
            else:
                conn = await aiosqlite.connect(self.db_path)
        
        try:
            yield conn
        finally:
            with self.lock:
                if len(self.connection_pool) < self.pool_size:
                    self.connection_pool.append(conn)
                else:
                    await conn.close()
    
    async def cache_token(self, token_data: TokenData) -> bool:
        """Cache token data with upsert logic"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO token_cache 
                    (address, chain, symbol, name, price, volume_24h, liquidity_usd, 
                     momentum_score, velocity, volatility, last_updated, scan_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP,
                            COALESCE((SELECT scan_count FROM token_cache WHERE address = ?), 0) + 1)
                """, (
                    token_data.address, token_data.chain, token_data.symbol,
                    token_data.name, token_data.price, token_data.volume_24h,
                    token_data.liquidity_usd, token_data.momentum_score,
                    token_data.velocity, token_data.volatility, token_data.address
                ))
                await db.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error caching token {token_data.address}: {e}")
            return False
    
    async def get_token_data(self, address: str, chain: str) -> Optional[TokenData]:
        """Retrieve cached token data"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT address, chain, symbol, name, price, volume_24h, 
                           liquidity_usd, momentum_score, velocity, volatility
                    FROM token_cache 
                    WHERE address = ? AND chain = ?
                """, (address, chain))
                
                row = await cursor.fetchone()
                if row:
                    return TokenData(*row)
                return None
        except Exception as e:
            self.logger.error(f"Error getting token data for {address}: {e}")
            return None
    
    async def record_trade(self, trade: TradeRecord) -> bool:
        """Record a trade execution"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO trades 
                    (trade_id, token_address, chain, side, amount_usd, amount_tokens,
                     entry_price, exit_price, profit_loss_usd, roi_percent,
                     hold_time_seconds, gas_cost_usd, slippage_percent,
                     confidence_score, momentum_score, exit_reason, tx_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.trade_id, trade.token_address, trade.chain, trade.side,
                    trade.amount_usd, trade.amount_tokens, trade.entry_price,
                    trade.exit_price, trade.profit_loss_usd, trade.roi_percent,
                    trade.hold_time_seconds, trade.gas_cost_usd, trade.slippage_percent,
                    trade.confidence_score, trade.momentum_score, trade.exit_reason,
                    trade.tx_hash
                ))
                await db.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error recording trade {trade.trade_id}: {e}")
            return False
    
    async def update_trade_exit(self, trade_id: str, exit_price: float, 
                               profit_loss: float, roi: float, exit_reason: str) -> bool:
        """Update trade with exit information"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    UPDATE trades 
                    SET exit_price = ?, profit_loss_usd = ?, roi_percent = ?,
                        exit_reason = ?, closed_at = CURRENT_TIMESTAMP,
                        hold_time_seconds = CAST(
                            (julianday(CURRENT_TIMESTAMP) - julianday(executed_at)) * 86400 AS INTEGER
                        )
                    WHERE trade_id = ?
                """, (exit_price, profit_loss, roi, exit_reason, trade_id))
                await db.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error updating trade exit {trade_id}: {e}")
            return False
    
    async def record_system_performance(self, metrics: Dict[str, float]) -> bool:
        """Record system performance metrics"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO system_performance 
                    (tokens_scanned_per_hour, signals_generated_per_hour, 
                     trades_executed_per_hour, cpu_usage_percent, memory_usage_percent,
                     network_latency_ms, error_rate_percent, uptime_hours,
                     portfolio_value_usd, total_pnl_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.get('tokens_per_hour', 0),
                    metrics.get('signals_per_hour', 0),
                    metrics.get('trades_per_hour', 0),
                    metrics.get('cpu_usage', 0),
                    metrics.get('memory_usage', 0),
                    metrics.get('network_latency', 0),
                    metrics.get('error_rate', 0),
                    metrics.get('uptime_hours', 0),
                    metrics.get('portfolio_value', 0),
                    metrics.get('total_pnl', 0)
                ))
                await db.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error recording system performance: {e}")
            return False
    
    async def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN profit_loss_usd > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(profit_loss_usd) as total_pnl,
                        AVG(roi_percent) as avg_roi,
                        MAX(profit_loss_usd) as max_win,
                        MIN(profit_loss_usd) as max_loss,
                        AVG(hold_time_seconds) as avg_hold_time,
                        AVG(confidence_score) as avg_confidence
                    FROM trades 
                    WHERE closed_at IS NOT NULL 
                    AND executed_at > datetime('now', '-{} days')
                """.format(days))
                
                row = await cursor.fetchone()
                if row:
                    return {
                        'total_trades': row[0] or 0,
                        'winning_trades': row[1] or 0,
                        'win_rate': (row[1] or 0) / max(row[0] or 1, 1),
                        'total_pnl': row[2] or 0,
                        'avg_roi': row[3] or 0,
                        'max_win': row[4] or 0,
                        'max_loss': row[5] or 0,
                        'avg_hold_time': row[6] or 0,
                        'avg_confidence': row[7] or 0
                    }
                return {}
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    async def get_top_tokens(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top performing tokens"""
        try:
            async with self.get_connection() as db:
                cursor = await db.execute("""
                    SELECT 
                        tc.symbol,
                        tc.address,
                        tc.chain,
                        COUNT(t.id) as trade_count,
                        SUM(t.profit_loss_usd) as total_pnl,
                        AVG(t.roi_percent) as avg_roi,
                        MAX(t.profit_loss_usd) as best_trade,
                        AVG(t.confidence_score) as avg_confidence
                    FROM trades t
                    JOIN token_cache tc ON t.token_address = tc.address
                    WHERE t.closed_at IS NOT NULL
                    GROUP BY tc.address
                    HAVING trade_count >= 1
                    ORDER BY total_pnl DESC
                    LIMIT ?
                """, (limit,))
                
                rows = await cursor.fetchall()
                return [
                    {
                        'symbol': row[0],
                        'address': row[1],
                        'chain': row[2],
                        'trade_count': row[3],
                        'total_pnl': row[4],
                        'avg_roi': row[5],
                        'best_trade': row[6],
                        'avg_confidence': row[7]
                    }
                    for row in rows
                ]
        except Exception as e:
            self.logger.error(f"Error getting top tokens: {e}")
            return []
    
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old data to maintain performance"""
        try:
            async with self.get_connection() as db:
                # Clean old price history
                await db.execute("""
                    DELETE FROM price_history 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days))
                
                # Clean old system performance records
                await db.execute("""
                    DELETE FROM system_performance 
                    WHERE recorded_at < datetime('now', '-{} days')
                """.format(days))
                
                await db.commit()
                self.logger.info(f"Cleaned up data older than {days} days")
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    async def close(self):
        """Close all database connections"""
        for conn in self.connection_pool:
            await conn.close()
        self.connection_pool.clear()

# Global database manager instance
db_manager = DatabaseManager()
