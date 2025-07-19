import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import aiosqlite
import asyncio
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class PerformanceMetric:
    timestamp: float
    metric_name: str
    metric_value: float
    metadata: Dict[str, Any]

@dataclass
class TradePerformance:
    trade_id: str
    token_address: str
    chain: str
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    position_size: float
    roi: float
    profit_usd: float
    strategy: str
    signals: Dict[str, float]
    execution_time: float

class PerformanceDatabase:
    def __init__(self, db_path: str = './cache/performance.db'):
        self.db_path = db_path
        self.connection = None
        
    async def initialize(self):
        self.connection = await aiosqlite.connect(self.db_path)
        await self.connection.execute('PRAGMA journal_mode=WAL')
        await self.connection.execute('PRAGMA synchronous=NORMAL')
        
        await self.create_tables()

    async def create_tables(self):
        await self.connection.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                metric_name TEXT,
                metric_value REAL,
                metadata TEXT
            )
        ''')
        
        await self.connection.execute('''
            CREATE TABLE IF NOT EXISTS trade_performance (
                trade_id TEXT PRIMARY KEY,
                token_address TEXT,
                chain TEXT,
                entry_time REAL,
                exit_time REAL,
                entry_price REAL,
                exit_price REAL,
                position_size REAL,
                roi REAL,
                profit_usd REAL,
                strategy TEXT,
                signals TEXT,
                execution_time REAL
            )
        ''')
        
        await self.connection.execute('''
            CREATE TABLE IF NOT EXISTS system_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                cpu_usage REAL,
                memory_usage REAL,
                tokens_scanned INTEGER,
                signals_generated INTEGER,
                trades_executed INTEGER,
                active_connections INTEGER
            )
        ''')
        
        await self.connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)
        ''')
        
        await self.connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_trade_time ON trade_performance(entry_time)
        ''')
        
        await self.connection.commit()

    async def record_metric(self, metric: PerformanceMetric):
        await self.connection.execute('''
            INSERT INTO performance_metrics (timestamp, metric_name, metric_value, metadata)
            VALUES (?, ?, ?, ?)
        ''', (metric.timestamp, metric.metric_name, metric.metric_value, json.dumps(metric.metadata)))
        
        await self.connection.commit()

    async def record_trade(self, trade: TradePerformance):
        await self.connection.execute('''
            INSERT OR REPLACE INTO trade_performance (
                trade_id, token_address, chain, entry_time, exit_time,
                entry_price, exit_price, position_size, roi, profit_usd,
                strategy, signals, execution_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.trade_id, trade.token_address, trade.chain,
            trade.entry_time, trade.exit_time, trade.entry_price,
            trade.exit_price, trade.position_size, trade.roi,
            trade.profit_usd, trade.strategy, json.dumps(trade.signals),
            trade.execution_time
        ))
        
        await self.connection.commit()

    async def record_system_performance(self, cpu_usage: float, memory_usage: float,
                                       tokens_scanned: int, signals_generated: int,
                                       trades_executed: int, active_connections: int):
        await self.connection.execute('''
            INSERT INTO system_performance (
                timestamp, cpu_usage, memory_usage, tokens_scanned,
                signals_generated, trades_executed, active_connections
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            time.time(), cpu_usage, memory_usage, tokens_scanned,
            signals_generated, trades_executed, active_connections
        ))
        
        await self.connection.commit()

    async def get_recent_metrics(self, metric_name: str, hours: int = 24) -> List[PerformanceMetric]:
        since = time.time() - (hours * 3600)
        
        cursor = await self.connection.execute('''
            SELECT timestamp, metric_name, metric_value, metadata
            FROM performance_metrics
            WHERE metric_name = ? AND timestamp > ?
            ORDER BY timestamp ASC
        ''', (metric_name, since))
        
        rows = await cursor.fetchall()
        
        return [
            PerformanceMetric(
                timestamp=row[0],
                metric_name=row[1],
                metric_value=row[2],
                metadata=json.loads(row[3]) if row[3] else {}
            ) for row in rows
        ]

    async def get_trading_performance(self, hours: int = 24) -> Dict[str, float]:
        since = time.time() - (hours * 3600)
        
        cursor = await self.connection.execute('''
            SELECT roi, profit_usd, execution_time
            FROM trade_performance
            WHERE entry_time > ?
        ''', (since,))
        
        rows = await cursor.fetchall()
        
        if not rows:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'avg_roi': 0.0,
                'avg_execution_time': 0.0
            }
        
        rois = [row[0] for row in rows]
        profits = [row[1] for row in rows]
        exec_times = [row[2] for row in rows]
        
        return {
            'total_trades': len(rows),
            'win_rate': sum(1 for roi in rois if roi > 0) / len(rois),
            'total_profit': sum(profits),
            'avg_roi': np.mean(rois),
            'avg_execution_time': np.mean(exec_times),
            'sharpe_ratio': np.mean(rois) / (np.std(rois) + 1e-8)
        }

    async def get_system_performance_summary(self, hours: int = 24) -> Dict[str, float]:
        since = time.time() - (hours * 3600)
        
        cursor = await self.connection.execute('''
            SELECT AVG(cpu_usage), AVG(memory_usage), SUM(tokens_scanned),
                   SUM(signals_generated), SUM(trades_executed), AVG(active_connections)
            FROM system_performance
            WHERE timestamp > ?
        ''', (since,))
        
        row = await cursor.fetchone()
        
        if not row or row[0] is None:
            return {
                'avg_cpu_usage': 0.0,
                'avg_memory_usage': 0.0,
                'total_tokens_scanned': 0,
                'total_signals_generated': 0,
                'total_trades_executed': 0,
                'avg_active_connections': 0.0
            }
        
        return {
            'avg_cpu_usage': row[0] or 0.0,
            'avg_memory_usage': row[1] or 0.0,
            'total_tokens_scanned': int(row[2] or 0),
            'total_signals_generated': int(row[3] or 0),
            'total_trades_executed': int(row[4] or 0),
            'avg_active_connections': row[5] or 0.0
        }

    async def cleanup_old_data(self, days: int = 7):
        cutoff = time.time() - (days * 24 * 3600)
        
        await self.connection.execute('''
            DELETE FROM performance_metrics WHERE timestamp < ?
        ''', (cutoff,))
        
        await self.connection.execute('''
            DELETE FROM trade_performance WHERE entry_time < ?
        ''', (cutoff,))
        
        await self.connection.execute('''
            DELETE FROM system_performance WHERE timestamp < ?
        ''', (cutoff,))
        
        await self.connection.commit()

    async def close(self):
        if self.connection:
            await self.connection.close()

performance_db = PerformanceDatabase()
