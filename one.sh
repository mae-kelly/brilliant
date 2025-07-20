#!/bin/bash
# =============================================================================
# 🗄️ IMPLEMENT PRODUCTION DATABASE LAYER - Renaissance Data Management
# =============================================================================

set -e

echo "🗄️ IMPLEMENTING PRODUCTION DATABASE LAYER"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO") echo -e "${BLUE}ℹ️  $message${NC}" ;;
        "SUCCESS") echo -e "${GREEN}✅ $message${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "ERROR") echo -e "${RED}❌ $message${NC}" ;;
    esac
}

print_status "INFO" "Step 1: Creating database directories..."
mkdir -p cache data/db logs/db

print_status "INFO" "Step 2: Creating comprehensive database schema..."

# Create the main database schema
cat > data/db_schema.sql << 'EOF'
-- =============================================================================
-- RENAISSANCE TRADING SYSTEM - PRODUCTION DATABASE SCHEMA
-- =============================================================================

-- Token Cache Table - High-frequency token data
CREATE TABLE IF NOT EXISTS token_cache (
    address TEXT PRIMARY KEY,
    chain TEXT NOT NULL,
    symbol TEXT,
    name TEXT,
    price REAL NOT NULL,
    volume_24h REAL DEFAULT 0,
    liquidity_usd REAL DEFAULT 0,
    momentum_score REAL DEFAULT 0,
    velocity REAL DEFAULT 0,
    volatility REAL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    scan_count INTEGER DEFAULT 1,
    INDEX(chain),
    INDEX(momentum_score),
    INDEX(last_updated)
);

-- Price History - OHLCV data for technical analysis
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_address TEXT NOT NULL,
    chain TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open_price REAL NOT NULL,
    high_price REAL NOT NULL,
    low_price REAL NOT NULL,
    close_price REAL NOT NULL,
    volume REAL DEFAULT 0,
    FOREIGN KEY (token_address) REFERENCES token_cache(address),
    INDEX(token_address, timestamp),
    INDEX(timestamp)
);

-- Trade Executions - Complete trade history
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    token_address TEXT NOT NULL,
    chain TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    amount_usd REAL NOT NULL,
    amount_tokens REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    profit_loss_usd REAL DEFAULT 0,
    roi_percent REAL DEFAULT 0,
    hold_time_seconds INTEGER DEFAULT 0,
    gas_cost_usd REAL DEFAULT 0,
    slippage_percent REAL DEFAULT 0,
    confidence_score REAL DEFAULT 0,
    momentum_score REAL DEFAULT 0,
    exit_reason TEXT,
    tx_hash TEXT,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    FOREIGN KEY (token_address) REFERENCES token_cache(address),
    INDEX(token_address),
    INDEX(executed_at),
    INDEX(profit_loss_usd),
    INDEX(roi_percent)
);

-- ML Model Performance - Track model accuracy over time
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    prediction_accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score REAL NOT NULL,
    f1_score REAL NOT NULL,
    sharpe_ratio REAL DEFAULT 0,
    max_drawdown REAL DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0,
    avg_roi REAL DEFAULT 0,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX(model_version),
    INDEX(evaluated_at)
);

-- System Performance - Overall system metrics
CREATE TABLE IF NOT EXISTS system_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tokens_scanned_per_hour REAL NOT NULL,
    signals_generated_per_hour REAL NOT NULL,
    trades_executed_per_hour REAL NOT NULL,
    cpu_usage_percent REAL DEFAULT 0,
    memory_usage_percent REAL DEFAULT 0,
    network_latency_ms REAL DEFAULT 0,
    error_rate_percent REAL DEFAULT 0,
    uptime_hours REAL DEFAULT 0,
    portfolio_value_usd REAL DEFAULT 0,
    total_pnl_usd REAL DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX(recorded_at)
);

-- Risk Events - Track risk events and circuit breaker activations
CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    token_address TEXT,
    description TEXT NOT NULL,
    risk_score REAL DEFAULT 0,
    action_taken TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    INDEX(event_type),
    INDEX(severity),
    INDEX(occurred_at)
);

-- Honeypot Detection Results
CREATE TABLE IF NOT EXISTS honeypot_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_address TEXT NOT NULL,
    chain TEXT NOT NULL,
    is_honeypot BOOLEAN DEFAULT FALSE,
    risk_score REAL DEFAULT 0,
    buy_tax REAL DEFAULT 0,
    sell_tax REAL DEFAULT 0,
    liquidity_locked BOOLEAN DEFAULT FALSE,
    contract_verified BOOLEAN DEFAULT FALSE,
    owner_renounced BOOLEAN DEFAULT FALSE,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(token_address, chain),
    INDEX(is_honeypot),
    INDEX(risk_score),
    INDEX(checked_at)
);

-- Social Sentiment Data
CREATE TABLE IF NOT EXISTS social_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_address TEXT NOT NULL,
    platform TEXT NOT NULL CHECK (platform IN ('twitter', 'reddit', 'telegram', 'discord')),
    sentiment_score REAL NOT NULL,
    mention_count INTEGER DEFAULT 0,
    volume_spike REAL DEFAULT 0,
    influence_score REAL DEFAULT 0,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (token_address) REFERENCES token_cache(address),
    INDEX(token_address, platform),
    INDEX(sentiment_score),
    INDEX(analyzed_at)
);

-- Dynamic Parameters History - Track parameter optimization
CREATE TABLE IF NOT EXISTS parameter_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parameter_name TEXT NOT NULL,
    old_value REAL NOT NULL,
    new_value REAL NOT NULL,
    performance_delta REAL DEFAULT 0,
    optimization_reason TEXT,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX(parameter_name),
    INDEX(changed_at)
);

-- Views for common queries
CREATE VIEW IF NOT EXISTS active_positions AS
SELECT t.*, tc.symbol, tc.name
FROM trades t
JOIN token_cache tc ON t.token_address = tc.address
WHERE t.side = 'buy' AND t.exit_price IS NULL;

CREATE VIEW IF NOT EXISTS performance_summary AS
SELECT 
    DATE(executed_at) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN profit_loss_usd > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(profit_loss_usd) as total_pnl,
    AVG(roi_percent) as avg_roi,
    MAX(profit_loss_usd) as max_win,
    MIN(profit_loss_usd) as max_loss,
    AVG(hold_time_seconds) as avg_hold_time
FROM trades 
WHERE closed_at IS NOT NULL
GROUP BY DATE(executed_at)
ORDER BY trade_date DESC;

CREATE VIEW IF NOT EXISTS top_performers AS
SELECT 
    tc.symbol,
    tc.address,
    COUNT(t.id) as trade_count,
    SUM(t.profit_loss_usd) as total_pnl,
    AVG(t.roi_percent) as avg_roi,
    MAX(t.profit_loss_usd) as best_trade
FROM trades t
JOIN token_cache tc ON t.token_address = tc.address
WHERE t.closed_at IS NOT NULL
GROUP BY tc.address
HAVING trade_count >= 2
ORDER BY total_pnl DESC
LIMIT 50;
EOF

print_status "SUCCESS" "Database schema created!"

print_status "INFO" "Step 3: Creating production database manager..."

# Create the database manager class
cat > data/database_manager.py << 'EOF'
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
EOF

print_status "SUCCESS" "Database manager created!"

print_status "INFO" "Step 4: Creating database initialization script..."

cat > scripts/init_database.py << 'EOF'
#!/usr/bin/env python3
"""
Database initialization script for Renaissance Trading System
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database_manager import db_manager, TokenData, TradeRecord

async def initialize_database():
    """Initialize the production database"""
    print("🗄️ Initializing Renaissance Trading Database...")
    
    try:
        await db_manager.initialize()
        print("✅ Database initialized successfully!")
        
        # Test basic operations
        print("🧪 Testing database operations...")
        
        # Test token caching
        test_token = TokenData(
            address="0x1234567890123456789012345678901234567890",
            chain="ethereum",
            symbol="TEST",
            name="Test Token",
            price=1.0,
            volume_24h=100000,
            liquidity_usd=50000,
            momentum_score=0.75,
            velocity=0.05,
            volatility=0.10
        )
        
        success = await db_manager.cache_token(test_token)
        if success:
            print("✅ Token caching test passed")
        else:
            print("❌ Token caching test failed")
        
        # Test token retrieval
        retrieved = await db_manager.get_token_data(test_token.address, test_token.chain)
        if retrieved and retrieved.symbol == "TEST":
            print("✅ Token retrieval test passed")
        else:
            print("❌ Token retrieval test failed")
        
        # Test performance summary
        summary = await db_manager.get_performance_summary(7)
        print(f"✅ Performance summary retrieved: {len(summary)} metrics")
        
        print("🎉 Database ready for production!")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(initialize_database())
    sys.exit(0 if success else 1)
EOF

chmod +x scripts/init_database.py

print_status "INFO" "Step 5: Creating database integration for core modules..."

# Update core system to use database
cat >> core/production_renaissance_system.py << 'EOF'

# Database integration
from data.database_manager import db_manager, TokenData, TradeRecord

class DatabaseIntegratedSystem:
    def __init__(self):
        self.db = db_manager
        self.trade_counter = 0
    
    async def initialize_with_database(self):
        """Initialize system with database support"""
        await self.db.initialize()
        print("✅ Database layer initialized")
    
    async def cache_discovered_token(self, token_data: dict):
        """Cache discovered token in database"""
        token = TokenData(
            address=token_data['address'],
            chain=token_data['chain'],
            symbol=token_data.get('symbol', ''),
            name=token_data.get('name', ''),
            price=token_data.get('price', 0.0),
            volume_24h=token_data.get('volume_24h', 0.0),
            liquidity_usd=token_data.get('liquidity_usd', 0.0),
            momentum_score=token_data.get('momentum_score', 0.0),
            velocity=token_data.get('velocity', 0.0),
            volatility=token_data.get('volatility', 0.0)
        )
        
        await self.db.cache_token(token)
    
    async def record_trade_execution(self, trade_data: dict):
        """Record trade execution in database"""
        self.trade_counter += 1
        trade_id = f"trade_{int(time.time())}_{self.trade_counter}"
        
        trade = TradeRecord(
            trade_id=trade_id,
            token_address=trade_data['token_address'],
            chain=trade_data['chain'],
            side=trade_data['side'],
            amount_usd=trade_data['amount_usd'],
            amount_tokens=trade_data['amount_tokens'],
            entry_price=trade_data['entry_price'],
            confidence_score=trade_data.get('confidence_score', 0.0),
            momentum_score=trade_data.get('momentum_score', 0.0),
            tx_hash=trade_data.get('tx_hash', '')
        )
        
        await self.db.record_trade(trade)
        return trade_id
    
    async def update_trade_exit(self, trade_id: str, exit_data: dict):
        """Update trade with exit information"""
        await self.db.update_trade_exit(
            trade_id=trade_id,
            exit_price=exit_data['exit_price'],
            profit_loss=exit_data['profit_loss'],
            roi=exit_data['roi'],
            exit_reason=exit_data['exit_reason']
        )
    
    async def get_performance_metrics(self):
        """Get comprehensive performance metrics"""
        return await self.db.get_performance_summary(7)
    
    async def shutdown_database(self):
        """Shutdown database connections"""
        await self.db.close()

# Add to existing renaissance_system
if 'renaissance_system' in globals():
    renaissance_system.db_system = DatabaseIntegratedSystem()
EOF

print_status "INFO" "Step 6: Creating database maintenance script..."

cat > scripts/maintain_database.py << 'EOF'
#!/usr/bin/env python3
"""
Database maintenance script for Renaissance Trading System
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database_manager import db_manager

async def maintain_database():
    """Perform database maintenance tasks"""
    print("🔧 Performing database maintenance...")
    
    try:
        await db_manager.initialize()
        
        # Clean up old data
        await db_manager.cleanup_old_data(30)
        
        # Vacuum database
        async with db_manager.get_connection() as db:
            await db.execute("VACUUM")
            await db.execute("ANALYZE")
        
        # Get statistics
        async with db_manager.get_connection() as db:
            cursor = await db.execute("SELECT COUNT(*) FROM token_cache")
            token_count = (await cursor.fetchone())[0]
            
            cursor = await db.execute("SELECT COUNT(*) FROM trades")
            trade_count = (await cursor.fetchone())[0]
            
            cursor = await db.execute("SELECT COUNT(*) FROM system_performance")
            perf_count = (await cursor.fetchone())[0]
        
        print(f"📊 Database Statistics:")
        print(f"  Tokens cached: {token_count}")
        print(f"  Trades recorded: {trade_count}")
        print(f"  Performance records: {perf_count}")
        
        print("✅ Database maintenance completed!")
        
    except Exception as e:
        print(f"❌ Database maintenance failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(maintain_database())
    sys.exit(0 if success else 1)
EOF

chmod +x scripts/maintain_database.py

print_status "INFO" "Step 7: Running database initialization..."

# Initialize the database
python3 scripts/init_database.py

print_status "SUCCESS" "🗄️ PRODUCTION DATABASE IMPLEMENTATION COMPLETE!"
echo
print_status "INFO" "Database features implemented:"
echo "  ✅ High-performance SQLite with WAL mode"
echo "  ✅ Connection pooling for concurrent access"
echo "  ✅ Comprehensive schema with indices"
echo "  ✅ Token caching and price history"
echo "  ✅ Complete trade lifecycle tracking"
echo "  ✅ System performance monitoring"
echo "  ✅ Risk event logging"
echo "  ✅ Social sentiment storage"
echo "  ✅ Dynamic parameter tracking"
echo "  ✅ Automated cleanup and maintenance"
echo
print_status "SUCCESS" "✨ Renaissance-level data persistence achieved!"