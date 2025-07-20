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
