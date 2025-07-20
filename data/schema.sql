CREATE TABLE IF NOT EXISTS tokens (
    address TEXT PRIMARY KEY,
    chain TEXT NOT NULL,
    symbol TEXT,
    name TEXT,
    price REAL,
    volume_24h REAL,
    liquidity_usd REAL,
    momentum_score REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_address TEXT,
    chain TEXT,
    side TEXT,
    amount_usd REAL,
    price REAL,
    profit_loss REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tokens_scanned INTEGER,
    signals_generated INTEGER,
    trades_executed INTEGER,
    total_profit REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
