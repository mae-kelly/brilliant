
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import sqlite3
import os
import threading
from datetime import datetime, timedelta

DB_PATH = './cache/token_cache.db'
DB_LOCK = threading.Lock()


def init_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT UNIQUE NOT NULL,
            name TEXT,
            symbol TEXT,
            decimals INTEGER,
            creation_time TIMESTAMP,
            chain TEXT,
            factory TEXT,
            dex TEXT,
            pair_address TEXT,
            verified BOOLEAN DEFAULT FALSE,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS token_safety (
            token_address TEXT PRIMARY KEY,
            honeypot_checked BOOLEAN DEFAULT FALSE,
            is_honeypot BOOLEAN,
            is_lp_locked BOOLEAN,
            has_trading_pause BOOLEAN,
            rug_score INTEGER,
            audit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS token_features (
            token_address TEXT PRIMARY KEY,
            price_delta REAL,
            volume_delta REAL,
            liquidity_delta REAL,
            volatility_burst REAL,
            velocity REAL,
            momentum REAL,
            entropy REAL,
            breakout_score REAL,
            classified_as_breakout BOOLEAN,
            prediction_timestamp TIMESTAMP
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS token_policy (
            token_address TEXT PRIMARY KEY,
            is_whitelisted BOOLEAN DEFAULT FALSE,
            is_blacklisted BOOLEAN DEFAULT FALSE,
            reason TEXT,
            flagged_by TEXT,
            flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_cache (
            token_address TEXT PRIMARY KEY,
            prediction_confidence REAL,
            entropy_score REAL,
            model_version TEXT,
            ttl_expiration TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT,
            timestamp TIMESTAMP,
            price REAL,
            volume REAL,
            liquidity REAL,
            volatility REAL
        );
        """)
        conn.commit()
        conn.close()


def insert_token_metadata(token):
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO tokens (
            token_address, name, symbol, decimals, creation_time, chain, factory, dex, pair_address, verified
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            token['address'],
            token['name'],
            token['symbol'],
            token['decimals'],
            token['created'],
            token['chain'],
            token['factory'],
            token['dex'],
            token['pair'],
            token.get('verified', True)
        ))
        conn.commit()
        conn.close()


def cache_token_features(token_address, features: dict):
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO token_features (
            token_address, price_delta, volume_delta, liquidity_delta, volatility_burst,
            velocity, momentum, entropy, breakout_score, classified_as_breakout, prediction_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            token_address,
            features.get('price_delta'),
            features.get('volume_delta'),
            features.get('liquidity_delta'),
            features.get('volatility_burst'),
            features.get('velocity'),
            features.get('momentum'),
            features.get('entropy'),
            features.get('breakout_score'),
            features.get('classified_as_breakout'),
            datetime.utcnow()
        ))
        conn.commit()
        conn.close()


def save_safety_audit(token_address, audit):
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO token_safety (
            token_address, honeypot_checked, is_honeypot,
            is_lp_locked, has_trading_pause, rug_score, audit_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            token_address,
            True,
            audit.get('is_honeypot'),
            audit.get('is_lp_locked'),
            audit.get('has_trading_pause'),
            audit.get('rug_score', 0),
            datetime.utcnow()
        ))
        conn.commit()
        conn.close()


def cache_model_prediction(token_address, confidence, entropy, model_version):
    ttl_expiration = datetime.utcnow() + timedelta(seconds=60)
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO prediction_cache (
            token_address, prediction_confidence, entropy_score, model_version,
            ttl_expiration, last_accessed
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            token_address, confidence, entropy, model_version,
            ttl_expiration, datetime.utcnow()
        ))
        conn.commit()
        conn.close()


def get_cached_prediction(token_address):
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        SELECT prediction_confidence, entropy_score, model_version, ttl_expiration
        FROM prediction_cache WHERE token_address = ?
        """, (token_address,))
        row = c.fetchone()
        conn.close()
        if row:
            ttl = datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
            if datetime.utcnow() < ttl:
                return {
                    'confidence': row[0],
                    'entropy': row[1],
                    'version': row[2],
                    'valid': True
                }
        return None


def blacklist_token(token_address, reason, flagged_by='sniper_ai'):
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        INSERT OR REPLACE INTO token_policy (
            token_address, is_blacklisted, reason, flagged_by, flagged_at
        ) VALUES (?, ?, ?, ?, ?)
        """, (
            token_address, True, reason, flagged_by, datetime.utcnow()
        ))
        conn.commit()
        conn.close()


def is_token_blacklisted(token_address):
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        SELECT is_blacklisted FROM token_policy WHERE token_address = ?
        """, (token_address,))
        row = c.fetchone()
        conn.close()
        return row and row[0] == 1


def log_price_snapshot(token_address, price, volume, liquidity, volatility):
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
        INSERT INTO price_history (
            token_address, timestamp, price, volume, liquidity, volatility
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            token_address, datetime.utcnow(), price, volume, liquidity, volatility
        ))
        conn.commit()
        conn.close()


# Initialize the database at import
if not os.path.exists(DB_PATH):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db()
