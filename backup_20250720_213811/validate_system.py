#!/usr/bin/env python3
"""
System validation script - ensures all components are working
"""

import asyncio
import logging
import yaml
import sys
from typing import Dict, List
import importlib

async def validate_configurations() -> bool:
    """Validate all configuration files"""
    try:
        # Validate settings.yaml
        with open('settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        
        required_sections = ['redis', 'trading', 'risk', 'ml', 'network_config', 'scanning']
        for section in required_sections:
            if section not in settings:
                logging.error(f"Missing required section in settings.yaml: {section}")
                return False
        
        # Validate .env file
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        required_env_vars = [
            'ARBITRUM_RPC_URL', 'POLYGON_RPC_URL', 'OPTIMISM_RPC_URL',
            'WALLET_ADDRESS', 'PRIVATE_KEY', 'ALCHEMY_API_KEY'
        ]
        
        for var in required_env_vars:
            if not os.getenv(var):
                logging.error(f"Missing required environment variable: {var}")
                return False
        
        logging.info("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        return False

async def validate_modules() -> bool:
    """Validate all required modules can be imported"""
    required_modules = [
        'signal_detector', 'trade_executor', 'inference_model',
        'risk_manager', 'safety_checks', 'token_profiler',
        'anti_rug_analyzer', 'mempool_watcher', 'feedback_loop'
    ]
    
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
            logging.info(f"✅ Module {module_name} imported successfully")
        except ImportError as e:
            logging.error(f"❌ Failed to import {module_name}: {e}")
            return False
    
    return True

async def validate_network_connections() -> bool:
    """Validate network connections to all chains"""
    from web3 import Web3
    import os
    
    chains = {
        'arbitrum': os.getenv('ARBITRUM_RPC_URL'),
        'polygon': os.getenv('POLYGON_RPC_URL'),
        'optimism': os.getenv('OPTIMISM_RPC_URL')
    }
    
    for chain, rpc_url in chains.items():
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))
            if w3.is_connected():
                block_number = w3.eth.block_number
                logging.info(f"✅ {chain} connected - Block #{block_number:,}")
            else:
                logging.error(f"❌ Failed to connect to {chain}")
                return False
        except Exception as e:
            logging.error(f"❌ {chain} connection error: {e}")
            return False
    
    return True

async def validate_database() -> bool:
    """Validate database connections and schema"""
    try:
        import sqlite3
        import redis
        
        # Test SQLite
        conn = sqlite3.connect('token_cache.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        logging.info(f"✅ SQLite database connected - {len(tables)} tables")
        
        # Test Redis
        with open('settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        
        r = redis.Redis(
            host=settings['redis']['host'],
            port=settings['redis']['port'],
            decode_responses=False
        )
        r.ping()
        logging.info("✅ Redis connection successful")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Database validation failed: {e}")
        return False

async def validate_ml_models() -> bool:
    """Validate ML models can be loaded"""
    try:
        from core.models.inference_model import backup_20250720_213811.inference_model
        
        model = MomentumEnsemble()
        logging.info("✅ ML model initialized successfully")
        
        # Test prediction with dummy data
        import pandas as pd
        import numpy as np
        
        dummy_features = pd.DataFrame({
            'returns': np.random.normal(0, 0.01, 50),
            'volatility': np.random.uniform(0.1, 0.3, 50),
            'momentum': np.random.normal(0, 0.05, 50),
            'rsi': np.random.uniform(30, 70, 50),
            'bb_position': np.random.uniform(0, 1, 50),
            'volume_ma': np.random.uniform(1000, 10000, 50),
            'whale_activity': np.random.uniform(0, 0.2, 50),
            'price_acceleration': np.random.normal(0, 0.001, 50),
            'volatility_ratio': np.random.uniform(0.8, 1.2, 50),
            'momentum_strength': np.random.uniform(0, 0.1, 50),
            'swap_volume': np.random.uniform(1000, 10000, 50)
        })
        
        prediction = model.predict(dummy_features.tail(1))
        logging.info(f"✅ ML model prediction test: {prediction:.4f}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ ML model validation failed: {e}")
        return False

async def main():
    """Run all validation checks"""
    from infrastructure.monitoring.logging_config import infrastructure.monitoring.logging_config
    setup_logging()
    
    logging.info("🚀 Starting system validation...")
    
    checks = [
        ("Configuration", validate_configurations),
        ("Modules", validate_modules),
        ("Network", validate_network_connections),
        ("Database", validate_database),
        ("ML Models", validate_ml_models)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        logging.info(f"Running {check_name} validation...")
        try:
            if await check_func():
                passed += 1
            else:
                logging.error(f"❌ {check_name} validation failed")
        except Exception as e:
            logging.error(f"❌ {check_name} validation error: {e}")
    
    success_rate = passed / total
    logging.info(f"📊 Validation Results: {passed}/{total} checks passed ({success_rate:.1%})")
    
    if success_rate == 1.0:
        logging.info("🎉 ALL VALIDATIONS PASSED! System ready for deployment.")
        return True
    else:
        logging.error("❌ Some validations failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
