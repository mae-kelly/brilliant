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
