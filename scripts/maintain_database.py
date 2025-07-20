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
