
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import time

async def test_production_system():
    print("🧪 Testing Production Trading System")
    print("=" * 50)
    
    try:
        from production_trading_system import production_system
        
        print("🚀 Initializing production system...")
        success = await production_system.initialize()
        
        if not success:
            print("❌ Initialization failed")
            return
            
        print("✅ System initialized successfully")
        print("⏳ Running test for 2 minutes...")
        
        start_time = time.time()
        trading_task = asyncio.create_task(production_system.trading_loop())
        
        await asyncio.sleep(120)
        
        production_system.running = False
        trading_task.cancel()
        
        await production_system.shutdown()
        
        print("\n🎯 TEST RESULTS:")
        print(f"Signals processed: {production_system.signals_processed}")
        print(f"Trades executed: {production_system.trades_executed}")
        print(f"Portfolio value: ${production_system.portfolio_value:.2f}")
        print(f"ROI: {((production_system.portfolio_value - 10.0) / 10.0) * 100:.2f}%")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_production_system())
