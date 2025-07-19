import asyncio
import time

async def test_fixed_system():
    print("🧪 Testing Fixed Production Trading System")
    print("=" * 60)
    
    try:
        from fixed_production_system import fixed_system
        
        print("🚀 Initializing fixed system...")
        success = await fixed_system.initialize()
        
        if not success:
            print("❌ Initialization failed")
            return
            
        print("✅ System initialized successfully")
        print("⏳ Running test for 2 minutes...")
        
        start_time = time.time()
        trading_task = asyncio.create_task(fixed_system.trading_loop())
        
        await asyncio.sleep(120)
        
        fixed_system.running = False
        trading_task.cancel()
        
        await fixed_system.shutdown()
        
        print("\n🎯 FIXED SYSTEM TEST RESULTS:")
        print(f"Signals processed: {fixed_system.signals_processed}")
        print(f"Trades executed: {fixed_system.trades_executed}")
        print(f"Portfolio value: ${fixed_system.portfolio_value:.2f}")
        print(f"ROI: {((fixed_system.portfolio_value - 10.0) / 10.0) * 100:.2f}%")
        print("\n✅ Fixed system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixed_system())
