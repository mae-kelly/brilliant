#!/usr/bin/env python3
import asyncio
import signal
import sys
import os
from mev_engine_optimized import mev_engine
from yield_optimizer_simplified import yield_optimizer
from bridge_arbitrage_simplified import bridge_arbitrage

class UltimateProfitSystem:
    def __init__(self):
        self.running = False
        self.engines = []
        
    async def start_system(self):
        print("🚀 ULTIMATE DeFi PROFIT SYSTEM")
        print("=" * 50)
        print("🔥 MEV Arbitrage Engine")
        print("🌾 Yield Farming Optimizer") 
        print("🌉 Cross-Chain Bridge Arbitrage")
        print("=" * 50)
        
        if not os.getenv('PRIVATE_KEY') or os.getenv('PRIVATE_KEY').startswith('0x0000'):
            print("⚠️  Running in SIMULATION mode")
            print("   Configure environment variables for live trading")
        else:
            print("💰 LIVE TRADING mode activated")
        
        print("=" * 50)
        
        self.running = True
        
        self.engines = [
            asyncio.create_task(mev_engine.start_mev_engine()),
            asyncio.create_task(yield_optimizer.start_yield_optimization()),
            asyncio.create_task(bridge_arbitrage.start_bridge_arbitrage())
        ]
        
        try:
            await asyncio.gather(*self.engines)
        except KeyboardInterrupt:
            await self.shutdown()
    
    async def shutdown(self):
        print("\n🛑 Shutting down profit engines...")
        
        mev_engine.running = False
        yield_optimizer.running = False
        bridge_arbitrage.running = False
        
        for task in self.engines:
            task.cancel()
        
        print("✅ System shutdown complete")

def signal_handler(sig, frame):
    print("\n⚠️ Interrupt received")
    sys.exit(0)

async def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    system = UltimateProfitSystem()
    await system.start_system()

if __name__ == "__main__":
    asyncio.run(main())
