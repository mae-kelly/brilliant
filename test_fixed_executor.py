
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import sys
import os
sys.path.append('executors')

from fixed_real_executor import fixed_executor

async def test_executor():
    print("🧪 Testing Fixed Real Executor")
    print("=" * 50)
    
    stats = fixed_executor.get_performance_stats()
    print(f"📡 Connected chains: {stats['connected_chains']}")
    print(f"🎮 Simulation mode: {stats['simulation_mode']}")
    print("")
    
    test_token = "0x1234567890123456789012345678901234567890"
    
    print("🟢 Testing buy execution...")
    buy_result = await fixed_executor.execute_buy_trade(test_token, "arbitrum", 0.01)
    print(f"   Success: {buy_result.success}")
    print(f"   TX Hash: {buy_result.tx_hash[:10]}...")
    print(f"   Execution time: {buy_result.execution_time:.3f}s")
    print("")
    
    print("🔴 Testing sell execution...")
    sell_result = await fixed_executor.execute_sell_trade(test_token, "arbitrum", 10000)
    print(f"   Success: {sell_result.success}")
    print(f"   TX Hash: {sell_result.tx_hash[:10]}...")
    print(f"   Profit/Loss: {sell_result.profit_loss:+.6f} ETH")
    print(f"   Execution time: {sell_result.execution_time:.3f}s")
    print("")
    
    final_stats = fixed_executor.get_performance_stats()
    print("📊 Final Statistics:")
    print(f"   Total trades: {final_stats['total_trades']}")
    print(f"   Total profit: {final_stats['total_profit']:+.6f} ETH")
    print(f"   Avg profit/trade: {final_stats['avg_profit_per_trade']:+.6f} ETH")
    
    return buy_result.success and sell_result.success

if __name__ == "__main__":
    success = asyncio.run(test_executor())
    print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")
    sys.exit(0 if success else 1)
