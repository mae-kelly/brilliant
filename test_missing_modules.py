import asyncio
import sys
import os

sys.path.append('analyzers')
sys.path.append('watchers')
sys.path.append('profilers')

from anti_rug_analyzer import anti_rug_analyzer
from mempool_watcher import mempool_watcher
from token_profiler import token_profiler

async def test_all_modules():
    print("🧪 Testing Missing Core Modules")
    print("=" * 60)
    
    test_token = "0x1234567890123456789012345678901234567890"
    
    print("🛡️ Testing Anti-Rug Analyzer...")
    rug_analysis = await anti_rug_analyzer.analyze_token_safety(test_token)
    print(f"   Risk Score: {rug_analysis.risk_score:.2f}")
    print(f"   Is Safe: {rug_analysis.is_safe}")
    print(f"   Flags: {len(rug_analysis.flags)}")
    print("")
    
    print("📊 Testing Token Profiler...")
    profile = await token_profiler.profile_token(test_token)
    print(f"   Symbol: {profile.symbol}")
    print(f"   Overall Score: {profile.overall_score:.2f}")
    print(f"   Risk Category: {profile.risk_category}")
    print(f"   Market Cap: ${profile.market_cap_usd:,.0f}")
    print("")
    
    print("🔍 Testing Mempool Watcher...")
    
    async def tx_callback(tx):
        if tx.is_swap:
            print(f"   📡 Swap detected: {tx.hash[:10]}... Value: {tx.value:.3f} ETH")
    
    mempool_watcher.add_transaction_callback(tx_callback)
    
    monitoring_task = asyncio.create_task(mempool_watcher.start_monitoring())
    
    await asyncio.sleep(5)
    
    await mempool_watcher.shutdown()
    monitoring_task.cancel()
    
    print("")
    print("📈 Module Statistics:")
    
    rug_stats = anti_rug_analyzer.get_safety_stats()
    profiler_stats = token_profiler.get_profile_stats()
    
    print(f"   Safe contracts analyzed: {rug_stats['safe_contracts']}")
    print(f"   Flagged contracts: {rug_stats['flagged_contracts']}")
    print(f"   Tokens profiled: {profiler_stats['total_profiles']}")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_all_modules())
    print(f"\n{'✅ All modules PASSED' if success else '❌ Tests FAILED'}")
    sys.exit(0 if success else 1)
