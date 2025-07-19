
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import sys
import os
sys.path.append('scanners')

from ultra_scale_scanner import ultra_scanner

async def test_scanner():
    print("🧪 Testing Ultra-Scale Scanner (10k+ tokens/day)")
    print("=" * 60)
    
    try:
        await ultra_scanner.initialize()
        print("✅ Scanner initialized successfully")
        
        print("🔍 Running 60-second performance test...")
        await asyncio.sleep(60)
        
        signals = await ultra_scanner.get_signals(10)
        print(f"📊 Generated {len(signals)} momentum signals")
        
        for i, signal in enumerate(signals[:3]):
            print(f"  Signal {i+1}: {signal.address[:8]}... Score: {signal.momentum_score:.3f}")
        
        await ultra_scanner.shutdown()
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_scanner())
    sys.exit(0 if success else 1)
