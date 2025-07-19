#!/usr/bin/env python3
"""
Test the optimized WebSocket scanner
"""

import asyncio
import time
from websocket_scanner_optimized import OptimizedWebSocketScanner

async def test_scanner():
    print("🧪 Testing Optimized WebSocket Scanner")
    print("=" * 40)
    
    scanner = OptimizedWebSocketScanner()
    
    try:
        # Initialize scanner
        print("🚀 Initializing scanner...")
        await scanner.initialize()
        
        # Test for 30 seconds
        print("⏳ Testing for 30 seconds...")
        start_time = time.time()
        total_signals = 0
        
        while time.time() - start_time < 30:
            signals = await scanner.get_signals()
            
            if signals:
                total_signals += len(signals)
                print(f"📊 Got {len(signals)} signals (Total: {total_signals})")
                
                for signal in signals[:3]:  # Show first 3
                    print(f"  🎯 {signal.address[:8]}... "
                          f"Score: {signal.momentum_score:.3f}")
            
            await asyncio.sleep(2)
            
        # Report results
        print("\n📈 Test Results:")
        print(f"Total signals detected: {total_signals}")
        print(f"Tokens processed: {scanner.tokens_processed}")
        print(f"Active connections: {len(scanner.connections)}")
        
        if total_signals > 0:
            print("✅ Scanner is working correctly!")
        else:
            print("⚠️ No signals detected (normal for short test)")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await scanner.shutdown()

if __name__ == "__main__":
    asyncio.run(test_scanner())
