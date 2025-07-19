#!/usr/bin/env python3
"""
Test the working WebSocket scanner with real token discovery
"""

import asyncio
import time
from websocket_scanner_working import WorkingWebSocketScanner

async def test_working_scanner():
    print("🧪 Testing Working WebSocket Scanner")
    print("=" * 40)
    
    scanner = WorkingWebSocketScanner()
    
    try:
        # Initialize scanner
        print("🚀 Initializing scanner with API fallback...")
        await scanner.initialize()
        
        # Test for 60 seconds to allow API calls
        print("⏳ Testing for 60 seconds...")
        start_time = time.time()
        total_signals = 0
        token_discoveries = 0
        
        while time.time() - start_time < 60:
            # Check for new token discoveries
            current_tokens = len(scanner.discovered_tokens)
            if current_tokens > token_discoveries:
                new_discoveries = current_tokens - token_discoveries
                token_discoveries = current_tokens
                print(f"🔍 Discovered {new_discoveries} new tokens (Total: {token_discoveries})")
            
            # Check for momentum signals
            signals = await scanner.get_signals()
            
            if signals:
                total_signals += len(signals)
                print(f"📊 Got {len(signals)} signals (Total: {total_signals})")
                
                for signal in signals[:3]:  # Show first 3
                    print(f"  🎯 {signal.address[:8]}... "
                          f"Chain: {signal.chain} "
                          f"Score: {signal.momentum_score:.3f} "
                          f"Liquidity: ${signal.liquidity_usd:,.0f}")
            
            await asyncio.sleep(3)
            
        # Report final results
        print("\n📈 Test Results:")
        print(f"Total tokens discovered: {len(scanner.discovered_tokens)}")
        print(f"Total signals generated: {total_signals}")
        print(f"Tokens processed: {scanner.tokens_processed}")
        print(f"API calls made: {scanner.api_calls_made}")
        
        if total_signals > 0:
            print("✅ Scanner is working correctly and generating signals!")
        elif len(scanner.discovered_tokens) > 0:
            print("✅ Scanner is discovering tokens but signals need tuning")
        else:
            print("⚠️ Scanner running but may need better endpoints")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await scanner.shutdown()

if __name__ == "__main__":
    asyncio.run(test_working_scanner())
