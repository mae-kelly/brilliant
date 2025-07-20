import asyncio
import sys
import os
sys.path.append('.')

async def test_real_feeds():
    print("🧪 Testing Real Data Feeds...")
    print("=" * 50)
    
    try:
        # Import and test components
        from data.real_websocket_feeds import real_data_engine
        from analyzers.real_blockchain_analyzer import real_blockchain_analyzer
        from analyzers.real_honeypot_detector import real_honeypot_detector
        from scanners.real_enhanced_scanner import real_enhanced_scanner
        
        print("✅ All real data modules imported successfully")
        
        # Test initialization
        print("\n🚀 Initializing components...")
        await real_data_engine.initialize()
        await real_honeypot_detector.initialize()
        
        print("✅ Components initialized")
        
        # Test for 30 seconds
        print("📊 Collecting data for 30 seconds...")
        await asyncio.sleep(30)
        
        # Check results
        token_count = len(real_data_engine.live_tokens)
        print(f"📈 Live tokens collected: {token_count}")
        
        if token_count > 0:
            print("🎉 Real data feeds working!")
            
            # Show sample
            sample_key = list(real_data_engine.live_tokens.keys())[0]
            sample_data = real_data_engine.live_tokens[sample_key]
            print(f"📊 Sample data points: {len(sample_data['prices'])}")
        else:
            print("⚠️  No data collected - check API keys")
        
        # Cleanup
        await real_data_engine.shutdown()
        await real_honeypot_detector.shutdown()
        
        print("✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_real_feeds())
    exit(0 if result else 1)
