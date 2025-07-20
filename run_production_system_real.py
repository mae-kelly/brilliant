#!/usr/bin/env python3
import asyncio
import argparse
import sys
import os

# Add paths
sys.path.append('.')
sys.path.append('data')
sys.path.append('analyzers')
sys.path.append('scanners')

async def validate_environment():
    """Validate environment setup"""
    required_vars = ['ALCHEMY_API_KEY', 'PRIVATE_KEY', 'WALLET_ADDRESS']
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.startswith('your_') or value == '0x0000000000000000000000000000000000000000':
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("\nPlease set these:")
        for var in missing_vars:
            print(f"export {var}=your_actual_value")
        return False
    
    print("✅ Environment validated")
    return True

async def test_connections():
    """Test blockchain connections"""
    print("🔍 Testing connections...")
    
    try:
        from web3 import Web3
        alchemy_key = os.getenv('ALCHEMY_API_KEY')
        
        w3 = Web3(Web3.HTTPProvider(f'https://eth-mainnet.g.alchemy.com/v2/{alchemy_key}'))
        if w3.is_connected():
            latest_block = w3.eth.block_number
            print(f"✅ Ethereum: Block {latest_block}")
            
            # Check balance
            wallet = os.getenv('WALLET_ADDRESS')
            balance = w3.eth.get_balance(wallet)
            balance_eth = w3.from_wei(balance, 'ether')
            print(f"💰 Balance: {balance_eth:.6f} ETH")
            
            return True
        else:
            print("❌ Connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

async def run_real_system(duration_hours=1, target_tokens=10000):
    print(f"""
🔥🔥🔥 REAL DATA RENAISSANCE SYSTEM 🔥🔥🔥
============================================

🎯 Configuration:
   • Duration: {duration_hours} hours
   • Target: {target_tokens:,} tokens/day
   • Mode: REAL BLOCKCHAIN DATA
   
🔥 Real Features:
   • Live WebSocket feeds
   • Real API integration
   • Authentic momentum detection
   • Live safety verification
   
============================================
""")
    
    # Validate setup
    if not await validate_environment():
        return False
    
    if not await test_connections():
        return False
    
    try:
        # Import components
        from data.real_websocket_feeds import real_data_engine
        from analyzers.real_honeypot_detector import real_honeypot_detector
        from scanners.real_enhanced_scanner import real_enhanced_scanner
        
        # Initialize
        print("🚀 Initializing real data system...")
        await real_data_engine.initialize()
        await real_honeypot_detector.initialize()
        await real_enhanced_scanner.initialize()
        
        print("✅ All systems operational!")
        print(f"🎯 Running for {duration_hours} hours...\n")
        
        # Main trading loop
        end_time = asyncio.get_event_loop().time() + (duration_hours * 3600)
        signal_count = 0
        
        while asyncio.get_event_loop().time() < end_time:
            # Get real signals
            signals = await real_enhanced_scanner.get_real_signals(max_signals=5)
            
            if signals:
                signal_count += len(signals)
                print(f"🔥 REAL SIGNALS: {len(signals)}")
                
                for signal in signals:
                    print(f"   📊 {signal.address[:8]}... "
                          f"Momentum: {signal.momentum_score:.3f} "
                          f"Safety: {signal.safety_score:.3f} "
                          f"Chain: {signal.chain}")
                    
                    if signal.momentum_score > 0.85 and signal.safety_score > 0.8:
                        print(f"      🚀 HIGH CONFIDENCE SIGNAL!")
            
            await asyncio.sleep(15)
        
        print(f"\n🏁 Session complete! Total signals: {signal_count}")
        
        # Cleanup
        await real_enhanced_scanner.shutdown()
        await real_data_engine.shutdown()
        await real_honeypot_detector.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ System error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Real Data Renaissance System')
    parser.add_argument('--duration', type=float, default=1.0, help='Duration in hours')
    parser.add_argument('--target', type=int, default=10000, help='Target tokens/day')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🔥 Starting Real Data Renaissance System...")
    
    try:
        result = asyncio.run(run_real_system(args.duration, args.target))
        if result:
            print("🎉 Real data session completed successfully!")
        else:
            print("⚠️ Session had issues")
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
