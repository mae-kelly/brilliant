#!/usr/bin/env python3
import asyncio
import argparse
import sys
import os

sys.path.append('.')

from production_renaissance_system import renaissance_system

async def run_system(duration_hours=1, target_tokens=10000):
    print(f"""
🚀🚀🚀 RENAISSANCE DEFI TRADING SYSTEM 🚀🚀🚀
================================================

🎯 Configuration:
   • Duration: {duration_hours} hours
   • Target: {target_tokens:,} tokens/day
   • Starting capital: $10.00
   • Mode: Production simulation
   
🧠 Features:
   • Ultra-scale token scanning
   • Real-time momentum detection  
   • ML-driven predictions
   • Multi-chain arbitrage
   • Dynamic position management
   • Online learning optimization
   
================================================
""")
    
    try:
        success = await renaissance_system.initialize_system()
        if success:
            print("🎯 All systems ready! Starting autonomous trading...\n")
            await renaissance_system.start_production_trading(duration_hours)
        else:
            print("❌ System initialization failed")
            return False
    except KeyboardInterrupt:
        print("\n🛑 Trading interrupted by user")
        await renaissance_system.shutdown_system()
    except Exception as e:
        print(f"❌ System error: {e}")
        await renaissance_system.shutdown_system()
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Renaissance DeFi Trading System')
    parser.add_argument('--duration', type=float, default=1.0, 
                       help='Trading duration in hours (default: 1.0)')
    parser.add_argument('--target', type=int, default=10000,
                       help='Target tokens per day (default: 10000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"Starting Renaissance Trading System...")
    print(f"Duration: {args.duration} hours")
    print(f"Target: {args.target:,} tokens/day")
    print()
    
    try:
        result = asyncio.run(run_system(args.duration, args.target))
        if result:
            print("🎉 Trading session completed successfully!")
        else:
            print("⚠️ Trading session completed with issues")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
