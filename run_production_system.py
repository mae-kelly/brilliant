#!/usr/bin/env python3
import asyncio
import argparse
from production_renaissance_system import renaissance_system

async def main():
    parser = argparse.ArgumentParser(description='Renaissance Trading System')
    parser.add_argument('--duration', type=float, default=1.0, help='Trading duration in hours')
    args = parser.parse_args()
    
    success = await renaissance_system.initialize_system()
    if success:
        await renaissance_system.start_production_trading(args.duration)
    else:
        print("❌ System initialization failed")

if __name__ == "__main__":
    asyncio.run(main())
