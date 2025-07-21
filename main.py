#!/usr/bin/env python3
"""
DeFi Trading System - Main Entry Point
Run with: python main.py
"""

import asyncio
import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core', 'engine'))

from pipeline import main_pipeline

if __name__ == "__main__":
    print("🚀 Starting DeFi Momentum Trading System")
    try:
        asyncio.run(main_pipeline())
    except KeyboardInterrupt:
        print("\n⏹️ Trading system stopped by user")
    except Exception as e:
        print(f"💥 System error: {e}")
        sys.exit(1)
