#!/bin/bash

echo "🚀 LAUNCHING OPTIMIZED RENAISSANCE TRADING SYSTEM"
echo "================================================="

# Check dependencies
echo "🔍 Checking system dependencies..."
python3 -c "
import asyncio, aiohttp, numpy
print('✅ Core dependencies available')
"

if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Run: pip install aiohttp numpy"
    exit 1
fi

# Set default environment variables if not set
export DRY_RUN=${DRY_RUN:-true}
export ENABLE_REAL_TRADING=${ENABLE_REAL_TRADING:-false}
export MAX_POSITION_USD=${MAX_POSITION_USD:-10.0}
export MAX_SLIPPAGE=${MAX_SLIPPAGE:-0.03}

echo "⚙️ Configuration:"
echo "   DRY_RUN: $DRY_RUN"
echo "   ENABLE_REAL_TRADING: $ENABLE_REAL_TRADING"
echo "   MAX_POSITION_USD: $MAX_POSITION_USD"
echo ""

# Make sure all files are executable
chmod +x *.py

echo "🎯 Starting complete trading system..."
echo "Press Ctrl+C to stop"
echo ""

# Launch the complete system
python3 complete_trading_system.py
