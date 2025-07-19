#!/bin/bash

echo "🧠 LAUNCHING RENAISSANCE DEFI TRADING SYSTEM"
echo "============================================="

echo "📦 Installing core dependencies..."
pip install web3 tensorflow numpy scipy scikit-learn requests aiohttp websockets pandas matplotlib

echo "🚀 Starting system..."
python3 launch_system.py
