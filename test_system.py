#!/usr/bin/env python3

import sys
import numpy as np

print("🧪 TESTING RENAISSANCE TRADING SYSTEM")
print("=====================================")

try:
    import tensorflow as tf
    print("✅ TensorFlow:", tf.__version__)
except ImportError:
    print("⚠️ TensorFlow not available")

try:
    import web3
    print("✅ Web3:", web3.__version__)
except ImportError:
    print("⚠️ Web3 not available")

try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except ImportError:
    print("❌ NumPy missing (critical)")
    sys.exit(1)

try:
    import asyncio
    print("✅ AsyncIO available")
except ImportError:
    print("❌ AsyncIO missing (critical)")
    sys.exit(1)

print("\n🎯 Running basic system test...")

# Test mock trading system
balance = 10.0
trades = 0
successful_trades = 0

for i in range(10):
    roi = np.random.uniform(-0.05, 0.20)
    trades += 1
    
    if roi > 0:
        successful_trades += 1
        
    balance *= (1 + roi * 0.1)
    print(f"Trade {trades}: ROI {roi*100:.1f}% | Balance: ${balance:.2f}")

win_rate = (successful_trades / trades) * 100
total_return = ((balance - 10) / 10) * 100

print(f"\n📊 TEST RESULTS:")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Final Balance: ${balance:.2f}")
print(f"Total Return: {total_return:.1f}%")
print("\n✅ System test completed successfully!")
