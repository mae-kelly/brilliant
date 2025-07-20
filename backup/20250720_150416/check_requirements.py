#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
except ImportError:
    def get_dynamic_config(): return {"volatility_threshold": 0.1, "confidence_threshold": 0.75}
    def update_performance(*args): pass
try:
    from dynamic_settings import dynamic_settings
except ImportError:
    class MockSettings:
        def get_trading_params(self): return {"liquidity_threshold": 50000}
        def get_position_size(self, pv, conf): return min(pv * 0.1, 1.0)
    dynamic_settings = MockSettings()
"""
Quick requirements checker for Renaissance Trading System
"""

import sys

REQUIRED_PACKAGES = [
    'numpy', 'pandas', 'aiohttp', 'web3', 'asyncio',
    'requests', 'json', 'time', 'os', 'threading'
]

OPTIONAL_PACKAGES = [
    'tensorflow', 'scikit-learn', 'scipy', 'fastapi'
]

def check_package(package_name, required=True):
    try:
        __import__(package_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        if required:
            print(f"❌ {package_name} (REQUIRED)")
        else:
            print(f"⚠️  {package_name} (optional)")
        return False

def main():
    print("🔍 Renaissance Trading System - Requirements Check")
    print("=" * 50)
    
    print("\n📋 Required Packages:")
    required_failed = 0
    for package in REQUIRED_PACKAGES:
        if not check_package(package, required=True):
            required_failed += 1
    
    print("\n📦 Optional Packages:")
    for package in OPTIONAL_PACKAGES:
        check_package(package, required=False)
    
    print(f"\n📊 Summary:")
    if required_failed == 0:
        print("🎉 All required packages available!")
        print("🚀 System ready for trading!")
        sys.exit(0)
    else:
        print(f"❌ {required_failed} required packages missing")
        print("🔧 Run the dependency installer again")
        sys.exit(1)

if __name__ == "__main__":
    main()
