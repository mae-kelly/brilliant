#!/usr/bin/env python3
"""
Minimal test to verify your DeFi trading system works
"""

import sys
import os
from dotenv import load_dotenv

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        import web3, pandas, numpy, yaml, aiohttp, fastapi, redis
        from eth_account import Account
        print("✅ Core packages imported")
    except ImportError as e:
        print(f"❌ Core import error: {e}")
        return False
    
    try:
        import abi
        print("✅ abi.py imported")
    except ImportError as e:
        print(f"❌ abi.py error: {e}")
        return False
    
    try:
        from intelligence.signals.signal_detector import SignalDetector
        print("✅ signal_detector.py imported")
    except ImportError as e:
        print(f"❌ signal_detector.py error: {e}")
        return False
        
    try:
        from core.models.inference_model import MomentumEnsemble
        print("✅ inference_model.py imported")
    except ImportError as e:
        print(f"❌ inference_model.py error: {e}")
        return False
    
    return True

def test_config():
    """Test configuration files"""
    print("\n📝 Testing configuration...")
    
    if not os.path.exists('.env'):
        print("❌ .env file missing")
        return False
    
    if not os.path.exists('settings.yaml'):
        print("❌ settings.yaml missing")
        return False
        
    try:
        load_dotenv()
        arbitrum_url = os.getenv('ARBITRUM_RPC_URL')
        if arbitrum_url and 'alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX' in arbitrum_url:
            print("✅ Alchemy RPC configured")
        else:
            print("⚠️ RPC URL not configured")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    
    return True

def test_rpc_connection():
    """Test RPC connection with your Alchemy key"""
    print("\n🌐 Testing RPC connection...")
    
    try:
        load_dotenv()
        from web3 import Web3
        
        arbitrum_url = os.getenv('ARBITRUM_RPC_URL')
        if not arbitrum_url:
            print("⚠️ No RPC URL configured")
            return False
            
        w3 = Web3(Web3.HTTPProvider(arbitrum_url, request_kwargs={'timeout': 10}))
        
        if w3.is_connected():
            block = w3.eth.block_number
            print(f"✅ Arbitrum connected (Block #{block:,})")
            return True
        else:
            print("❌ RPC connection failed")
            return False
            
    except Exception as e:
        print(f"❌ RPC error: {e}")
        return False

def test_wallet():
    """Test wallet configuration"""
    print("\n🔐 Testing wallet...")
    
    try:
        load_dotenv()
        from eth_account import Account
        
        private_key = os.getenv('PRIVATE_KEY')
        wallet_address = os.getenv('WALLET_ADDRESS')
        
        if not private_key or not wallet_address:
            print("⚠️ Wallet not configured")
            return False
            
        account = Account.from_key(private_key)
        
        if account.address.lower() == wallet_address.lower():
            print(f"✅ Wallet valid: {wallet_address[:6]}...{wallet_address[-4:]}")
            return True
        else:
            print("❌ Wallet mismatch")
            return False
            
    except Exception as e:
        print(f"❌ Wallet error: {e}")
        return False

def test_components():
    """Test core trading components"""
    print("\n🔧 Testing components...")
    
    try:
        # Test signal detector
        from intelligence.signals.signal_detector import SignalDetector
        from unittest.mock import Mock
        
        mock_chains = {'arbitrum': Mock()}
        mock_redis = Mock()
        
        detector = SignalDetector(mock_chains, mock_redis)
        print("✅ SignalDetector initialized")
        
        # Test inference model
        from core.models.inference_model import MomentumEnsemble
        
        model = MomentumEnsemble()
        print("✅ MomentumEnsemble initialized")
        
        # Test prediction with dummy data
        import pandas as pd
        import numpy as np
        
        dummy_data = pd.DataFrame({
            'returns': np.random.normal(0, 0.01, 20),
            'volatility': np.random.uniform(0.1, 0.3, 20),
            'momentum': np.random.normal(0, 0.05, 20),
            'rsi': np.random.uniform(30, 70, 20),
            'bb_position': np.random.uniform(0, 1, 20),
            'volume_ma': np.random.uniform(1000, 10000, 20),
            'whale_activity': np.random.uniform(0, 0.2, 20),
            'price_acceleration': np.random.normal(0, 0.001, 20),
            'volatility_ratio': np.random.uniform(0.8, 1.2, 20),
            'momentum_strength': np.random.uniform(0, 0.1, 20),
            'swap_volume': np.random.uniform(1000, 10000, 20)
        })
        
        score = model.predict(dummy_data.tail(1))
        print(f"✅ Model prediction: {score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 DeFi Trading System - Minimal Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config, 
        test_rpc_connection,
        test_wallet,
        test_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready!")
        print("\n🎯 Next steps:")
        print("1. Fund your wallet with 0.01 ETH")
        print("2. Run: python3 pipeline.py")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)