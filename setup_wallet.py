#!/usr/bin/env python3
"""
Secure Ethereum Wallet Setup for DeFi Trading
Generates a new Ethereum wallet for testing purposes
"""

import os
from eth_account import Account
from web3 import Web3
import secrets

def generate_secure_wallet():
    """Generate a cryptographically secure Ethereum wallet"""
    
    print("🔐 Generating secure Ethereum wallet for DeFi trading...")
    
    # Generate cryptographically secure private key
    private_key = "0x" + secrets.token_hex(32)
    
    # Create account from private key
    account = Account.from_key(private_key)
    
    print(f"\n✅ New Ethereum Wallet Generated:")
    print(f"📍 Address: {account.address}")
    print(f"🔑 Private Key: {private_key}")
    
    print(f"\n🚨 IMPORTANT SECURITY NOTES:")
    print(f"1. This wallet is for TESTING ONLY with small amounts")
    print(f"2. Never share your private key publicly") 
    print(f"3. Store your private key securely")
    print(f"4. This address works on Ethereum, Arbitrum, Polygon, and Optimism")
    
    return account.address, private_key

def create_env_file(wallet_address, private_key):
    """Create .env file with wallet configuration"""
    
    env_content = f"""# DeFi Trading System Environment Variables
# Generated: {import datetime; datetime.datetime.now().isoformat()}

# Alchemy RPC URLs (using your API key)
ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX
OPTIMISM_RPC_URL=https://opt-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX

# Backup RPC URLs
ARBITRUM_BACKUP_RPC_URL=https://arbitrum-one.publicnode.com
POLYGON_BACKUP_RPC_URL=https://polygon.llamarpc.com
OPTIMISM_BACKUP_RPC_URL=https://mainnet.optimism.io

# API Keys
ETHERSCAN_API_KEY=K4SEVFZ3PI8STM73VKV84C8PYZJUK7HB2G
ALCHEMY_API_KEY=alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX

# Generated Wallet Configuration
WALLET_ADDRESS={wallet_address}
PRIVATE_KEY={private_key}

# Trading Configuration
STARTING_BALANCE=0.01
MAX_POSITION_SIZE=0.002
ENABLE_LIVE_TRADING=false

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n📄 Created .env file with wallet configuration")

def test_rpc_connections():
    """Test RPC connections with Alchemy"""
    
    print(f"\n🌐 Testing RPC connections...")
    
    rpc_urls = {
        'Arbitrum': 'https://arb-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX',
        'Polygon': 'https://polygon-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX', 
        'Optimism': 'https://opt-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX'
    }
    
    for network, url in rpc_urls.items():
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 10}))
            if w3.is_connected():
                latest_block = w3.eth.block_number
                print(f"✅ {network}: Connected (Block #{latest_block:,})")
            else:
                print(f"❌ {network}: Connection failed")
        except Exception as e:
            print(f"❌ {network}: Error - {e}")

def check_wallet_balances(wallet_address):
    """Check wallet balances across all networks"""
    
    print(f"\n💰 Checking wallet balances for {wallet_address}...")
    
    rpc_urls = {
        'Arbitrum': 'https://arb-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX',
        'Polygon': 'https://polygon-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX',
        'Optimism': 'https://opt-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX'
    }
    
    total_balance = 0
    
    for network, url in rpc_urls.items():
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 10}))
            if w3.is_connected():
                balance_wei = w3.eth.get_balance(wallet_address)
                balance_eth = w3.from_wei(balance_wei, 'ether')
                total_balance += balance_eth
                
                if balance_eth > 0:
                    print(f"💎 {network}: {balance_eth:.6f} ETH (${balance_eth * 3000:.2f})")
                else:
                    print(f"🔍 {network}: 0.000000 ETH (Empty)")
            else:
                print(f"❌ {network}: Connection failed")
        except Exception as e:
            print(f"❌ {network}: Error - {e}")
    
    print(f"\n💼 Total Balance: {total_balance:.6f} ETH (${total_balance * 3000:.2f})")
    
    if total_balance == 0:
        print(f"\n⚠️  FUND YOUR WALLET:")
        print(f"   Send 0.01-0.02 ETH to: {wallet_address}")
        print(f"   Recommended: Start with 0.01 ETH (~$30) for testing")
        print(f"   You can get ETH from:")
        print(f"   - Coinbase, Binance, or other exchanges")
        print(f"   - Bridge from other chains")
        print(f"   - Faucets (for testnets only)")

def main():
    print("🚀 DeFi Momentum Trading Wallet Setup")
    print("=" * 50)
    
    # Generate new wallet
    wallet_address, private_key = generate_secure_wallet()
    
    # Create .env file
    create_env_file(wallet_address, private_key)
    
    # Test connections
    test_rpc_connections()
    
    # Check balances
    check_wallet_balances(wallet_address)
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Fund your wallet with 0.01-0.02 ETH")
    print(f"2. Run: chmod +x test_pipeline.sh && ./test_pipeline.sh")
    print(f"3. If tests pass, start trading with: python3 pipeline.py")
    print(f"4. Monitor at: http://localhost:8001/metrics")
    
    print(f"\n🔒 SECURITY REMINDER:")
    print(f"   - Keep your private key safe and never share it")
    print(f"   - This is for testing with small amounts only")
    print(f"   - Enable 2FA on all your accounts")

if __name__ == "__main__":
    main()