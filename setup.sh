#!/bin/bash

# Quick DeFi Trading System Setup - Clean Version
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 DeFi Trading System - Quick Setup${NC}"
echo "=========================================="

# Clean up corrupted packages
echo -e "${BLUE}🧹 Cleaning up corrupted packages...${NC}"
rm -rf ~/.pyenv/versions/3.11.9/lib/python3.11/site-packages/~orch 2>/dev/null || true
rm -rf ~/.pyenv/versions/3.11.9/lib/python3.11/site-packages/~andas 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Create .env with your API keys
echo -e "${BLUE}📝 Creating .env configuration...${NC}"
cat > .env << 'EOF'
# DeFi Trading System Environment Variables

# Alchemy RPC URLs (your API key)
ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX
OPTIMISM_RPC_URL=https://opt-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX

# API Keys
ETHERSCAN_API_KEY=K4SEVFZ3PI8STM73VKV84C8PYZJUK7HB2G
ALCHEMY_API_KEY=alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX

# Will be filled by script
WALLET_ADDRESS=
PRIVATE_KEY=

# Trading Configuration
STARTING_BALANCE=0.01
ENABLE_LIVE_TRADING=false
EOF

# Install only essential packages
echo -e "${BLUE}📦 Installing essential packages...${NC}"
pip install --quiet --no-cache-dir web3 eth-account pandas numpy fastapi aiohttp redis pyyaml python-dotenv scikit-learn

# Generate wallet
echo -e "${BLUE}🔐 Generating Ethereum wallet...${NC}"
python3 -c "
import os
from eth_account import Account
import secrets

private_key = '0x' + secrets.token_hex(32)
account = Account.from_key(private_key)

print(f'Generated Wallet:')
print(f'Address: {account.address}')
print(f'Private Key: {private_key}')

# Update .env
with open('.env', 'r') as f:
    content = f.read()

content = content.replace('WALLET_ADDRESS=', f'WALLET_ADDRESS={account.address}')
content = content.replace('PRIVATE_KEY=', f'PRIVATE_KEY={private_key}')

with open('.env', 'w') as f:
    f.write(content)

print('✅ Wallet saved to .env')
"

# Test core functionality
echo -e "${BLUE}🧪 Testing core functionality...${NC}"
python3 -c "
import web3, pandas, numpy, fastapi, aiohttp, redis, yaml
from eth_account import Account
print('✅ Core packages working')

try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    # Test RPC connection
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(os.getenv('ARBITRUM_RPC_URL')))
    if w3.is_connected():
        print(f'✅ Arbitrum RPC connected (Block #{w3.eth.block_number})')
    else:
        print('⚠️ RPC connection failed')
        
except Exception as e:
    print(f'⚠️ RPC test: {e}')
"

# Test your specific components
echo -e "${BLUE}🔧 Testing your components...${NC}"
python3 -c "
try:
    import abi
    print('✅ abi.py working')
except Exception as e:
    print(f'❌ abi.py: {e}')

try:
    from signal_detector import SignalDetector
    print('✅ signal_detector.py working')
except Exception as e:
    print(f'❌ signal_detector.py: {e}')
    
try:
    from inference_model import MomentumEnsemble
    print('✅ inference_model.py working')
except Exception as e:
    print(f'❌ inference_model.py: {e}')

try:
    from pipeline import main_pipeline
    print('✅ pipeline.py working')
except Exception as e:
    print(f'❌ pipeline.py: {e}')
"

echo -e "\n${GREEN}🎉 Setup Complete!${NC}"
echo -e "\n${BLUE}📋 NEXT STEPS:${NC}"
echo -e "1. ${YELLOW}Fund your wallet${NC} with 0.01 ETH"
echo -e "2. ${YELLOW}Test system${NC}: python3 -c 'from pipeline import main_pipeline; print(\"Pipeline ready\")'"
echo -e "3. ${YELLOW}Start trading${NC}: python3 pipeline.py"

echo -e "\n${BLUE}💰 TO FUND YOUR WALLET:${NC}"
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
wallet = os.getenv('WALLET_ADDRESS')
print(f'Send 0.01-0.02 ETH to: {wallet}')
print('(This address works on Arbitrum, Polygon, and Optimism)')
"

echo -e "\n${YELLOW}🔧 If you get import errors, run:${NC}"
echo -e "${BLUE}pip install torch xgboost prometheus-client matplotlib${NC}"