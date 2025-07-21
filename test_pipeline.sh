#!/bin/bash

# Simple DeFi Trading System Test - Debug Version
# Run this if the main test script fails

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== SIMPLE DEBUG TEST ===${NC}"
echo "Current directory: $(pwd)"
echo "Files present:"
ls -la

echo -e "\n${BLUE}=== PYTHON VERSION ===${NC}"
python3 --version || {
    echo -e "${RED}Python3 not found${NC}"
    exit 1
}

echo -e "\n${BLUE}=== BASIC IMPORTS ===${NC}"
python3 -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python path: {sys.path[0]}')

try:
    import pandas as pd
    print('✓ pandas imported')
except ImportError as e:
    print(f'✗ pandas: {e}')

try:
    import numpy as np
    print('✓ numpy imported')
except ImportError as e:
    print(f'✗ numpy: {e}')

try:
    import yaml
    print('✓ yaml imported')
except ImportError as e:
    print(f'✗ yaml: {e}')

try:
    import web3
    print('✓ web3 imported')
except ImportError as e:
    print(f'✗ web3: {e}')

try:
    import aiohttp
    print('✓ aiohttp imported')
except ImportError as e:
    print(f'✗ aiohttp: {e}')

try:
    import fastapi
    print('✓ fastapi imported')
except ImportError as e:
    print(f'✗ fastapi: {e}')

print('Basic import test completed')
"

echo -e "\n${BLUE}=== SETTINGS FILE ===${NC}"
if [[ -f "settings.yaml" ]]; then
    echo "✓ settings.yaml exists"
    python3 -c "
import yaml
try:
    with open('settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('✓ YAML syntax valid')
    print(f'✓ Config sections: {list(config.keys())}')
except Exception as e:
    print(f'✗ YAML error: {e}')
"
else
    echo "✗ settings.yaml missing"
    echo "Creating basic settings.yaml..."
    cat > settings.yaml << 'EOF'
redis:
  host: localhost
  port: 6379

trading:
  base_position_size: 0.001
  momentum_threshold: 0.09

risk:
  max_position_size: 0.01
  min_position_size: 0.0001

ml:
  retrain_threshold: 1000
  prediction_confidence: 0.75
EOF
    echo "✓ Basic settings.yaml created"
fi

echo -e "\n${BLUE}=== COMPONENT TESTS ===${NC}"

# Test individual components
components=("abi" "settings.yaml")

echo "Testing abi.py..."
if [[ -f "abi.py" ]]; then
    python3 -c "
try:
    import abi
    print('✓ abi.py imported successfully')
    if hasattr(abi, 'ERC20_ABI'):
        print('✓ ERC20_ABI found')
    if hasattr(abi, 'UNISWAP_V3_POOL_ABI'):
        print('✓ UNISWAP_V3_POOL_ABI found')
except Exception as e:
    print(f'✗ abi.py error: {e}')
"
else
    echo "✗ abi.py missing"
fi

echo -e "\nTesting signal_detector.py..."
if [[ -f "signal_detector.py" ]]; then
    python3 -c "
try:
    from signal_detector import SignalDetector
    print('✓ SignalDetector imported successfully')
except Exception as e:
    print(f'✗ SignalDetector error: {e}')
"
else
    echo "✗ signal_detector.py missing"
fi

echo -e "\nTesting inference_model.py..."
if [[ -f "inference_model.py" ]]; then
    python3 -c "
try:
    from inference_model import MomentumEnsemble
    print('✓ MomentumEnsemble imported successfully')
except Exception as e:
    print(f'✗ MomentumEnsemble error: {e}')
"
else
    echo "✗ inference_model.py missing"
fi

echo -e "\n${BLUE}=== DATABASE TEST ===${NC}"
python3 -c "
import sqlite3
try:
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE test (id INTEGER, value TEXT)')
    cursor.execute('INSERT INTO test VALUES (1, \"hello\")')
    result = cursor.fetchone()
    conn.close()
    print('✓ SQLite working')
except Exception as e:
    print(f'✗ SQLite error: {e}')
"

echo -e "\n${BLUE}=== ENVIRONMENT VARIABLES ===${NC}"
echo "ARBITRUM_RPC_URL: ${ARBITRUM_RPC_URL:-'Not set'}"
echo "POLYGON_RPC_URL: ${POLYGON_RPC_URL:-'Not set'}"
echo "OPTIMISM_RPC_URL: ${OPTIMISM_RPC_URL:-'Not set'}"
echo "PRIVATE_KEY: ${PRIVATE_KEY:+Set (hidden)}${PRIVATE_KEY:-'Not set'}"

echo -e "\n${BLUE}=== REDIS TEST ===${NC}"
python3 -c "
try:
    import redis
    r = redis.Redis(host='localhost', port=6379, socket_timeout=3, decode_responses=False)
    r.ping()
    print('✓ Redis connection successful')
except redis.ConnectionError:
    print('✗ Redis not running (install: apt-get install redis-server)')
except ImportError:
    print('✗ Redis module not installed (pip install redis)')
except Exception as e:
    print(f'✗ Redis error: {e}')
"

echo -e "\n${GREEN}=== SIMPLE TEST COMPLETED ===${NC}"
echo -e "${YELLOW}If this test passes, try running the full test with:${NC}"
echo -e "${YELLOW}./test_pipeline.sh${NC}"

echo -e "\n${YELLOW}To install missing dependencies:${NC}"
echo "pip install pandas numpy web3 pyyaml aiohttp fastapi redis"
echo "pip install torch tensorflow  # For ML components"

echo -e "\n${YELLOW}To start Redis (if needed):${NC}"
echo "# Ubuntu/Debian:"
echo "sudo apt-get install redis-server"
echo "sudo service redis-server start"
echo ""
echo "# macOS:"
echo "brew install redis"
echo "brew services start redis"