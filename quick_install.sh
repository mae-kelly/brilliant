#!/bin/bash

echo "=== QUICK INSTALL MISSING DEPENDENCIES ==="

pip3 install web3 requests python-dotenv cryptography flask numpy pandas scikit-learn

echo ""
echo "Checking installation..."
python3 -c "
try:
    import web3, requests, dotenv, cryptography, flask, numpy, pandas, sklearn
    print('✅ All core dependencies installed successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('Try: pip3 install --upgrade pip && pip3 install -r requirements.txt')
"

echo ""
echo "Run ./verify_setup.sh to check complete setup"
