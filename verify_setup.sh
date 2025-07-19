#!/bin/bash

echo "=== TRADING BOT SETUP VERIFICATION ==="

PASS=0
FAIL=0

check_file() {
    if [ -f "$1" ]; then
        echo "✅ $1 exists"
        PASS=$((PASS + 1))
    else
        echo "❌ $1 missing"
        FAIL=$((FAIL + 1))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "✅ Directory $1 exists"
        PASS=$((PASS + 1))
    else
        echo "❌ Directory $1 missing"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "Checking required files..."
check_file ".env.template"
check_file "requirements.txt"
check_file "secure_loader.py"
check_file "safe_operations.py"
check_file "dev_mode.py"
check_file "pre_commit_check.py"
check_file "test_framework.py"
check_file "start_bot.py"

echo ""
echo "Checking directories..."
check_dir "abi"
check_dir "logs"
check_dir "cache"
check_dir "models"

echo ""
echo "Checking ABI files..."
check_file "abi/erc20.json"
check_file "abi/uniswap_router.json"
check_file "abi/uniswap_factory.json"

echo ""
echo "Checking environment..."
if [ -f .env ]; then
    echo "✅ .env file exists"
    PASS=$((PASS + 1))
else
    echo "⚠️  .env file not found (copy from .env.template)"
fi

echo ""
echo "Checking Python modules..."
python3 -c "import web3; print('✅ web3 installed')" 2>/dev/null || echo "❌ web3 not installed"
python3 -c "import requests; print('✅ requests installed')" 2>/dev/null || echo "❌ requests not installed"
python3 -c "from dotenv import load_dotenv; print('✅ python-dotenv installed')" 2>/dev/null || echo "❌ python-dotenv not installed"

echo ""
echo "Running safety checks..."
if python3 pre_commit_check.py >/dev/null 2>&1; then
    echo "✅ Safety checks passed"
    PASS=$((PASS + 1))
else
    echo "⚠️  Safety checks found issues"
fi

echo ""
echo "Testing configuration..."
if python3 -c "from secure_loader import config; print('✅ Configuration module working')" 2>/dev/null; then
    PASS=$((PASS + 1))
else
    echo "⚠️  Configuration module has issues"
fi

echo ""
echo "=== VERIFICATION SUMMARY ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "🎉 Setup verification PASSED!"
    echo ""
    echo "Next steps:"
    echo "1. Configure .env file with your API keys"
    echo "2. Run: python3 start_bot.py"
    echo "3. Bot will start in SAFE MODE"
else
    echo ""
    echo "❌ Setup verification FAILED!"
    echo "Please fix the issues above before proceeding."
fi
