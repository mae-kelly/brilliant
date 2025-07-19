#!/bin/bash

echo "🚀 INSTALLING OPTIMIZED WEBSOCKET SCANNER"
echo "========================================="

# Install dependencies without problematic packages
echo "📦 Installing core dependencies..."
pip install websockets aiohttp uvloop aiodns
pip install numpy scipy pandas scikit-learn
pip install web3 eth-abi eth-utils hexbytes
pip install prometheus-client psutil aioredis
pip install python-dotenv requests ujson

# Make scripts executable
chmod +x websocket_scanner_optimized.py
chmod +x test_optimized_scanner.py
chmod +x integrate_optimized_scanner.py

echo ""
echo "✅ INSTALLATION COMPLETE!"
echo ""
echo "🧪 To test:"
echo "   python3 test_optimized_scanner.py"
echo ""
echo "🚀 To run integration:"
echo "   python3 integrate_optimized_scanner.py"
echo ""
echo "🔧 To integrate with existing pipeline:"
echo "   Replace your scanner imports with:"
echo "   from integrate_optimized_scanner import scan_tokens"

