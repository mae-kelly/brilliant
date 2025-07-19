#!/bin/bash

echo "🚀 RENAISSANCE WEBSOCKET SCANNER INSTALLATION"
echo "============================================="

# Check if running in proper environment
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Run this script from your trading bot directory"
    exit 1
fi

# Backup existing scanner files
echo "💾 Backing up existing scanner files..."
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
cp scanner_v*.py backups/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true

# Install high-performance dependencies
echo "📦 Installing WebSocket dependencies..."
pip install websockets>=11.0.0 aiohttp>=3.8.6 uvloop>=0.17.0 aiodns>=3.0.0
pip install aioredis>=2.0.1 prometheus-client>=0.17.0 psutil>=5.9.0

# Make scripts executable
echo "🔧 Setting up execution permissions..."
chmod +x websocket_scanner_v5.py
chmod +x integrate_websocket_scanner.py
chmod +x test_websocket_scanner.py
chmod +x update_pipeline_for_websockets.py

# Create logs directory
mkdir -p logs/websocket_scanner

# Create Redis configuration (optional)
if command -v redis-server &> /dev/null; then
    echo "✅ Redis found - WebSocket scanner will use distributed caching"
else
    echo "⚠️  Redis not found - WebSocket scanner will use local caching"
    echo "   Install Redis for better performance: sudo apt-get install redis-server"
fi

# Environment variable check
echo "🔐 Checking environment variables..."
if [ -z "$ALCHEMY_API_KEY" ] && [ -z "$INFURA_API_KEY" ]; then
    echo "⚠️  Warning: Set ALCHEMY_API_KEY or INFURA_API_KEY for WebSocket connections"
    echo "   Add to your .env file: ALCHEMY_API_KEY=your_key_here"
fi

# Test installation
echo "🧪 Testing WebSocket scanner installation..."
if python3 -c "import websockets, aiohttp, uvloop; print('✅ All dependencies available')" 2>/dev/null; then
    echo "✅ Installation successful!"
else
    echo "❌ Installation issues detected. Please check dependencies."
    exit 1
fi

echo ""
echo "🎉 WEBSOCKET SCANNER INSTALLATION COMPLETE!"
echo ""
echo "Next steps:"
echo "1. Set environment variables in .env:"
echo "   ALCHEMY_API_KEY=your_alchemy_key"
echo "   INFURA_API_KEY=your_infura_key"
echo ""
echo "2. Test the scanner:"
echo "   python3 test_websocket_scanner.py"
echo ""
echo "3. Update your pipeline:"
echo "   python3 update_pipeline_for_websockets.py"
echo ""
echo "4. Start real-time scanning:"
echo "   python3 integrate_websocket_scanner.py"
echo ""
echo "🚀 Your trading system now has Renaissance-level real-time scanning!"

