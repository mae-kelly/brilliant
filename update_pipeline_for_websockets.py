#!/usr/bin/env python3
"""
Update existing pipeline to use WebSocket scanner
"""

import os
import sys

def update_scanner_imports():
    """Update all files that import the old scanner"""
    
    files_to_update = [
        'run_pipeline.ipynb',
        'launch_system.py',
        'start_bot.py'
    ]
    
    for filename in files_to_update:
        if os.path.exists(filename):
            print(f"📝 Updating {filename}...")
            
            with open(filename, 'r') as f:
                content = f.read()
            
            # Replace old scanner imports
            content = content.replace(
                'from scanner_v3 import TokenScanner',
                'from integrate_websocket_scanner import scanner_integration'
            )
            content = content.replace(
                'from scanner_v4 import UltraScaleScanner',
                'from integrate_websocket_scanner import scanner_integration'
            )
            content = content.replace(
                'scanner = TokenScanner()',
                'scanner = scanner_integration'
            )
            content = content.replace(
                'scanner = UltraScaleScanner()',
                'scanner = scanner_integration'
            )
            content = content.replace(
                'scanner.scan()',
                'await scanner.scan_for_momentum()'
            )
            content = content.replace(
                'scanner.scan_10k_tokens_parallel()',
                'scanner.scan_for_momentum()'
            )
            
            with open(filename, 'w') as f:
                f.write(content)
                
            print(f"✅ Updated {filename}")

def create_websocket_requirements():
    """Create requirements file for WebSocket dependencies"""
    
    requirements = """
# WebSocket Scanner Dependencies
websockets>=11.0.0
aiohttp>=3.8.6
asyncio-mqtt>=0.11.1
uvloop>=0.17.0
aiodns>=3.0.0
aioredis>=2.0.1

# Existing dependencies
web3>=6.11.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
scikit-learn>=1.3.0
eth-abi>=4.2.1

# Performance monitoring
prometheus-client>=0.17.0
psutil>=5.9.0
"""

    with open('requirements_websocket_upgrade.txt', 'w') as f:
        f.write(requirements)
        
    print("📦 Created requirements_websocket_upgrade.txt")

def create_installation_script():
    """Create installation script for WebSocket upgrade"""
    
    script = """#!/bin/bash

echo "🚀 Installing WebSocket Scanner Upgrade..."

# Install new dependencies
pip install -r requirements_websocket_upgrade.txt

# Backup old scanner files
mkdir -p backups/old_scanners
cp scanner_v*.py backups/old_scanners/ 2>/dev/null || true

# Make new files executable
chmod +x websocket_scanner_v5.py
chmod +x integrate_websocket_scanner.py
chmod +x update_pipeline_for_websockets.py

echo "✅ WebSocket scanner upgrade installed!"
echo ""
echo "Next steps:"
echo "1. Test the new scanner: python integrate_websocket_scanner.py"
echo "2. Update your pipeline: python update_pipeline_for_websockets.py"
echo "3. Run your trading system with real-time WebSocket scanning!"
"""

    with open('install_websocket_upgrade.sh', 'w') as f:
        f.write(script)
        
    os.chmod('install_websocket_upgrade.sh', 0o755)
    print("📋 Created install_websocket_upgrade.sh")

if __name__ == "__main__":
    print("🔧 Updating pipeline for WebSocket integration...")
    
    update_scanner_imports()
    create_websocket_requirements()
    create_installation_script()
    
    print("\n✅ Pipeline update complete!")
    print("\nTo activate WebSocket scanning:")
    print("1. Run: ./install_websocket_upgrade.sh")
    print("2. Update environment with your API keys")
    print("3. Test: python integrate_websocket_scanner.py")
