#!/usr/bin/env python3
"""
Security check for WebSocket scanner files
"""

import os
import re

def check_for_hardcoded_addresses():
    """Check for hardcoded wallet addresses"""
    issues = []
    
    files_to_check = [
        'websocket_scanner_secure.py',
        'websocket_config_secure.py',
        'websocket_scanner_working.py'
    ]
    
    # Pattern for Ethereum addresses (but allow safe defaults)
    address_pattern = r'0x[a-fA-F0-9]{40}'
    safe_addresses = [
        '',  # Safe default
        ''   # Template placeholder
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
                
            # Find all addresses
            addresses = re.findall(address_pattern, content)
            
            for addr in addresses:
                if addr not in safe_addresses:
                    # Check if it's wrapped in os.getenv
                    context_start = max(0, content.find(addr) - 50)
                    context_end = min(len(content), content.find(addr) + 100)
                    context = content[context_start:context_end]
                    
                    if 'os.getenv' not in context and 'getenv' not in context:
                        issues.append(f"{filename}: Hardcoded address {addr}")
    
    return issues

if __name__ == "__main__":
    issues = check_for_hardcoded_addresses()
    
    if issues:
        print("❌ SECURITY ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        exit(1)
    else:
        print("✅ No hardcoded addresses found - security check passed!")
        exit(0)
