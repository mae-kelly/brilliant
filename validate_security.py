#!/usr/bin/env python3
"""
Security validation script to ensure no hardcoded addresses
"""

import os
import re
import sys

def validate_no_hardcoded_addresses():
    """Validate that no hardcoded addresses exist in codebase"""
    print("🔒 Running security validation...")
    
    issues = []
    
    # Patterns to detect
    patterns = {
        'ethereum_address': r'0x[a-fA-F0-9]{40}',
        'private_key': r'0x[a-fA-F0-9]{64}',
        'api_key_pattern': r'[a-zA-Z0-9]{32,}',
    }
    
    # Files to check
    files_to_check = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith(('.py', '.js', '.json', '.yaml', '.yml', '.sh')):
                files_to_check.append(os.path.join(root, file))
    
    for filepath in files_to_check:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for Ethereum addresses
            eth_addresses = re.findall(patterns['ethereum_address'], content)
            for addr in eth_addresses:
                # Allow certain safe patterns
                if addr in ['0x' + '0' * 40, '0x' + 'f' * 40, '0x' + 'a' * 40]:
                    continue
                
                # Check if it's properly using environment variables
                context_start = max(0, content.find(addr) - 50)
                context_end = min(len(content), content.find(addr) + 50)
                context = content[context_start:context_end]
                
                if 'os.getenv' not in context and 'getenv' not in context:
                    issues.append(f"{filepath}: Hardcoded address {addr}")
            
            # Check for private keys
            private_keys = re.findall(patterns['private_key'], content)
            for key in private_keys:
                if key != '0x' + '0' * 64:  # Allow zero key
                    context_start = max(0, content.find(key) - 50)
                    context_end = min(len(content), content.find(key) + 50)
                    context = content[context_start:context_end]
                    
                    if 'os.getenv' not in context and 'getenv' not in context:
                        issues.append(f"{filepath}: Hardcoded private key")
            
        except Exception as e:
            print(f"⚠️ Could not read {filepath}: {e}")
    
    if issues:
        print("❌ SECURITY ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Security validation passed - no hardcoded addresses found")
        return True

if __name__ == "__main__":
    success = validate_no_hardcoded_addresses()
    sys.exit(0 if success else 1)
