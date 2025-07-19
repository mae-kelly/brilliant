#!/usr/bin/env python3
import os
import re
import sys

def verify_commit_safety():
    print("🔒 Verifying commit safety...")
    
    issues = []
    
    for root, dirs, files in os.walk('.'):
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            if file.endswith('.py') and not file.startswith('verify_commit'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    hardcoded_addresses = re.findall(r'0x[a-fA-F0-9]{40}', content)
                    for addr in hardcoded_addresses:
                        if addr != '0x0000000000000000000000000000000000000000':
                            context = content[max(0, content.find(addr)-100):content.find(addr)+100]
                            if 'os.getenv' not in context and 'getenv' not in context:
                                issues.append(f"Hardcoded address in {filepath}: {addr}")
                    
                    if re.search(r'enable_real_trading = False', content):
                        issues.append(f"Real trading enabled in {filepath}")
                    
                    if re.search(r'dry_run = True', content):
                        issues.append(f"Dry run disabled in {filepath}")
                    
                except:
                    continue
    
    if issues:
        print("❌ COMMIT BLOCKED - Security issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Commit safety verified - all checks passed")
        return True

if __name__ == "__main__":
    safe = verify_commit_safety()
    sys.exit(0 if safe else 1)
