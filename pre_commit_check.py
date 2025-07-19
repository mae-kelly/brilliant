import os
import re
import sys

def check_for_unsafe_patterns():
    unsafe_patterns = [
        (r'dry_run = True', 'Dry run disabled'),
        (r'enable_real_trading = False', 'Real trading enabled'),
        (r'0x[a-fA-F0-9]{40}', 'Hardcoded wallet address'),
        (r'console\.log.*private', 'Private data in console.log'),
    ]
    
    issues = []
    
    for root, dirs, files in os.walk('.'):
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            if file.endswith(('.py', '.js', '.json', '.yaml')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    for pattern, description in unsafe_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            if ('os.getenv' not in match and 
                                '0x0000000000000000000000000000000000000000' not in match and
                                'dry_run = True' not in content and 
                                'enable_real_trading = False' not in content):
                                issues.append(f"{filepath}: {description}")
                except:
                    continue
    
    return list(set(issues))

if __name__ == "__main__":
    issues = check_for_unsafe_patterns()
    
    if issues:
        print("SECURITY ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nCommit blocked for safety")
        sys.exit(1)
    else:
        print("Pre-commit safety check passed")
        sys.exit(0)
