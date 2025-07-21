#!/usr/bin/env python3
"""
Update import paths after reorganization
"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path):
    """Update import statements in a Python file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Update common import patterns
        updates = {
            r'from intelligence.signals.signal_detector import': 'from intelligence.signals.signal_detector import',
            r'from core.execution.trade_executor import': 'from core.execution.trade_executor import',
            r'from core.models.inference_model import': 'from core.models.inference_model import',
            r'from core.execution.risk_manager import': 'from core.execution.risk_manager import',
            r'from security.validators.safety_checks import': 'from security.validators.safety_checks import',
            r'from security.rugpull.anti_rug_analyzer import': 'from security.rugpull.anti_rug_analyzer import',
            r'from security.mempool.mempool_watcher import': 'from security.mempool.mempool_watcher import',
            r'from core.engine.batch_processor import': 'from core.engine.batch_processor import',
            r'from core.features.vectorized_features import': 'from core.features.vectorized_features import',
        }
        
        for old_pattern, new_import in updates.items():
            content = re.sub(old_pattern, new_import, content)
        
        with open(file_path, 'w') as f:
            f.write(content)
            
        return True
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all Python files"""
    updated_count = 0
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports_in_file(file_path):
                    updated_count += 1
    
    print(f"✅ Updated imports in {updated_count} files")

if __name__ == "__main__":
    main()
