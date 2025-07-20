#!/usr/bin/env python3
import os
import re
import glob

def replace_in_file(filepath, replacements):
    """Replace hardcoded values in file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Updated {filepath}")
        
    except Exception as e:
        print(f"❌ Error updating {filepath}: {e}")

# Define replacements
replacements = {
    r'confidence_threshold\s*=\s*[\d.]+': 'confidence_threshold = get_dynamic_config()["confidence_threshold"]',
    r'min_momentum_score\s*=\s*[\d.]+': 'min_momentum_score = get_dynamic_config()["min_momentum_score"]',
    r'position_size\s*=\s*[\d.]+': 'position_size = base_position * get_dynamic_config()["position_size_multiplier"]',
    r'max_hold_time\s*=\s*\d+': 'max_hold_time = int(get_dynamic_config()["max_hold_time"])',
    r'slippage_tolerance\s*=\s*[\d.]+': 'slippage_tolerance = get_dynamic_config()["slippage_tolerance"]',
    r'stop_loss_threshold\s*=\s*[\d.]+': 'stop_loss_threshold = get_dynamic_config()["stop_loss_threshold"]',
    r'take_profit_threshold\s*=\s*[\d.]+': 'take_profit_threshold = get_dynamic_config()["take_profit_threshold"]',
    r'volatility_threshold\s*=\s*[\d.]+': 'volatility_threshold = get_dynamic_config()["volatility_threshold"]',
}

# Add import statement
import_statement = """
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance
"""

# Find all Python files
python_files = glob.glob("**/*.py", recursive=True)

for filepath in python_files:
    if 'dynamic_parameters.py' in filepath:
        continue
        
    # Add import and replace values
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Add import if not present
        if 'get_dynamic_config' not in content:
            content = import_statement + "\n" + content
        
        # Apply replacements
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content)
        
        with open(filepath, 'w') as f:
            f.write(content)
            
        print(f"✅ Processed {filepath}")
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")

print("🎯 Hardcoded value replacement complete!")
