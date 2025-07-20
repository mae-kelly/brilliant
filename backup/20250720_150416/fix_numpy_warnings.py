#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
except ImportError:
    def get_dynamic_config(): return {"volatility_threshold": 0.1, "confidence_threshold": 0.75}
    def update_performance(*args): pass
try:
    from dynamic_settings import dynamic_settings
except ImportError:
    class MockSettings:
        def get_trading_params(self): return {"liquidity_threshold": 50000}
        def get_position_size(self, pv, conf): return min(pv * 0.1, 1.0)
    dynamic_settings = MockSettings()
"""
Suppress NumPy compatibility warnings for Renaissance Trading System
Add this import at the top of your main files to suppress warnings
"""

import warnings
import numpy

# Suppress specific NumPy version warnings
warnings.filterwarnings('ignore', message='A NumPy version.*is required')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')

# Alternative: Set NumPy to not show version warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

print("🔇 NumPy compatibility warnings suppressed")

# Test that everything works
try:
    import numpy as np
    import scipy
    import pandas as pd
    print(f"✅ NumPy {np.__version__} - Working silently")
    print(f"✅ SciPy {scipy.__version__} - No warnings")
    print(f"✅ Pandas {pd.__version__} - Ready for trading")
except Exception as e:
    print(f"❌ Error: {e}")
