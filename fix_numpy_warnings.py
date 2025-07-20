#!/usr/bin/env python3
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
