#!/usr/bin/env python3
"""
Renaissance Trading System - Setup Validation
Validates that all components are properly installed and configured
"""

import os
import sys
import importlib
import json
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Requires 3.8+)")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    
    required_packages = [
        'web3', 'aiohttp', 'websockets', 'tensorflow', 
        'sklearn', 'pandas', 'numpy', 'plotly', 'fastapi'
    ]
    
    print("📦 Checking required packages...")
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (Missing)")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_gpu_availability():
    """Check GPU availability"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU Available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        else:
            print("⚠️ No GPU detected (CPU mode)")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def check_project_structure():
    """Check project directory structure"""
    
    required_dirs = ['models', 'cache', 'logs', 'config', 'data']
    required_files = ['run_pipeline.ipynb', 'install_dependencies.py']
    
    print("📁 Checking project structure...")
    
    missing_items = []
    
    # Check directories
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}/")
        else:
            print(f"   ❌ {directory}/ (Missing)")
            missing_items.append(directory)
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (Missing)")
            missing_items.append(file)
    
    return len(missing_items) == 0, missing_items

def check_configuration():
    """Check configuration files"""
    
    print("🔧 Checking configuration...")
    
    config_issues = []
    
    # Check for .env file
    if os.path.exists('.env'):
        print("   ✅ .env file found")
        
        # Check environment variables
        required_env_vars = ['ALCHEMY_API_KEY', 'WALLET_ADDRESS', 'ENABLE_REAL_TRADING']
        for var in required_env_vars:
            if var in os.environ:
                print(f"   ✅ {var} set")
            else:
                print(f"   ⚠️ {var} not set")
                config_issues.append(var)
    else:
        print("   ⚠️ .env file not found (will be created during setup)")
        config_issues.append('.env')
    
    return len(config_issues) == 0, config_issues

def generate_validation_report():
    """Generate comprehensive validation report"""
    
    print("🔍 Renaissance Trading System - Validation Report")
    print("=" * 60)
    
    checks = []
    
    # Python version
    python_ok = check_python_version()
    checks.append(('Python Version', python_ok))
    
    print()
    
    # Required packages
    packages_ok, missing_packages = check_required_packages()
    checks.append(('Required Packages', packages_ok))
    
    print()
    
    # GPU availability
    gpu_ok = check_gpu_availability()
    checks.append(('GPU Acceleration', gpu_ok))
    
    print()
    
    # Project structure
    structure_ok, missing_items = check_project_structure()
    checks.append(('Project Structure', structure_ok))
    
    print()
    
    # Configuration
    config_ok, config_issues = check_configuration()
    checks.append(('Configuration', config_ok))
    
    print()
    print("=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")
    
    overall_ready = all(status for _, status in checks)
    
    print()
    if overall_ready:
        print("🎉 SYSTEM FULLY VALIDATED - READY FOR TRADING!")
        print("🚀 You can now run the trading pipeline")
    else:
        print("⚠️ SYSTEM VALIDATION INCOMPLETE")
        print("🔧 Please address the issues above before running")
        
        if not packages_ok:
            print(f"   📦 Install missing packages: {missing_packages}")
        if not structure_ok:
            print(f"   📁 Create missing items: {missing_items}")
        if not config_ok:
            print(f"   🔧 Configure missing settings: {config_issues}")
    
    print("=" * 60)
    return overall_ready

if __name__ == "__main__":
    ready = generate_validation_report()
    sys.exit(0 if ready else 1)
