#!/usr/bin/env python3
"""
Renaissance Trading System - Dependency Installer
Installs all required packages for Colab and local environments
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install all required dependencies"""
    
    # Core packages
    core_packages = [
        "web3>=6.0.0",
        "aiohttp>=3.8.0", 
        "websockets>=11.0.0",
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "dash>=2.12.0",
        "streamlit>=1.25.0"
    ]
    
    # Monitoring and utilities
    monitoring_packages = [
        "prometheus-client>=0.17.0",
        "psutil>=5.9.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "asyncpg>=0.28.0",
        "aiosqlite>=0.19.0"
    ]
    
    # ML and data science
    ml_packages = [
        "joblib>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "ipywidgets>=8.0.0"
    ]
    
    all_packages = core_packages + monitoring_packages + ml_packages
    
    print("🚀 Installing Renaissance Trading System Dependencies...")
    print(f"📦 Total packages to install: {len(all_packages)}")
    print("=" * 60)
    
    failed_packages = []
    
    for package in all_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print("=" * 60)
    print(f"✅ Successfully installed: {len(all_packages) - len(failed_packages)}/{len(all_packages)} packages")
    
    if failed_packages:
        print(f"❌ Failed packages: {failed_packages}")
        return False
    else:
        print("🎉 All dependencies installed successfully!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
