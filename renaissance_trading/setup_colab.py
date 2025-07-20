#!/usr/bin/env python3
"""
Renaissance Trading System - Google Colab Setup
Configures Colab environment for optimal performance
"""

import os
import sys
from pathlib import Path

def setup_colab_environment():
    """Setup Google Colab environment"""
    
    print("🔧 Setting up Google Colab environment...")
    
    # Check if in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✅ Google Colab detected")
    except ImportError:
        IN_COLAB = False
        print("📍 Local environment detected")
        return True
    
    if IN_COLAB:
        # Mount Google Drive
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✅ Google Drive mounted")
        except Exception as e:
            print(f"⚠️ Drive mount failed: {e}")
        
        # Set up persistent directory
        persistent_dir = '/content/drive/MyDrive/renaissance_trading'
        os.makedirs(persistent_dir, exist_ok=True)
        os.chdir(persistent_dir)
        print(f"📂 Working directory: {persistent_dir}")
        
        # Configure GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_GPU_MEMORY_GROWTH'] = 'true'
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
        print("🚀 GPU configuration set")
        
        # Install system packages
        os.system("apt-get update -qq")
        os.system("apt-get install -y -qq git wget curl")
        print("✅ System packages installed")
        
        return True
    
    return False

def create_project_structure():
    """Create project directory structure"""
    
    directories = [
        'models',
        'cache', 
        'logs',
        'config',
        'data',
        'scripts',
        'charts'
    ]
    
    print("📁 Creating project structure...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   📂 {directory}/")
    
    print("✅ Project structure created")

def main():
    """Main setup function"""
    
    print("🚀 Renaissance Trading System - Colab Setup")
    print("=" * 50)
    
    # Setup environment
    setup_colab_environment()
    
    # Create project structure
    create_project_structure()
    
    # Install dependencies
    print("📦 Installing dependencies...")
    os.system("python install_dependencies.py")
    
    print("=" * 50)
    print("🎉 Colab setup complete!")
    print("📋 Next steps:")
    print("   1. Open run_pipeline.ipynb")
    print("   2. Configure your trading parameters")
    print("   3. Run the autonomous trading system")

if __name__ == "__main__":
    main()
