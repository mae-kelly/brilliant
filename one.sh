#!/bin/bash

# =============================================================================
# 📦 RENAISSANCE TRADING SYSTEM - DEPENDENCY INSTALLER
# =============================================================================
# Installs all required dependencies for the Renaissance trading system

echo "📦 RENAISSANCE TRADING SYSTEM - DEPENDENCY INSTALLER"
echo "===================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO") echo -e "${BLUE}ℹ️  $message${NC}" ;;
        "SUCCESS") echo -e "${GREEN}✅ $message${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "ERROR") echo -e "${RED}❌ $message${NC}" ;;
    esac
}

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PIP_CMD=pip3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
    PIP_CMD=pip
else
    print_status "ERROR" "No Python installation found!"
    exit 1
fi

print_status "INFO" "Using Python: $PYTHON_CMD"
print_status "INFO" "Using pip: $PIP_CMD"

echo ""
print_status "INFO" "🔧 Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip

echo ""
print_status "INFO" "📊 Installing core data science packages..."

# Core data packages (most critical)
CORE_PACKAGES=(
    "numpy>=1.24.0"
    "pandas>=2.0.0" 
    "scipy>=1.11.0"
    "scikit-learn>=1.3.0"
)

for package in "${CORE_PACKAGES[@]}"; do
    print_status "INFO" "Installing $package..."
    if $PIP_CMD install "$package"; then
        print_status "SUCCESS" "Installed $package"
    else
        print_status "ERROR" "Failed to install $package"
    fi
done

echo ""
print_status "INFO" "🌐 Installing networking packages..."

# Networking packages
NETWORK_PACKAGES=(
    "aiohttp>=3.8.0"
    "websockets>=11.0.0"
    "requests>=2.31.0"
    "aiofiles>=23.0.0"
)

for package in "${NETWORK_PACKAGES[@]}"; do
    print_status "INFO" "Installing $package..."
    if $PIP_CMD install "$package"; then
        print_status "SUCCESS" "Installed $package"
    else
        print_status "ERROR" "Failed to install $package"
    fi
done

echo ""
print_status "INFO" "🔗 Installing blockchain packages..."

# Blockchain packages
BLOCKCHAIN_PACKAGES=(
    "web3>=6.0.0"
    "eth-account>=0.9.0"
    "eth-abi>=4.0.0"
)

for package in "${BLOCKCHAIN_PACKAGES[@]}"; do
    print_status "INFO" "Installing $package..."
    if $PIP_CMD install "$package"; then
        print_status "SUCCESS" "Installed $package"
    else
        print_status "ERROR" "Failed to install $package"
    fi
done

echo ""
print_status "INFO" "🧠 Installing ML packages..."

# ML packages (optional, might fail on some systems)
ML_PACKAGES=(
    "tensorflow>=2.13.0"
    "joblib>=1.3.0"
)

for package in "${ML_PACKAGES[@]}"; do
    print_status "INFO" "Installing $package..."
    if $PIP_CMD install "$package"; then
        print_status "SUCCESS" "Installed $package"
    else
        print_status "WARNING" "Failed to install $package (optional)"
    fi
done

echo ""
print_status "INFO" "⚙️ Installing utility packages..."

# Utility packages
UTILITY_PACKAGES=(
    "python-dotenv>=1.0.0"
    "psutil>=5.9.0"
    "PyYAML>=6.0.0"
    "fastapi>=0.100.0"
    "uvicorn>=0.20.0"
)

for package in "${UTILITY_PACKAGES[@]}"; do
    print_status "INFO" "Installing $package..."
    if $PIP_CMD install "$package"; then
        print_status "SUCCESS" "Installed $package"
    else
        print_status "WARNING" "Failed to install $package"
    fi
done

echo ""
print_status "INFO" "🧪 Testing critical imports..."

# Test critical imports
CRITICAL_IMPORTS=(
    "numpy:NumPy"
    "pandas:Pandas"
    "aiohttp:Async HTTP"
    "web3:Web3"
    "asyncio:Asyncio"
)

failed_imports=0
for import_test in "${CRITICAL_IMPORTS[@]}"; do
    IFS=':' read -r module name <<< "$import_test"
    if $PYTHON_CMD -c "import $module" 2>/dev/null; then
        print_status "SUCCESS" "$name import test passed"
    else
        print_status "ERROR" "$name import test failed"
        ((failed_imports++))
    fi
done

echo ""
print_status "INFO" "📋 Installation Summary"
echo "======================="

if [[ $failed_imports -eq 0 ]]; then
    print_status "SUCCESS" "🎉 ALL CRITICAL DEPENDENCIES INSTALLED!"
    echo ""
    echo "✅ Your system is ready for Renaissance trading!"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Run: ./one.sh (to recheck sync)"
    echo "   2. If sync passes: python production_renaissance_system.py"
    echo "   3. Or use Jupyter: jupyter notebook run_pipeline.ipynb"
    
elif [[ $failed_imports -le 2 ]]; then
    print_status "WARNING" "🔧 MOSTLY READY - $failed_imports minor issues"
    echo ""
    echo "⚠️  Some optional packages failed, but core system should work"
    echo ""
    echo "🚀 Try running:"
    echo "   ./one.sh"
    
else
    print_status "ERROR" "🛠️ DEPENDENCY ISSUES - $failed_imports critical failures"
    echo ""
    echo "❌ Multiple critical packages failed to install"
    echo ""
    echo "🔧 Try these fixes:"
    echo "   1. Update your Python: brew install python3"
    echo "   2. Use virtual environment: python3 -m venv venv && source venv/bin/activate"
    echo "   3. Try conda instead: conda install pandas numpy scipy"
    echo "   4. Check your system: $PYTHON_CMD --version"
fi

echo ""
print_status "INFO" "🔍 Python Environment Info:"
echo "   Python: $($PYTHON_CMD --version)"
echo "   Pip: $($PIP_CMD --version)"
echo "   Platform: $(uname -s)"

# Create requirements verification script
cat > check_requirements.py << 'EOF'
#!/usr/bin/env python3
"""
Quick requirements checker for Renaissance Trading System
"""

import sys

REQUIRED_PACKAGES = [
    'numpy', 'pandas', 'aiohttp', 'web3', 'asyncio',
    'requests', 'json', 'time', 'os', 'threading'
]

OPTIONAL_PACKAGES = [
    'tensorflow', 'scikit-learn', 'scipy', 'fastapi'
]

def check_package(package_name, required=True):
    try:
        __import__(package_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        if required:
            print(f"❌ {package_name} (REQUIRED)")
        else:
            print(f"⚠️  {package_name} (optional)")
        return False

def main():
    print("🔍 Renaissance Trading System - Requirements Check")
    print("=" * 50)
    
    print("\n📋 Required Packages:")
    required_failed = 0
    for package in REQUIRED_PACKAGES:
        if not check_package(package, required=True):
            required_failed += 1
    
    print("\n📦 Optional Packages:")
    for package in OPTIONAL_PACKAGES:
        check_package(package, required=False)
    
    print(f"\n📊 Summary:")
    if required_failed == 0:
        print("🎉 All required packages available!")
        print("🚀 System ready for trading!")
        sys.exit(0)
    else:
        print(f"❌ {required_failed} required packages missing")
        print("🔧 Run the dependency installer again")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x check_requirements.py

echo ""
print_status "SUCCESS" "Created check_requirements.py for future verification"
echo ""
print_status "INFO" "Run './check_requirements.py' anytime to verify your setup"