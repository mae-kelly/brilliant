#!/bin/bash

echo "🚀 DEPLOYING COMPLETE RENAISSANCE DEFI TRADING SYSTEM"
echo "====================================================="

# Create all necessary directories
echo "📁 Creating directory structure..."
mkdir -p {scanners,executors,analyzers,profilers,watchers,models,data,monitoring,config,logs,cache}

# Set up Python environment
echo "🐍 Setting up Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

# Install requirements
echo "📦 Installing requirements..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements_final.txt

# Generate synthetic training data
echo "🧠 Generating ML training data..."
if [ -f "synthetic_training_data.py" ]; then
    $PYTHON_CMD synthetic_training_data.py
else
    echo "⚠️ Synthetic training data generator not found"
fi

# Set permissions
echo "🔧 Setting permissions..."
chmod +x *.sh
chmod +x run_production_system.py

# Create environment template
echo "⚙️ Creating environment template..."
cat > .env.template << 'EOL'
# API Keys (replace with your actual keys)
ALCHEMY_API_KEY=your_alchemy_api_key_here
INFURA_API_KEY=your_infura_api_key_here
ETHERSCAN_API_KEY=your_etherscan_api_key_here

# Wallet Configuration (NEVER commit real values)
WALLET_ADDRESS=0x0000000000000000000000000000000000000000
PRIVATE_KEY=0x0000000000000000000000000000000000000000000000000000000000000000

# Trading Configuration
ENABLE_REAL_TRADING=false
DRY_RUN=true
MAX_POSITION_USD=10.0
MAX_DAILY_LOSS_USD=50.0
EOL

# Verify system components
echo "🔍 Verifying system components..."

COMPONENTS=(
    "production_renaissance_system.py:🎪 Main Production System"
    "run_production_system.py:🚀 CLI Runner"
    "run_pipeline.ipynb:📓 Jupyter Orchestrator"
    "config/dynamic_parameters.py:⚙️ Dynamic Configuration"
    "scanners/enhanced_ultra_scanner.py:🔍 Ultra-Scale Scanner"
    "executors/position_manager.py:💼 Position Management"
    "models/online_learner.py:🧠 Online Learning"
    "data/async_token_cache.py:💾 Async Database"
)

missing_components=()
for component in "${COMPONENTS[@]}"; do
    file="${component%%:*}"
    name="${component##*:}"
    
    if [ -f "$file" ]; then
        echo "✅ $name"
    else
        echo "❌ $name - MISSING: $file"
        missing_components+=("$file")
    fi
done

echo ""
if [ ${#missing_components[@]} -eq 0 ]; then
    echo "🎉 ALL COMPONENTS VERIFIED - SYSTEM READY!"
    echo ""
    echo "🚀 QUICK START OPTIONS:"
    echo "======================="
    echo ""
    echo "1. 📓 Jupyter Notebook (Recommended):"
    echo "   jupyter notebook run_pipeline.ipynb"
    echo ""
    echo "2. 🖥️  Command Line:"
    echo "   python run_production_system.py --duration 0.5"
    echo ""
    echo "3. 🔧 Custom Configuration:"
    echo "   python run_production_system.py --duration 24 --target 15000"
    echo ""
    echo "4. 🛠️  Setup for Real Trading:"
    echo "   cp .env.template .env"
    echo "   # Edit .env with your API keys"
    echo "   export ENABLE_REAL_TRADING=true"
    echo ""
    echo "🎯 TARGET: 10,000+ tokens/day autonomous scanning"
    echo "💰 STARTING CAPITAL: $10.00"
    echo "🤖 MODE: Fully autonomous AI-driven trading"
    echo ""
    echo "🏆 RENAISSANCE-LEVEL SYSTEM DEPLOYED SUCCESSFULLY!"
else
    echo "⚠️ DEPLOYMENT INCOMPLETE"
    echo "Missing ${#missing_components[@]} critical components:"
    for component in "${missing_components[@]}"; do
        echo "   - $component"
    done
    echo ""
    echo "Run the individual setup scripts to generate missing components"
fi

echo "====================================================="
