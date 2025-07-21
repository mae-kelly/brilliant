#!/bin/bash

# DeFi Trading System Repository Reorganization Script
# Reorganizes existing files by functional requirements and removes redundancy

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔧 DeFi Trading Repo Reorganization${NC}"
echo -e "${BLUE}========================================${NC}"

# Backup original structure
echo -e "\n${YELLOW}💾 Creating backup...${NC}"
cp -r . ../defi_trading_backup_$(date +%Y%m%d_%H%M%S)
echo -e "${GREEN}✅ Backup created${NC}"

# Create organized directory structure
echo -e "\n${YELLOW}📁 Creating organized directory structure...${NC}"

mkdir -p core/engine
mkdir -p core/execution
mkdir -p core/models
mkdir -p core/features
mkdir -p intelligence/signals
mkdir -p intelligence/analysis
mkdir -p intelligence/sentiment
mkdir -p intelligence/graph
mkdir -p security/validators
mkdir -p security/rugpull
mkdir -p security/mempool
mkdir -p infrastructure/monitoring
mkdir -p infrastructure/config
mkdir -p infrastructure/deployment
mkdir -p data/cache
mkdir -p data/models
mkdir -p notebooks
mkdir -p scripts
mkdir -p tests

echo -e "${GREEN}✅ Directory structure created${NC}"

# 1. CORE ENGINE - Main pipeline and orchestration
echo -e "\n${YELLOW}🚀 Organizing Core Engine Files...${NC}"

mv pipeline.py core/engine/
mv batch_processor.py core/engine/
mv vectorized_features.py core/features/
mv model_manager.py core/models/
mv inference_model.py core/models/

echo -e "${GREEN}✅ Core engine files organized${NC}"

# 2. EXECUTION - Trading and risk management
echo -e "\n${YELLOW}💼 Organizing Execution Layer...${NC}"

mv trade_executor.py core/execution/
mv risk_manager.py core/execution/
mv scanner_v3.py core/execution/  # Scanner is part of execution pipeline

echo -e "${GREEN}✅ Execution layer organized${NC}"

# 3. INTELLIGENCE - Signals and analysis
echo -e "\n${YELLOW}🧠 Organizing Intelligence Layer...${NC}"

mv signal_detector.py intelligence/signals/
mv advanced_ensemble.py intelligence/analysis/
mv continuous_optimizer.py intelligence/analysis/
mv feedback_loop.py intelligence/analysis/

# Create sentiment analysis directory and move files
mkdir -p intelligence/sentiment
mv -f *sentiment* intelligence/sentiment/ 2>/dev/null || true

echo -e "${GREEN}✅ Intelligence layer organized${NC}"

# 4. SECURITY - Safety and protection systems
echo -e "\n${YELLOW}🛡️ Organizing Security Layer...${NC}"

mv safety_checks.py security/validators/
mv anti_rug_analyzer.py security/rugpull/
mv mempool_watcher.py security/mempool/
mv token_profiler.py security/validators/

echo -e "${GREEN}✅ Security layer organized${NC}"

# 5. INFRASTRUCTURE - Config, monitoring, deployment
echo -e "\n${YELLOW}🏗️ Organizing Infrastructure...${NC}"

mv settings.yaml infrastructure/config/
mv *.conf infrastructure/config/ 2>/dev/null || true
mv performance_optimizer.py infrastructure/monitoring/
mv logging_config.py infrastructure/monitoring/
mv error_handler.py infrastructure/monitoring/

# Docker and deployment files
mv docker-compose.yml infrastructure/deployment/
mv docker infrastructure/deployment/ 2>/dev/null || true
mv Dockerfile infrastructure/deployment/ 2>/dev/null || true

echo -e "${GREEN}✅ Infrastructure organized${NC}"

# 6. DATA - Cache, models, databases
echo -e "\n${YELLOW}💾 Organizing Data Layer...${NC}"

mv *.db data/cache/ 2>/dev/null || true
mv *.tflite data/models/ 2>/dev/null || true
mv *.pkl data/cache/ 2>/dev/null || true

echo -e "${GREEN}✅ Data layer organized${NC}"

# 7. NOTEBOOKS - Jupyter notebooks and analysis
echo -e "\n${YELLOW}📓 Organizing Notebooks...${NC}"

mv *.ipynb notebooks/ 2>/dev/null || true

echo -e "${GREEN}✅ Notebooks organized${NC}"

# 8. SCRIPTS - Setup and utility scripts
echo -e "\n${YELLOW}📜 Organizing Scripts...${NC}"

mv *.sh scripts/
mv setup_wallet.py scripts/
mv minimal_test.py scripts/
mv validate_system.py scripts/
mv final_benchmark.py scripts/

echo -e "${GREEN}✅ Scripts organized${NC}"

# 9. TESTS - Test files
echo -e "\n${YELLOW}🧪 Organizing Tests...${NC}"

mv backtesting.py tests/
mv *test* tests/ 2>/dev/null || true

echo -e "${GREEN}✅ Tests organized${NC}"

# 10. MERGE REDUNDANT FILES
echo -e "\n${YELLOW}🔄 Merging redundant files...${NC}"

# Merge similar API files
if [ -f "api_server.py" ]; then
    echo "# Additional API endpoints from api_server.py" >> core/execution/trade_executor.py
    cat api_server.py >> core/execution/trade_executor.py
    rm api_server.py
    echo -e "${GREEN}✅ Merged api_server.py into trade_executor.py${NC}"
fi

# Merge grafana dashboard files
if [ -f "grafana_dashboard.json" ]; then
    mv grafana_dashboard.json infrastructure/monitoring/
    rm -f docker/grafana-dashboard.json 2>/dev/null || true
    echo -e "${GREEN}✅ Consolidated Grafana dashboards${NC}"
fi

echo -e "${GREEN}✅ File merging completed${NC}"

# 11. DELETE REDUNDANT/EMPTY FILES
echo -e "\n${YELLOW}🗑️ Cleaning up redundant files...${NC}"

# Remove empty or placeholder files
find . -type f -empty -delete

# Remove duplicate ABI definitions (keep one)
if [ -f "abi.py" ]; then
    mv abi.py core/
    echo -e "${GREEN}✅ Moved abi.py to core/${NC}"
fi

echo -e "${GREEN}✅ Cleanup completed${NC}"

# 12. CREATE NEW CONSOLIDATED FILES
echo -e "\n${YELLOW}📝 Creating consolidated entry points...${NC}"

# Create main entry point
cat > main.py << 'EOF'
#!/usr/bin/env python3
"""
DeFi Trading System - Main Entry Point
Run with: python main.py
"""

import asyncio
import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core', 'engine'))

from pipeline import main_pipeline

if __name__ == "__main__":
    print("🚀 Starting DeFi Momentum Trading System")
    try:
        asyncio.run(main_pipeline())
    except KeyboardInterrupt:
        print("\n⏹️ Trading system stopped by user")
    except Exception as e:
        print(f"💥 System error: {e}")
        sys.exit(1)
EOF

# Create requirements consolidation
cat > requirements.txt << 'EOF'
# Core Dependencies
web3>=6.0.0
pandas>=1.5.0
numpy>=1.24.0
aiohttp>=3.8.0
redis>=4.5.0
pyyaml>=6.0

# ML Dependencies
torch>=2.0.0
tensorflow>=2.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
optuna>=3.1.0

# API Dependencies
fastapi>=0.95.0
uvicorn>=0.21.0
prometheus-client>=0.16.0

# Crypto Dependencies
eth-account>=0.8.0
eth-abi>=4.0.0

# Utility Dependencies
python-dotenv>=1.0.0
psutil>=5.9.0
numba>=0.56.0
EOF

# Create unified configuration
cat > config.py << 'EOF'
"""
Unified Configuration - Consolidates settings.yaml and environment variables
"""

import os
import yaml
from pathlib import Path

def load_config():
    """Load configuration from multiple sources"""
    config_path = Path(__file__).parent / "infrastructure" / "config" / "settings.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables
    config['wallet'] = {
        'address': os.getenv('WALLET_ADDRESS'),
        'private_key': os.getenv('PRIVATE_KEY')
    }
    
    config['rpc_urls'] = {
        'arbitrum': os.getenv('ARBITRUM_RPC_URL'),
        'polygon': os.getenv('POLYGON_RPC_URL'),
        'optimism': os.getenv('OPTIMISM_RPC_URL')
    }
    
    return config

# Global config instance
CONFIG = load_config()
EOF

echo -e "${GREEN}✅ Created consolidated entry points${NC}"

# 13. UPDATE IMPORT PATHS
echo -e "\n${YELLOW}🔧 Updating import paths...${NC}"

# Create path updater script
cat > scripts/update_imports.py << 'EOF'
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
            r'from signal_detector import': 'from intelligence.signals.signal_detector import',
            r'from trade_executor import': 'from core.execution.trade_executor import',
            r'from inference_model import': 'from core.models.inference_model import',
            r'from risk_manager import': 'from core.execution.risk_manager import',
            r'from safety_checks import': 'from security.validators.safety_checks import',
            r'from anti_rug_analyzer import': 'from security.rugpull.anti_rug_analyzer import',
            r'from mempool_watcher import': 'from security.mempool.mempool_watcher import',
            r'from batch_processor import': 'from core.engine.batch_processor import',
            r'from vectorized_features import': 'from core.features.vectorized_features import',
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
EOF

python scripts/update_imports.py

echo -e "${GREEN}✅ Import paths updated${NC}"

# 14. CREATE README FOR NEW STRUCTURE
cat > README.md << 'EOF'
# 🚀 DeFi Momentum Trading System

## 📁 Repository Structure

```
├── core/                          # Core trading engine
│   ├── engine/                    # Main pipeline and orchestration
│   │   ├── pipeline.py           # Main trading pipeline
│   │   └── batch_processor.py    # High-performance token processing
│   ├── execution/                 # Trade execution and risk management
│   │   ├── trade_executor.py     # Trade execution logic
│   │   ├── risk_manager.py       # Risk management systems
│   │   └── scanner_v3.py         # Token scanning engine
│   ├── models/                    # ML models and inference
│   │   ├── inference_model.py    # Main ML inference engine
│   │   └── model_manager.py      # Model lifecycle management
│   └── features/                  # Feature engineering
│       └── vectorized_features.py # High-performance feature extraction
│
├── intelligence/                  # AI and analysis systems
│   ├── signals/                   # Signal detection
│   │   └── signal_detector.py    # Momentum signal detection
│   ├── analysis/                  # Advanced analysis
│   │   ├── advanced_ensemble.py  # Multi-modal analysis
│   │   ├── continuous_optimizer.py # Parameter optimization
│   │   └── feedback_loop.py      # Learning feedback loops
│   └── sentiment/                 # Sentiment analysis (if exists)
│
├── security/                      # Security and safety systems
│   ├── validators/                # Input validation and safety
│   │   ├── safety_checks.py      # Comprehensive safety validation
│   │   └── token_profiler.py     # Token analysis and profiling
│   ├── rugpull/                   # Rugpull detection
│   │   └── anti_rug_analyzer.py  # Advanced rugpull protection
│   └── mempool/                   # MEV and mempool protection
│       └── mempool_watcher.py    # Frontrunning and MEV protection
│
├── infrastructure/               # Infrastructure and ops
│   ├── config/                   # Configuration files
│   │   └── settings.yaml        # Main configuration
│   ├── monitoring/               # Monitoring and logging
│   │   ├── performance_optimizer.py # System optimization
│   │   ├── logging_config.py    # Logging configuration
│   │   └── error_handler.py     # Error handling utilities
│   └── deployment/               # Deployment configs
│       ├── docker-compose.yml   # Docker orchestration
│       └── docker/              # Docker configurations
│
├── data/                         # Data storage
│   ├── cache/                    # Database and cache files
│   └── models/                   # Trained model files
│
├── notebooks/                    # Jupyter notebooks
├── scripts/                      # Utility scripts
├── tests/                        # Test files
│
├── main.py                       # Main entry point
├── config.py                     # Unified configuration
└── requirements.txt              # Dependencies
```

## 🚀 Quick Start

1. **Setup Environment:**
   ```bash
   pip install -r requirements.txt
   cp .env.example .env  # Configure your keys
   ```

2. **Run Trading System:**
   ```bash
   python main.py
   ```

3. **Run in Notebook:**
   ```bash
   jupyter notebook notebooks/run_pipeline.ipynb
   ```

## 📊 Architecture

- **Core Engine**: High-performance trading pipeline
- **Intelligence**: AI-driven signal detection and analysis  
- **Security**: Comprehensive safety and rugpull protection
- **Infrastructure**: Monitoring, logging, and deployment

## 🔧 Configuration

All configuration is centralized in `infrastructure/config/settings.yaml` and can be overridden with environment variables in `.env`.
EOF

echo -e "${GREEN}✅ README created${NC}"

# 15. FINAL SUMMARY
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}📊 REORGANIZATION SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${GREEN}✅ COMPLETED ACTIONS:${NC}"
echo -e "   📁 Created organized directory structure"
echo -e "   🚀 Moved core engine files to core/"
echo -e "   💼 Organized execution layer"
echo -e "   🧠 Structured intelligence components"
echo -e "   🛡️ Consolidated security systems"
echo -e "   🏗️ Organized infrastructure"
echo -e "   💾 Structured data storage"
echo -e "   📜 Consolidated scripts and utilities"
echo -e "   🔄 Merged redundant files"
echo -e "   🗑️ Cleaned up empty files"
echo -e "   🔧 Updated import paths"
echo -e "   📝 Created main entry points"

echo -e "\n${YELLOW}📂 NEW STRUCTURE:${NC}"
echo -e "   core/           - Trading engine and execution"
echo -e "   intelligence/   - AI and analysis systems"
echo -e "   security/       - Safety and protection"
echo -e "   infrastructure/ - Config and monitoring"
echo -e "   data/          - Cache and models"
echo -e "   notebooks/     - Jupyter notebooks"
echo -e "   scripts/       - Utility scripts"
echo -e "   tests/         - Test files"

echo -e "\n${PURPLE}🚀 NEXT STEPS:${NC}"
echo -e "   1. Test the reorganized structure: python main.py"
echo -e "   2. Update any remaining import paths manually"
echo -e "   3. Run tests: python -m pytest tests/"
echo -e "   4. Deploy: docker-compose up"

echo -e "\n${GREEN}✅ REORGANIZATION COMPLETE!${NC}"
echo -e "The repository is now organized by functional requirements for Renaissance Tech standards."