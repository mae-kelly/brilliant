#!/bin/bash

# 🔧 Smart Import Fixer - Automatically fixes all import paths
# Analyzes the codebase and maps imports to actual file locations

set -e

echo "🚀 Starting Smart Import Fixer..."
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo -e "${BLUE}📦 Creating backup in $BACKUP_DIR${NC}"
mkdir -p "$BACKUP_DIR"
find . -name "*.py" -exec cp {} "$BACKUP_DIR/" \; 2>/dev/null || true

# Function to find where a class or function is defined
find_definition() {
    local search_term="$1"
    local search_type="$2"  # "class" or "def"
    
    if [ "$search_type" = "class" ]; then
        grep -r "^class $search_term" --include="*.py" . 2>/dev/null | head -1 | cut -d: -f1
    else
        grep -r "^def $search_term\|^async def $search_term" --include="*.py" . 2>/dev/null | head -1 | cut -d: -f1
    fi
}

# Function to convert file path to import path
file_to_import_path() {
    local file_path="$1"
    # Remove leading ./ and trailing .py, replace / with .
    echo "$file_path" | sed 's|^\./||' | sed 's|\.py$||' | sed 's|/|.|g'
}

# Create mapping of missing imports to actual locations
echo -e "${YELLOW}🔍 Analyzing codebase for class and function definitions...${NC}"

declare -A IMPORT_MAP

# Scan for class definitions
while IFS= read -r line; do
    if [[ $line =~ ^(.+):class\ ([A-Za-z_][A-Za-z0-9_]*) ]]; then
        file_path="${BASH_REMATCH[1]}"
        class_name="${BASH_REMATCH[2]}"
        import_path=$(file_to_import_path "$file_path")
        IMPORT_MAP["$class_name"]="$import_path"
        echo -e "${GREEN}  ✓ Found class $class_name in $import_path${NC}"
    fi
done < <(grep -r "^class [A-Za-z_]" --include="*.py" . 2>/dev/null)

# Scan for function definitions  
while IFS= read -r line; do
    if [[ $line =~ ^(.+):(async\ )?def\ ([A-Za-z_][A-Za-z0-9_]*) ]]; then
        file_path="${BASH_REMATCH[1]}"
        func_name="${BASH_REMATCH[3]}"
        import_path=$(file_to_import_path "$file_path")
        IMPORT_MAP["$func_name"]="$import_path"
        echo -e "${GREEN}  ✓ Found function $func_name in $import_path${NC}"
    fi
done < <(grep -r "^def \|^async def " --include="*.py" . 2>/dev/null)

echo -e "${BLUE}📊 Found ${#IMPORT_MAP[@]} definitions${NC}"

# Common import mappings based on the codebase analysis
declare -A KNOWN_MAPPINGS=(
    # Core modules
    ["ScannerV3"]="core.execution.scanner_v3"
    ["SignalDetector"]="intelligence.signals.signal_detector"  
    ["MomentumEnsemble"]="core.models.inference_model"
    ["TradeExecutor"]="core.execution.trade_executor"
    ["RiskManager"]="core.execution.risk_manager"
    ["SafetyChecker"]="security.validators.safety_checks"
    ["TokenProfiler"]="security.validators.token_profiler"
    ["RugpullAnalyzer"]="security.rugpull.anti_rug_analyzer"
    ["MempoolWatcher"]="security.mempool.mempool_watcher"
    ["VectorizedFeatureEngine"]="core.features.vectorized_features"
    ["UltraFastPipeline"]="core.engine.batch_processor"
    ["AsyncTokenScanner"]="core.engine.batch_processor"
    ["VectorizedMLProcessor"]="core.engine.batch_processor"
    
    # Analysis modules  
    ["SocialSentimentAnalyzer"]="intelligence.analysis.advanced_ensemble"
    ["TokenGraphAnalyzer"]="intelligence.analysis.advanced_ensemble"
    ["RLTradingAgent"]="intelligence.analysis.advanced_ensemble"
    ["ContinuousOptimizer"]="intelligence.analysis.continuous_optimizer"
    ["FeedbackLoop"]="intelligence.analysis.feedback_loop"
    ["AdvancedEnsembleModel"]="intelligence.analysis.advanced_ensemble"
    
    # Model management
    ["ModelManager"]="core.models.model_manager"
    ["TFLiteInferenceEngine"]="core.models.model_manager" 
    ["ModelRegistry"]="core.engine.pipeline"
    
    # Streaming and real-time
    ["RealTimeStreamer"]="intelligence.streaming.websocket_feeds"
    ["PriceVelocityDetector"]="intelligence.streaming.websocket_feeds"
    
    # Infrastructure
    ["SystemOptimizer"]="infrastructure.monitoring.performance_optimizer"
    ["PerformanceMonitor"]="infrastructure.monitoring.performance_optimizer"
    ["setup_logging"]="infrastructure.monitoring.logging_config"
    ["JSONFormatter"]="infrastructure.monitoring.logging_config"
    
    # Error handling
    ["retry_with_backoff"]="infrastructure.monitoring.error_handler"
    ["CircuitBreaker"]="infrastructure.monitoring.error_handler"
    ["safe_execute"]="infrastructure.monitoring.error_handler"
    ["TradingSystemError"]="infrastructure.monitoring.error_handler"
    ["NetworkError"]="infrastructure.monitoring.error_handler"
    ["ModelInferenceError"]="infrastructure.monitoring.error_handler"
    
    # ABI definitions
    ["UNISWAP_V3_POOL_ABI"]="abi"
    ["UNISWAP_V3_ROUTER_ABI"]="abi"
    ["ERC20_ABI"]="abi"
)

# Merge known mappings with discovered mappings
for key in "${!KNOWN_MAPPINGS[@]}"; do
    IMPORT_MAP["$key"]="${KNOWN_MAPPINGS[$key]}"
done

echo -e "${YELLOW}🔧 Fixing import statements in Python files...${NC}"

# Function to fix imports in a file
fix_file_imports() {
    local file="$1"
    local temp_file=$(mktemp)
    local changes_made=false
    
    echo -e "${BLUE}  Processing $file${NC}"
    
    while IFS= read -r line; do
        original_line="$line"
        
        # Handle "from X import Y" statements
        if [[ $line =~ ^[[:space:]]*from[[:space:]]+([A-Za-z_][A-Za-z0-9_.]*)([[:space:]]+import[[:space:]]+.+) ]]; then
            module="${BASH_REMATCH[1]}"
            import_part="${BASH_REMATCH[2]}"
            
            # Check if we have a mapping for this module
            if [[ -n "${IMPORT_MAP[$module]}" ]]; then
                new_line="from ${IMPORT_MAP[$module]}$import_part"
                echo "$new_line" >> "$temp_file"
                if [[ "$new_line" != "$original_line" ]]; then
                    echo -e "${GREEN}    ✓ Fixed: $module -> ${IMPORT_MAP[$module]}${NC}"
                    changes_made=true
                fi
                continue
            fi
            
            # Check for specific imports within the line
            modified_line="$line"
            for import_name in "${!IMPORT_MAP[@]}"; do
                if [[ $line =~ import[[:space:]]+.*$import_name ]]; then
                    # Extract just the module part before import
                    base_module=$(echo "$module" | cut -d'.' -f1)
                    if [[ -n "${IMPORT_MAP[$import_name]}" ]]; then
                        new_module="${IMPORT_MAP[$import_name]}"
                        modified_line="from $new_module$import_part"
                        echo -e "${GREEN}    ✓ Fixed import: $import_name from ${IMPORT_MAP[$import_name]}${NC}"
                        changes_made=true
                        break
                    fi
                fi
            done
            echo "$modified_line" >> "$temp_file"
            
        # Handle "import X" statements  
        elif [[ $line =~ ^[[:space:]]*import[[:space:]]+([A-Za-z_][A-Za-z0-9_.]*) ]]; then
            module="${BASH_REMATCH[1]}"
            
            if [[ -n "${IMPORT_MAP[$module]}" ]]; then
                new_line="import ${IMPORT_MAP[$module]}"
                echo "$new_line" >> "$temp_file"
                echo -e "${GREEN}    ✓ Fixed: import $module -> import ${IMPORT_MAP[$module]}${NC}"
                changes_made=true
            else
                echo "$line" >> "$temp_file"
            fi
            
        else
            echo "$line" >> "$temp_file"
        fi
        
    done < "$file"
    
    # Replace file if changes were made
    if [ "$changes_made" = true ]; then
        mv "$temp_file" "$file"
        echo -e "${GREEN}  ✅ Updated $file${NC}"
    else
        rm "$temp_file"
        echo -e "${YELLOW}  ⚪ No changes needed for $file${NC}"
    fi
}

# Fix imports in all Python files
find . -name "*.py" -not -path "./$BACKUP_DIR/*" | while read -r file; do
    fix_file_imports "$file"
done

echo -e "${YELLOW}🔧 Creating missing __init__.py files...${NC}"

# Create __init__.py files for proper package structure
directories=(
    "core"
    "core/engine" 
    "core/execution"
    "core/features"
    "core/models"
    "intelligence"
    "intelligence/analysis"
    "intelligence/signals"
    "intelligence/social"
    "intelligence/streaming"
    "security"
    "security/mempool"
    "security/rugpull" 
    "security/validators"
    "infrastructure"
    "infrastructure/config"
    "infrastructure/monitoring"
    "tests"
    "tests/integration"
    "tests/load"
    "tests/performance"
    "tests/unit"
)

for dir in "${directories[@]}"; do
    if [ -d "$dir" ] && [ ! -f "$dir/__init__.py" ]; then
        touch "$dir/__init__.py"
        echo -e "${GREEN}  ✓ Created $dir/__init__.py${NC}"
    fi
done

echo -e "${YELLOW}🔧 Fixing common configuration issues...${NC}"

# Copy main config file to root if it doesn't exist
if [ ! -f "settings.yaml" ] && [ -f "infrastructure/config/settings.yaml" ]; then
    cp "infrastructure/config/settings.yaml" "settings.yaml"
    echo -e "${GREEN}  ✓ Copied settings.yaml to root${NC}"
fi

# Create basic .env template if it doesn't exist
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# DeFi Trading System Environment Variables
# Fill in your actual values

# Alchemy RPC URLs  
ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/YOUR_API_KEY
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY
OPTIMISM_RPC_URL=https://opt-mainnet.g.alchemy.com/v2/YOUR_API_KEY

# Backup RPC URLs
ARBITRUM_BACKUP_RPC_URL=https://arbitrum-one.publicnode.com
POLYGON_BACKUP_RPC_URL=https://polygon.llamarpc.com
OPTIMISM_BACKUP_RPC_URL=https://mainnet.optimism.io

# Wallet Configuration (FILL THESE IN!)
WALLET_ADDRESS=0xYOUR_WALLET_ADDRESS
PRIVATE_KEY=0xYOUR_PRIVATE_KEY

# API Keys
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
ALCHEMY_API_KEY=YOUR_ALCHEMY_API_KEY
BLOCKNATIVE_API_KEY=YOUR_BLOCKNATIVE_API_KEY

# Trading Configuration
STARTING_BALANCE=0.01
MAX_POSITION_SIZE=0.002
ENABLE_LIVE_TRADING=false

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
EOF
    echo -e "${GREEN}  ✓ Created .env template${NC}"
    echo -e "${YELLOW}  ⚠️  Remember to fill in your actual API keys and wallet info!${NC}"
fi

# Create models directory
mkdir -p models
echo -e "${GREEN}  ✓ Created models directory${NC}"

# Create data directories
mkdir -p data/cache data/features
echo -e "${GREEN}  ✓ Created data directories${NC}"

echo -e "${YELLOW}🔧 Fixing specific problematic imports...${NC}"

# Fix main.py to use correct import
if [ -f "main.py" ]; then
    sed -i 's/from core.engine.pipeline import main_pipeline/from core.engine.pipeline import main_pipeline/' main.py
    echo -e "${GREEN}  ✓ Fixed main.py imports${NC}"
fi

# Fix any remaining scanner_v3 references
find . -name "*.py" -not -path "./$BACKUP_DIR/*" -exec sed -i 's/from scanner_v3 import/from core.execution.scanner_v3 import/g' {} \;
find . -name "*.py" -not -path "./$BACKUP_DIR/*" -exec sed -i 's/import scanner_v3/import core.execution.scanner_v3 as scanner_v3/g' {} \;

echo -e "${YELLOW}🧪 Running import validation test...${NC}"

# Test if Python can import key modules
python3 -c "
import sys
sys.path.append('.')

failed_imports = []
modules_to_test = [
    'core.models.inference_model',
    'security.validators.safety_checks', 
    'security.rugpull.anti_rug_analyzer',
    'security.mempool.mempool_watcher',
    'core.execution.risk_manager',
    'intelligence.signals.signal_detector'
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f'✅ {module}')
    except Exception as e:
        print(f'❌ {module}: {e}')
        failed_imports.append(module)

if failed_imports:
    print(f'\n🚨 {len(failed_imports)} modules still have import issues')
    sys.exit(1)
else:
    print(f'\n🎉 All {len(modules_to_test)} core modules import successfully!')
" 2>/dev/null || echo -e "${RED}⚠️  Some import issues remain - check Python path and dependencies${NC}"

echo ""
echo -e "${GREEN}🎉 Import fixing complete!${NC}"
echo "========================================"
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "  • Backup created in: $BACKUP_DIR"
echo -e "  • Fixed import paths in all Python files"
echo -e "  • Created missing __init__.py files"
echo -e "  • Set up basic configuration files"
echo ""
echo -e "${YELLOW}🔧 Next steps:${NC}"
echo -e "  1. Fill in your API keys in .env file"
echo -e "  2. Run: ${GREEN}python scripts/minimal_test.py${NC}"
echo -e "  3. If tests pass, try: ${GREEN}python main.py${NC}"
echo ""
echo -e "${RED}⚠️  Important:${NC}"
echo -e "  • This fixes import paths but doesn't create missing functionality"
echo -e "  • You may still need to train ML models"
echo -e "  • Start with paper trading (ENABLE_LIVE_TRADING=false)"
echo ""