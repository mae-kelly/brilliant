#!/bin/bash
# =============================================================================
# 🧹 RENAISSANCE CLEANUP - Remove Redundant & Low-Quality Files
# =============================================================================

set -e

echo "🧹 RENAISSANCE SYSTEM CLEANUP - Removing redundant files..."
echo "============================================================="

# Colors for output
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

# Create backup directory
print_status "INFO" "Creating backup directory..."
mkdir -p backup/$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backup/$(date +%Y%m%d_%H%M%S)"

# Function to safely remove file with backup
safe_remove() {
    local file=$1
    local reason=$2
    
    if [[ -f "$file" ]]; then
        print_status "WARNING" "Removing $file - $reason"
        cp "$file" "$BACKUP_DIR/" 2>/dev/null || true
        rm "$file"
    fi
}

# Function to safely remove directory with backup
safe_remove_dir() {
    local dir=$1
    local reason=$2
    
    if [[ -d "$dir" ]]; then
        print_status "WARNING" "Removing directory $dir - $reason"
        cp -r "$dir" "$BACKUP_DIR/" 2>/dev/null || true
        rm -rf "$dir"
    fi
}

print_status "INFO" "Step 1: Removing redundant scanner files..."
# Keep only scanner_v3.py (most advanced)
safe_remove "scanners/ultra_scale_scanner.py" "Redundant - functionality merged into scanner_v3"
safe_remove "scanners/real_enhanced_scanner.py" "Redundant - real features moved to scanner_v3"

print_status "INFO" "Step 2: Removing redundant executor files..."
# Keep executor_v3.py, remove older versions
safe_remove "executors/real_dex_executor.py" "Redundant - merged into executor_v3"
safe_remove "executors/real_trading_engine.py" "Redundant - merged into executor_v3"

print_status "INFO" "Step 3: Removing redundant honeypot detectors..."
# Keep honeypot_detector.py, remove real_ version
safe_remove "analyzers/real_honeypot_detector.py" "Redundant - merged into honeypot_detector"
safe_remove "analyzers/real_blockchain_analyzer.py" "Redundant - functionality integrated"

print_status "INFO" "Step 4: Removing redundant social sentiment analyzers..."
# Keep social_sentiment.py, remove real_ version
safe_remove "analyzers/real_social_sentiment.py" "Redundant - merged into social_sentiment"

print_status "INFO" "Step 5: Removing redundant model files..."
safe_remove "models/train_model.py" "Redundant - use model_trainer.py"
safe_remove "models/enhanced_momentum_analyzer.py" "Redundant - moved to analyzers/"

print_status "INFO" "Step 6: Removing development/test files..."
safe_remove "check_requirements.py" "Development utility - not needed in production"
safe_remove "fix_numpy_warnings.py" "Development utility"
safe_remove "intelligent_trade_executor.py" "Redundant - functionality in executor_v3"
safe_remove "trading_safeguards.py" "Redundant - functionality in utils/safety_manager"
safe_remove "synthetic_training_data.py" "Development utility"
safe_remove "one.py" "Development utility"
safe_remove "one.sh" "Development utility"

print_status "INFO" "Step 7: Removing empty/placeholder files..."
safe_remove "production_renaissance_system.py" "Empty file - real implementation in core/"

print_status "INFO" "Step 8: Consolidating requirements files..."
# Keep only requirements.txt and requirements_complete.txt
safe_remove "requirements_final.txt" "Redundant"
safe_remove "requirements_fixed.txt" "Redundant"
safe_remove "requirements_production_trading.txt" "Redundant"
safe_remove "requirements_renaissance.txt" "Redundant"
safe_remove "requirements_ultra_scanner.txt" "Redundant"

print_status "INFO" "Step 9: Removing redundant documentation..."
safe_remove "system_checklist.md" "Development artifact"

print_status "INFO" "Step 10: Cleaning up development scripts..."
safe_remove_dir "scripts/backup_manager.py" "Basic functionality - replaced with proper backup system"
safe_remove "scripts/model_versioning_system.py" "Over-engineered - simplified version in optimizer"

print_status "INFO" "Step 11: Removing redundant monitoring files..."
safe_remove "monitoring/monitoring.py" "Redundant - functionality merged into performance_tracker"

print_status "INFO" "Step 12: Cleaning up config files..."
# Keep only the essential config files
if [[ -f "settings.yaml" && -f "config/settings.yaml" ]]; then
    safe_remove "settings.yaml" "Duplicate - keeping config/settings.yaml"
fi

print_status "INFO" "Step 13: Removing backup files..."
find . -name "*.bak" -delete 2>/dev/null || true
find . -name "*.backup" -delete 2>/dev/null || true
find . -name "*.tmp" -delete 2>/dev/null || true

print_status "INFO" "Step 14: Cleaning up Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

print_status "SUCCESS" "Cleanup completed!"
echo
print_status "INFO" "Files backed up to: $BACKUP_DIR"
echo
print_status "SUCCESS" "✨ Repository cleaned and ready for Renaissance optimization!"

# Show final structure
print_status "INFO" "Final core structure:"
echo "├── core/production_renaissance_system.py"
echo "├── config/dynamic_parameters.py"
echo "├── config/dynamic_settings.py"
echo "├── config/optimizer.py"
echo "├── config/settings.yaml"
echo "├── scanners/scanner_v3.py"
echo "├── scanners/graphql_scanner.py"
echo "├── executors/executor_v3.py"
echo "├── executors/position_manager.py"
echo "├── executors/mev_protection.py"
echo "├── executors/gas_optimizer.py"
echo "├── models/model_inference.py"
echo "├── models/model_trainer.py"
echo "├── models/online_learner.py"
echo "├── models/renaissance_transformer.py"
echo "├── models/inference_server.py"
echo "├── analyzers/honeypot_detector.py"
echo "├── analyzers/token_profiler.py"
echo "├── analyzers/social_sentiment.py"
echo "├── analyzers/advanced_microstructure.py"
echo "├── monitoring/mempool_watcher.py"
echo "├── monitoring/performance_tracker.py"
echo "├── scripts/feedback_loop.py"
echo "├── utils/risk_manager.py"
echo "├── utils/safety_manager.py"
echo "├── utils/safe_operations.py"
echo "├── notebooks/run_pipeline.ipynb"
echo "├── requirements.txt"
echo "└── run_renaissance.py"