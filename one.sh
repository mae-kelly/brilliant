#!/bin/bash

# =============================================================================
# CODEBASE CLEANUP SCRIPT - Remove Unneeded Files
# Keeps only production-ready, Renaissance-level components
# =============================================================================

set -e

echo "🧹 CLEANING UP CODEBASE FOR PRODUCTION"
echo "======================================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[CLEANUP] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[REMOVE] $1${NC}"
}

# Create backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Function to safely remove files
safe_remove() {
    local file="$1"
    local reason="$2"
    
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/" 2>/dev/null || true
        rm "$file"
        error "Removed $file - $reason"
    fi
}

# Function to safely remove directories
safe_remove_dir() {
    local dir="$1"
    local reason="$2"
    
    if [ -d "$dir" ]; then
        cp -r "$dir" "$BACKUP_DIR/" 2>/dev/null || true
        rm -rf "$dir"
        error "Removed directory $dir - $reason"
    fi
}

log "Starting cleanup process..."

# =============================================================================
# REMOVE REDUNDANT/OUTDATED SCANNER FILES
# =============================================================================

warn "Removing redundant scanner versions..."

# Keep only the production ultra-scale scanner
safe_remove "scanner_v3.py" "Replaced by ultra_scale_scanner.py"
safe_remove "websocket_scanner_working.py" "Redundant with ultra_scale_scanner.py"
safe_remove "integrate_websocket_scanner.py" "Integrated into main scanner"

# Remove old data streams
safe_remove "live_data_streams.py" "Replaced by live_data_streams_fixed.py"

# =============================================================================
# REMOVE REDUNDANT EXECUTOR FILES
# =============================================================================

warn "Removing redundant executor versions..."

# Keep only the production executors
safe_remove "executor_v3.py" "Replaced by production_dex_executor.py"
safe_remove "real_dex_executor.py" "Replaced by real_dex_executor_fixed.py"

# Remove redundant executor in executors directory
safe_remove "executors/fixed_real_executor.py" "Redundant with main executor"

# =============================================================================
# REMOVE TEST/DEVELOPMENT FILES
# =============================================================================

warn "Removing test and development files..."

# Test files
safe_remove "test_complete_system.py" "Development testing only"
safe_remove "test_fixed_executor.py" "Development testing only"
safe_remove "test_fixed_system.py" "Development testing only"
safe_remove "test_framework.py" "Development testing only"
safe_remove "test_missing_modules.py" "Development testing only"
safe_remove "test_production_system.py" "Development testing only"
safe_remove "test_real_executor.py" "Development testing only"
safe_remove "test_ultra_scanner.py" "Development testing only"

# Development mode files
safe_remove "dev_mode.py" "Development wrapper not needed in production"
safe_remove "start_bot.py" "Replaced by production system"

# =============================================================================
# REMOVE REDUNDANT SYSTEM FILES
# =============================================================================

warn "Removing redundant system implementations..."

# Keep only the main complete system
safe_remove "complete_trading_system.py" "Redundant with production system"
safe_remove "fixed_production_system.py" "Redundant with main production system"

# Remove universal executor (redundant)
safe_remove "universal_dex_executor.py" "Redundant with specialized executors"

# =============================================================================
# REMOVE CONFIGURATION DUPLICATES
# =============================================================================

warn "Removing configuration duplicates..."

# Keep YAML, remove JSON duplicates
safe_remove "config/settings.json" "Duplicate of settings.yaml"

# Remove redundant config files
safe_remove "production_config.py" "Integrated into main config"

# =============================================================================
# REMOVE UTILITY DUPLICATES
# =============================================================================

warn "Removing utility duplicates..."

# Remove redundant optimization files
safe_remove "optimizer.py" "Replaced by dynamic_parameters.py"

# Remove redundant error handling
safe_remove "error_handler.py" "Integrated into safe_operations.py"

# Remove redundant logging
safe_remove "logging_config.py" "Integrated into main modules"

# =============================================================================
# REMOVE DEPLOYMENT/BUILD FILES
# =============================================================================

warn "Removing deployment helper files..."

# Keep only the master deployment script
safe_remove "implement_production.sh" "Redundant with one.sh"
safe_remove "fix_production_setup.sh" "Temporary fix script"
safe_remove "deploy/init_pipeline.sh" "Redundant with main init"

# Remove update scripts
safe_remove "update_pipeline_for_websockets.py" "One-time update script"

# =============================================================================
# REMOVE EMPTY/PLACEHOLDER FILES
# =============================================================================

warn "Removing placeholder files..."

# Remove version suffixes that indicate outdated files
safe_remove "requirements_production.txt" "Use requirements_production_trading.txt"

# =============================================================================
# CLEAN UP DIRECTORIES
# =============================================================================

warn "Cleaning up directories..."

# Remove deploy directory if mostly empty utilities
if [ -d "deploy" ]; then
    file_count=$(find deploy -name "*.py" | wc -l)
    if [ "$file_count" -le 2 ]; then
        safe_remove_dir "deploy" "Contains only utility files"
    fi
fi

# =============================================================================
# REMOVE LEGACY/BACKUP FILES
# =============================================================================

warn "Removing legacy and backup files..."

# Remove any backup or temporary files
find . -name "*.bak" -exec rm -f {} \; 2>/dev/null || true
find . -name "*.tmp" -exec rm -f {} \; 2>/dev/null || true
find . -name "*.old" -exec rm -f {} \; 2>/dev/null || true
find . -name "*~" -exec rm -f {} \; 2>/dev/null || true

# Remove any .pyc files
find . -name "*.pyc" -exec rm -f {} \; 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null || true

# =============================================================================
# VERIFY CORE FILES REMAIN
# =============================================================================

log "Verifying core production files remain..."

# Critical files that must exist
CORE_FILES=(
    "run_pipeline.ipynb"
    "production_trading_system.py"
    "scanners/ultra_scale_scanner.py"
    "executors/real_dex_executor_fixed.py"
    "live_data_streams_fixed.py"
    "enhanced_momentum_analyzer.py"
    "intelligent_trade_executor.py"
    "analyzers/anti_rug_analyzer.py"
    "profilers/token_profiler.py"
    "watchers/mempool_watcher.py"
    "config/dynamic_parameters.py"
    "config/settings.yaml"
    "models/train_model.py"
    "deploy/inference_server.py"
    "synthetic_training_data.py"
    "one.sh"
)

missing_files=()
for file in "${CORE_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    error "CRITICAL: Missing core files after cleanup:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please restore from backup: $BACKUP_DIR"
    exit 1
fi

# =============================================================================
# FINAL CLEANUP SUMMARY
# =============================================================================

log "Cleanup completed successfully!"
echo ""
echo "📊 CLEANUP SUMMARY:"
echo "===================="

echo "✅ KEPT (Core Production Files):"
echo "  🚀 Main System: production_trading_system.py"
echo "  🔍 Scanner: scanners/ultra_scale_scanner.py" 
echo "  💼 Executor: executors/real_dex_executor_fixed.py"
echo "  🧠 ML: enhanced_momentum_analyzer.py, intelligent_trade_executor.py"
echo "  🛡️ Safety: analyzers/anti_rug_analyzer.py"
echo "  📊 Profiling: profilers/token_profiler.py"
echo "  📡 Mempool: watchers/mempool_watcher.py"
echo "  ⚙️ Config: config/dynamic_parameters.py, config/settings.yaml"
echo "  🏗️ Training: synthetic_training_data.py, models/train_model.py"
echo "  🎯 Orchestrator: run_pipeline.ipynb"
echo "  🚀 Deployment: one.sh"
echo ""

echo "❌ REMOVED (Redundant/Outdated):"
if [ -d "$BACKUP_DIR" ] && [ "$(ls -A $BACKUP_DIR)" ]; then
    echo "  📁 Backed up to: $BACKUP_DIR"
    echo "  📝 File count: $(find $BACKUP_DIR -type f | wc -l) files"
else
    echo "  🎉 No files needed removal - codebase was already clean!"
fi

echo ""
echo "🎯 NEXT STEPS:"
echo "=============="
echo "1. Review remaining files match your requirements"
echo "2. Run: ./one.sh to deploy the cleaned system"
echo "3. Test: python production_trading_system.py"
echo "4. Launch: Open run_pipeline.ipynb in Colab"
echo ""

log "🧹 Codebase cleanup complete! Ready for Renaissance-level trading."