#!/bin/bash

# Renaissance DeFi Trading System - Repository Cleanup Script
# Removes unnecessary, duplicate, and placeholder files

set -e  # Exit on any error

echo "🧹 Starting Renaissance Trading System Repository Cleanup..."
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to safely delete file
delete_file() {
    local file="$1"
    if [[ -f "$file" ]]; then
        echo -e "${RED}🗑️  Deleting: $file${NC}"
        rm "$file"
    else
        echo -e "${YELLOW}⚠️  File not found: $file${NC}"
    fi
}

# Function to safely delete directory
delete_directory() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        echo -e "${RED}📁 Deleting directory: $dir${NC}"
        rm -rf "$dir"
    else
        echo -e "${YELLOW}⚠️  Directory not found: $dir${NC}"
    fi
}

echo -e "${BLUE}Phase 1: Removing duplicate/redundant files${NC}"
echo "--------------------------------------------------------"

# Duplicate files
delete_file "run_pipeline.ipynb"
delete_file "real_dex_executor_fixed.py" 
delete_file "live_data_streams_fixed.py"
delete_file "production_trading_system.py"

echo -e "\n${BLUE}Phase 2: Removing incomplete/placeholder files${NC}"
echo "--------------------------------------------------------"

# Placeholder/incomplete files
delete_file "enhanced_honeypot_detector.py"
delete_file "social_sentiment.py"
delete_file "token_graph.py"

echo -e "\n${BLUE}Phase 3: Removing development/testing files${NC}"
echo "--------------------------------------------------------"

# Development and testing files
delete_file "test_real_data_feeds.py"
delete_file "validate_real_data_setup.py"
delete_file "synthetic_training_data.py"

echo -e "\n${BLUE}Phase 4: Removing setup/installation files${NC}"
echo "--------------------------------------------------------"

# Setup files
delete_file "renaissance_trading/install_dependencies.py"
delete_file "renaissance_trading/setup_colab.py"
delete_file "renaissance_trading/validate_setup.py"

echo -e "\n${BLUE}Phase 5: Removing redundant configuration files${NC}"
echo "--------------------------------------------------------"

# Redundant configs
delete_file "config/renaissance_settings.yaml"
delete_file "config/production.yaml"

echo -e "\n${BLUE}Phase 6: Removing standalone runners${NC}"
echo "--------------------------------------------------------"

# Standalone runners
delete_file "run_production_system.py"
delete_file "run_production_system_real.py"

echo -e "\n${BLUE}Phase 7: Moving development scripts to archive${NC}"
echo "--------------------------------------------------------"

# Create archive directory for scripts we might need later
mkdir -p archive/dev_scripts

# Move instead of delete - these might be useful for development
if [[ -f "scripts/replace_hardcoded.py" ]]; then
    echo -e "${YELLOW}📦 Archiving: scripts/replace_hardcoded.py${NC}"
    mv "scripts/replace_hardcoded.py" "archive/dev_scripts/"
fi

if [[ -f "scripts/train_transformer.py" ]]; then
    echo -e "${YELLOW}📦 Archiving: scripts/train_transformer.py${NC}"
    mv "scripts/train_transformer.py" "archive/dev_scripts/"
fi

if [[ -f "scripts/validate_transformer.py" ]]; then
    echo -e "${YELLOW}📦 Archiving: scripts/validate_transformer.py${NC}"
    mv "scripts/validate_transformer.py" "archive/dev_scripts/"
fi

echo -e "\n${BLUE}Phase 8: Cleaning up empty directories${NC}"
echo "--------------------------------------------------------"

# Remove empty directories
find . -type d -empty -delete 2>/dev/null || true

echo -e "\n${BLUE}Phase 9: Repository structure optimization${NC}"
echo "--------------------------------------------------------"

# Create proper directory structure if missing
mkdir -p {models,cache,logs,data,charts}
mkdir -p {analyzers,scanners,executors,monitoring}
mkdir -p config

# Create .gitkeep files for empty directories that should exist
touch cache/.gitkeep
touch logs/.gitkeep
touch models/.gitkeep
touch charts/.gitkeep

echo -e "\n${GREEN}✅ Repository cleanup completed successfully!${NC}"
echo "================================================================"

echo -e "\n${BLUE}📊 Cleanup Summary:${NC}"
echo "• Removed duplicate and redundant files"
echo "• Deleted placeholder/incomplete implementations"  
echo "• Archived development scripts"
echo "• Cleaned up configuration redundancy"
echo "• Optimized directory structure"

echo -e "\n${GREEN}🎯 Next Steps:${NC}"
echo "1. Run 'git status' to review changes"
echo "2. Update imports in remaining files if needed"
echo "3. Test the main pipeline: python production_renaissance_system.py"
echo "4. Commit cleaned repository: git add . && git commit -m 'Repository cleanup'"

echo -e "\n${YELLOW}⚠️  Note: Check 'archive/dev_scripts/' for any scripts you might need${NC}"

# Final file count
echo -e "\n${BLUE}📈 Repository Statistics:${NC}"
PYTHON_FILES=$(find . -name "*.py" | wc -l)
TOTAL_FILES=$(find . -type f | wc -l)
echo "• Python files: $PYTHON_FILES"
echo "• Total files: $TOTAL_FILES"

echo -e "\n🎉 Repository is now optimized for production use!