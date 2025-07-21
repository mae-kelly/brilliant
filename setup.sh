#!/bin/bash

# 🚀 DeFi Repository Optimization Runner
# Intelligently compresses and optimizes the repository structure

echo "🤖 DeFi Repository Optimization Starting..."
echo "========================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.7+"
    exit 1
fi

# Install required dependencies if not present
echo "📦 Installing required dependencies..."
pip install -q scikit-learn numpy

# Get current directory
REPO_DIR=$(pwd)
echo "📁 Repository path: $REPO_DIR"

# Create the optimization script if it doesn't exist
if [ ! -f "optimize_defi_repo.py" ]; then
    echo "❌ Optimization script not found!"
    echo "Please ensure optimize_defi_repo.py is in the current directory"
    exit 1
fi

# Run analysis first (dry run)
echo ""
echo "🔍 Phase 1: Repository Analysis (Dry Run)"
echo "----------------------------------------"
python3 optimize_defi_repo.py --repo-path "$REPO_DIR" --dry-run

echo ""
echo "⚠️  WARNING: This will modify your repository structure!"
echo "A backup will be created before any changes."
echo ""
read -p "Do you want to proceed with optimization? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Phase 2: Executing Optimizations"
    echo "-----------------------------------"
    python3 optimize_defi_repo.py --repo-path "$REPO_DIR"
    
    echo ""
    echo "✅ Optimization Complete!"
    echo "📊 Repository structure has been optimized"
    echo "💾 Original files backed up to: backup_original/"
    echo ""
    echo "🎯 Next Steps:"
    echo "1. Review the optimized structure"
    echo "2. Test the pipeline: python3 main.py"
    echo "3. Run benchmarks: python3 scripts/final_benchmark.py"
    
else
    echo ""
    echo "❌ Optimization cancelled by user"
    echo "Repository unchanged"
fi

echo ""
echo "========================================================"
echo "🏁 Repository optimization process complete"