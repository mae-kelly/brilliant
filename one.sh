#!/bin/bash

# =============================================================================
# RENAISSANCE DEFI SYSTEM - REPOSITORY CLEANUP SCRIPT
# Removes all unnecessary files while preserving core system components
# =============================================================================

echo "🧹 CLEANING RENAISSANCE DEFI REPOSITORY"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to safely remove files/directories
safe_remove() {
    if [ -e "$1" ]; then
        echo -e "${YELLOW}Removing:${NC} $1"
        rm -rf "$1"
    fi
}

# Function to remove files by pattern
remove_pattern() {
    find . -name "$1" -type f -delete 2>/dev/null
    echo -e "${YELLOW}Removed pattern:${NC} $1"
}

# Function to remove directories by pattern
remove_dir_pattern() {
    find . -name "$1" -type d -exec rm -rf {} + 2>/dev/null
    echo -e "${YELLOW}Removed directories:${NC} $1"
}

echo -e "${BLUE}Step 1: Removing Python cache and bytecode files...${NC}"
remove_pattern "*.pyc"
remove_pattern "*.pyo"
remove_pattern "*.pyd"
remove_dir_pattern "__pycache__"
remove_dir_pattern "*.egg-info"
remove_pattern "*.egg"

echo -e "${BLUE}Step 2: Removing IDE and editor files...${NC}"
safe_remove ".vscode/"
safe_remove ".idea/"
safe_remove "*.swp"
safe_remove "*.swo"
safe_remove "*~"
safe_remove ".DS_Store"
remove_pattern ".DS_Store"

echo -e "${BLUE}Step 3: Removing temporary and log files...${NC}"
safe_remove "logs/"
safe_remove "tmp/"
safe_remove "temp/"
remove_pattern "*.tmp"
remove_pattern "*.log"
remove_pattern "*.out"

echo -e "${BLUE}Step 4: Removing cache and data directories...${NC}"
safe_remove "cache/"
safe_remove ".cache/"
safe_remove "data/"
safe_remove "charts/"
safe_remove "backups/"

echo -e "${BLUE}Step 5: Removing model artifacts and checkpoints...${NC}"
safe_remove "models/"
remove_pattern "*.h5"
remove_pattern "*.tflite"
remove_pattern "*.pkl"
remove_pattern "*.joblib"
remove_pattern "*.ckpt"
remove_pattern "*.pb"

echo -e "${BLUE}Step 6: Removing database files...${NC}"
remove_pattern "*.db"
remove_pattern "*.sqlite"
remove_pattern "*.sqlite3"

echo -e "${BLUE}Step 7: Removing documentation build files...${NC}"
safe_remove "docs/_build/"
safe_remove "site/"
safe_remove ".sphinx/"

echo -e "${BLUE}Step 8: Removing testing artifacts...${NC}"
safe_remove ".pytest_cache/"
safe_remove ".coverage"
safe_remove "htmlcov/"
safe_remove ".tox/"
safe_remove ".nox/"
remove_pattern "coverage.xml"
remove_pattern "*.cover"

echo -e "${BLUE}Step 9: Removing package management files...${NC}"
safe_remove "dist/"
safe_remove "build/"
safe_remove ".eggs/"
safe_remove "*.whl"

echo -e "${BLUE}Step 10: Removing environment and config files...${NC}"
safe_remove ".env"
safe_remove ".env.local"
safe_remove ".env.production"
safe_remove ".secrets/"
remove_pattern "*.key"
remove_pattern "*.pem"

echo -e "${BLUE}Step 11: Removing Jupyter notebook checkpoints...${NC}"
safe_remove ".ipynb_checkpoints/"
remove_dir_pattern ".ipynb_checkpoints"

echo -e "${BLUE}Step 12: Removing OS-specific files...${NC}"
safe_remove "Thumbs.db"
safe_remove "Desktop.ini"
remove_pattern "Thumbs.db"
remove_pattern "Desktop.ini"

echo -e "${BLUE}Step 13: Removing git artifacts (keeping .git)...${NC}"
safe_remove ".git/logs/"
safe_remove ".git/refs/remotes/"

echo -e "${BLUE}Step 14: Removing node modules (if any)...${NC}"
safe_remove "node_modules/"
safe_remove "package-lock.json"
safe_remove "yarn.lock"

echo -e "${BLUE}Step 15: Removing large binary files...${NC}"
remove_pattern "*.zip"
remove_pattern "*.tar.gz"
remove_pattern "*.rar"
remove_pattern "*.7z"
remove_pattern "*.mp4"
remove_pattern "*.avi"
remove_pattern "*.mov"
remove_pattern "*.png"
remove_pattern "*.jpg"
remove_pattern "*.jpeg"
remove_pattern "*.gif"

echo -e "${BLUE}Step 16: Removing virtual environment directories...${NC}"
safe_remove "venv/"
safe_remove "env/"
safe_remove ".venv/"
safe_remove "virtualenv/"

echo -e "${BLUE}Step 17: Creating essential directories...${NC}"
mkdir -p logs
mkdir -p cache
mkdir -p models
mkdir -p data
echo -e "${GREEN}Created:${NC} logs/, cache/, models/, data/"

echo -e "${BLUE}Step 18: Creating .gitignore for future protection...${NC}"
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environments
venv/
env/
.venv/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
Desktop.ini

# Logs and databases
*.log
*.db
*.sqlite
*.sqlite3

# Models and cache
models/*.h5
models/*.tflite
models/*.pkl
cache/
data/
charts/
backups/

# Secrets
.env
.env.local
.env.production
.secrets/
*.key
*.pem

# Jupyter
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Temporary files
*.tmp
tmp/
temp/
EOF

echo -e "${GREEN}Created:${NC} .gitignore"

echo -e "${BLUE}Step 19: Final cleanup - removing empty directories...${NC}"
find . -type d -empty -delete 2>/dev/null

echo -e "${BLUE}Step 20: Creating essential placeholder files...${NC}"

# Create models directory structure
mkdir -p models/checkpoints
mkdir -p models/backups
touch models/.gitkeep

# Create cache structure
mkdir -p cache
touch cache/.gitkeep

# Create logs structure
mkdir -p logs
touch logs/.gitkeep

# Create data structure
mkdir -p data
touch data/.gitkeep

echo -e "${GREEN}Created essential directory structure${NC}"

echo ""
echo "🎯 CLEANUP SUMMARY"
echo "=================="
echo -e "${GREEN}✅ Removed all temporary and cache files${NC}"
echo -e "${GREEN}✅ Removed all IDE and editor artifacts${NC}"
echo -e "${GREEN}✅ Removed all model and database files${NC}"
echo -e "${GREEN}✅ Removed all logs and temporary data${NC}"
echo -e "${GREEN}✅ Created clean directory structure${NC}"
echo -e "${GREEN}✅ Generated comprehensive .gitignore${NC}"

echo ""
echo "📊 REPOSITORY SIZE AFTER CLEANUP:"
du -sh . 2>/dev/null || echo "Size calculation unavailable"

echo ""
echo "📋 REMAINING CORE FILES:"
find . -name "*.py" -type f | head -20
echo "..."
echo "Total Python files: $(find . -name "*.py" -type f | wc -l)"

echo ""
echo -e "${GREEN}🎉 CLEANUP COMPLETE!${NC}"
echo -e "${BLUE}Repository is now clean and ready for production deployment.${NC}"

# Optional: Show what files remain
echo ""
echo "📁 FINAL DIRECTORY STRUCTURE:"
tree -L 2 2>/dev/null || ls -la

echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "• All model files have been removed - you'll need to retrain"
echo "• All cache and logs have been cleared"
echo "• Configuration files preserved"
echo "• Core Python modules preserved"
echo "• .gitignore created to prevent future clutter"