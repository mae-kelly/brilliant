#!/usr/bin/env python3
"""
Advanced Python Import Fixer
Analyzes the codebase and intelligently fixes all import statements
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import Dict, Set, List, Optional
import shutil
from datetime import datetime

class ImportFixer:
    def __init__(self):
        self.project_root = Path('.')
        self.backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.class_to_file = {}
        self.function_to_file = {}
        self.import_mappings = {}
        self.fixed_files = set()
        
        # Known mappings based on analysis
        self.known_mappings = {
            # Scanner and execution
            'ScannerV3': 'core.execution.scanner_v3',
            'scanner_v3': 'core.execution.scanner_v3',
            'SignalDetector': 'intelligence.signals.signal_detector',
            'signal_detector': 'intelligence.signals.signal_detector',
            
            # Models and ML
            'MomentumEnsemble': 'core.models.inference_model',
            'inference_model': 'core.models.inference_model',
            'ModelManager': 'core.models.model_manager',
            'TFLiteInferenceEngine': 'core.models.model_manager',
            'model_manager': 'core.models.model_manager',
            
            # Execution and trading
            'TradeExecutor': 'core.execution.trade_executor',
            'trade_executor': 'core.execution.trade_executor',
            'RiskManager': 'core.execution.risk_manager',
            'risk_manager': 'core.execution.risk_manager',
            
            # Security modules
            'SafetyChecker': 'security.validators.safety_checks',
            'safety_checks': 'security.validators.safety_checks',
            'TokenProfiler': 'security.validators.token_profiler',
            'token_profiler': 'security.validators.token_profiler',
            'RugpullAnalyzer': 'security.rugpull.anti_rug_analyzer',
            'anti_rug_analyzer': 'security.rugpull.anti_rug_analyzer',
            'MempoolWatcher': 'security.mempool.mempool_watcher',
            'mempool_watcher': 'security.mempool.mempool_watcher',
            
            # Advanced analysis
            'ContinuousOptimizer': 'intelligence.analysis.continuous_optimizer',
            'continuous_optimizer': 'intelligence.analysis.continuous_optimizer',
            'FeedbackLoop': 'intelligence.analysis.feedback_loop',
            'feedback_loop': 'intelligence.analysis.feedback_loop',
            'AdvancedEnsembleModel': 'intelligence.analysis.advanced_ensemble',
            'advanced_ensemble': 'intelligence.analysis.advanced_ensemble',
            'SocialSentimentAnalyzer': 'intelligence.analysis.advanced_ensemble',
            'TokenGraphAnalyzer': 'intelligence.analysis.advanced_ensemble',
            'RLTradingAgent': 'intelligence.analysis.advanced_ensemble',
            
            # Features and processing
            'VectorizedFeatureEngine': 'core.features.vectorized_features',
            'vectorized_features': 'core.features.vectorized_features',
            'UltraFastPipeline': 'core.engine.batch_processor',
            'AsyncTokenScanner': 'core.engine.batch_processor',
            'VectorizedMLProcessor': 'core.engine.batch_processor',
            'batch_processor': 'core.engine.batch_processor',
            
            # Streaming and real-time
            'RealTimeStreamer': 'intelligence.streaming.websocket_feeds',
            'PriceVelocityDetector': 'intelligence.streaming.websocket_feeds',
            'websocket_feeds': 'intelligence.streaming.websocket_feeds',
            
            # Infrastructure
            'SystemOptimizer': 'infrastructure.monitoring.performance_optimizer',
            'PerformanceMonitor': 'infrastructure.monitoring.performance_optimizer',
            'performance_optimizer': 'infrastructure.monitoring.performance_optimizer',
            'setup_logging': 'infrastructure.monitoring.logging_config',
            'JSONFormatter': 'infrastructure.monitoring.logging_config',
            'logging_config': 'infrastructure.monitoring.logging_config',
            
            # Error handling
            'retry_with_backoff': 'infrastructure.monitoring.error_handler',
            'CircuitBreaker': 'infrastructure.monitoring.error_handler',
            'safe_execute': 'infrastructure.monitoring.error_handler',
            'TradingSystemError': 'infrastructure.monitoring.error_handler',
            'NetworkError': 'infrastructure.monitoring.error_handler',
            'ModelInferenceError': 'infrastructure.monitoring.error_handler',
            'error_handler': 'infrastructure.monitoring.error_handler',
            
            # ABI and utilities
            'UNISWAP_V3_POOL_ABI': 'abi',
            'UNISWAP_V3_ROUTER_ABI': 'abi',
            'ERC20_ABI': 'abi',
            'abi': 'abi',
        }
    
    def create_backup(self):
        """Create backup of all Python files"""
        print(f"📦 Creating backup in {self.backup_dir}")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        for py_file in self.project_root.rglob("*.py"):
            if not str(py_file).startswith(self.backup_dir):
                try:
                    shutil.copy2(py_file, self.backup_dir)
                except Exception as e:
                    print(f"⚠️  Could not backup {py_file}: {e}")
    
    def scan_definitions(self):
        """Scan all Python files for class and function definitions"""
        print("🔍 Scanning for class and function definitions...")
        
        for py_file in self.project_root.rglob("*.py"):
            if str(py_file).startswith(self.backup_dir):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse with AST for accurate analysis
                try:
                    tree = ast.parse(content)
                    module_path = self.file_to_module_path(py_file)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            self.class_to_file[node.name] = module_path
                            print(f"  ✓ Found class {node.name} in {module_path}")
                        elif isinstance(node, ast.FunctionDef):
                            self.function_to_file[node.name] = module_path
                            
                except SyntaxError:
                    # Fallback to regex for files with syntax errors
                    self.scan_with_regex(py_file, content)
                    
            except Exception as e:
                print(f"⚠️  Error scanning {py_file}: {e}")
    
    def scan_with_regex(self, py_file: Path, content: str):
        """Fallback regex scanning for files with syntax errors"""
        module_path = self.file_to_module_path(py_file)
        
        # Find class definitions
        for match in re.finditer(r'^class\s+([A-Za-z_][A-Za-z0-9_]*)', content, re.MULTILINE):
            class_name = match.group(1)
            self.class_to_file[class_name] = module_path
            
        # Find function definitions
        for match in re.finditer(r'^(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)', content, re.MULTILINE):
            func_name = match.group(1)
            self.function_to_file[func_name] = module_path
    
    def file_to_module_path(self, file_path: Path) -> str:
        """Convert file path to Python import path"""
        relative_path = file_path.relative_to(self.project_root)
        module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')
        return module_path
    
    def build_import_mappings(self):
        """Build comprehensive import mappings"""
        print("🗺️  Building import mappings...")
        
        # Combine discovered mappings with known mappings
        self.import_mappings.update(self.known_mappings)
        self.import_mappings.update(self.class_to_file)
        self.import_mappings.update(self.function_to_file)
        
        print(f"📊 Total mappings: {len(self.import_mappings)}")
    
    def fix_file_imports(self, file_path: Path) -> bool:
        """Fix imports in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix "from X import Y" statements
            def fix_from_import(match):
                full_match = match.group(0)
                module = match.group(1)
                import_part = match.group(2)
                
                # Check if we have a direct mapping
                if module in self.import_mappings:
                    return f"from {self.import_mappings[module]}{import_part}"
                
                # Check for partial matches
                for key, value in self.import_mappings.items():
                    if module.endswith(key) or key in module:
                        return f"from {value}{import_part}"
                
                return full_match
            
            content = re.sub(
                r'from\s+([A-Za-z_][A-Za-z0-9_.]*)((?:\s+import\s+.+))',
                fix_from_import,
                content
            )
            
            # Fix "import X" statements
            def fix_import(match):
                full_match = match.group(0)
                module = match.group(1)
                
                if module in self.import_mappings:
                    return f"import {self.import_mappings[module]}"
                
                return full_match
            
            content = re.sub(
                r'import\s+([A-Za-z_][A-Za-z0-9_.]*)',
                fix_import,
                content
            )
            
            # Write back if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.add(str(file_path))
                return True
                
        except Exception as e:
            print(f"⚠️  Error fixing {file_path}: {e}")
            
        return False
    
    def fix_all_imports(self):
        """Fix imports in all Python files"""
        print("🔧 Fixing import statements...")
        
        fixed_count = 0
        for py_file in self.project_root.rglob("*.py"):
            if str(py_file).startswith(self.backup_dir):
                continue
                
            if self.fix_file_imports(py_file):
                print(f"  ✅ Fixed {py_file}")
                fixed_count += 1
            else:
                print(f"  ⚪ No changes needed for {py_file}")
        
        print(f"📊 Fixed imports in {fixed_count} files")
    
    def create_init_files(self):
        """Create missing __init__.py files"""
        print("📁 Creating missing __init__.py files...")
        
        directories = [
            "core", "core/engine", "core/execution", "core/features", "core/models",
            "intelligence", "intelligence/analysis", "intelligence/signals", 
            "intelligence/social", "intelligence/streaming",
            "security", "security/mempool", "security/rugpull", "security/validators",
            "infrastructure", "infrastructure/config", "infrastructure/monitoring",
            "tests", "tests/integration", "tests/load", "tests/performance", "tests/unit"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            init_file = dir_path / "__init__.py"
            
            if dir_path.exists() and not init_file.exists():
                init_file.touch()
                print(f"  ✓ Created {init_file}")
    
    def setup_config_files(self):
        """Set up basic configuration files"""
        print("⚙️  Setting up configuration files...")
        
        # Copy settings.yaml to root if needed
        settings_source = self.project_root / "infrastructure" / "config" / "settings.yaml"
        settings_target = self.project_root / "settings.yaml"
        
        if settings_source.exists() and not settings_target.exists():
            shutil.copy2(settings_source, settings_target)
            print("  ✓ Copied settings.yaml to root")
        
        # Create .env template if needed
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_template = '''# DeFi Trading System Environment Variables
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

# Trading Configuration
STARTING_BALANCE=0.01
ENABLE_LIVE_TRADING=false
'''
            with open(env_file, 'w') as f:
                f.write(env_template)
            print("  ✓ Created .env template")
        
        # Create necessary directories
        for directory in ["models", "data", "data/cache", "data/features"]:
            (self.project_root / directory).mkdir(exist_ok=True)
            print(f"  ✓ Created {directory} directory")
    
    def validate_imports(self):
        """Test if key modules can be imported"""
        print("🧪 Validating imports...")
        
        test_modules = [
            'core.models.inference_model',
            'security.validators.safety_checks',
            'security.rugpull.anti_rug_analyzer',
            'security.mempool.mempool_watcher',
            'core.execution.risk_manager',
            'intelligence.signals.signal_detector',
            'abi'
        ]
        
        failed_imports = []
        
        for module in test_modules:
            try:
                __import__(module)
                print(f"  ✅ {module}")
            except Exception as e:
                print(f"  ❌ {module}: {e}")
                failed_imports.append(module)
        
        if failed_imports:
            print(f"\n🚨 {len(failed_imports)} modules still have import issues")
        else:
            print(f"\n🎉 All {len(test_modules)} core modules import successfully!")
        
        return len(failed_imports) == 0
    
    def run(self):
        """Run the complete import fixing process"""
        print("🚀 Starting Advanced Import Fixer")
        print("=" * 50)
        
        # Create backup
        self.create_backup()
        
        # Scan for definitions
        self.scan_definitions()
        
        # Build mappings
        self.build_import_mappings()
        
        # Fix imports
        self.fix_all_imports()
        
        # Create init files
        self.create_init_files()
        
        # Setup config
        self.setup_config_files()
        
        # Validate
        success = self.validate_imports()
        
        print("\n" + "=" * 50)
        print("🎉 Import fixing complete!")
        print(f"📊 Summary:")
        print(f"  • Backup created in: {self.backup_dir}")
        print(f"  • Fixed {len(self.fixed_files)} files")
        print(f"  • Mapped {len(self.import_mappings)} definitions")
        print(f"  • Import validation: {'✅ PASSED' if success else '❌ ISSUES REMAIN'}")
        
        print(f"\n🔧 Next steps:")
        print(f"  1. Fill in your API keys in .env file")
        print(f"  2. Run: python scripts/minimal_test.py")
        print(f"  3. If tests pass, try: python main.py")
        
        return success

if __name__ == "__main__":
    fixer = ImportFixer()
    success = fixer.run()
    sys.exit(0 if success else 1)