
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
except ImportError:
    def get_dynamic_config(): return {"volatility_threshold": 0.1, "confidence_threshold": 0.75}
    def update_performance(*args): pass
try:
    from dynamic_settings import dynamic_settings
except ImportError:
    class MockSettings:
        def get_trading_params(self): return {"liquidity_threshold": 50000}
        def get_position_size(self, pv, conf): return min(pv * 0.1, 1.0)
    dynamic_settings = MockSettings()
# Dynamic configuration import


import os
import json
import time
import shutil
from datetime import datetime
import sqlite3

class BackupManager:
    def __init__(self):
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
        os.makedirs(backup_path, exist_ok=True)
        
        if os.path.exists("cache/token_cache.db"):
            shutil.copy2("cache/token_cache.db", backup_path)
        
        if os.path.exists("logs"):
            shutil.copytree("logs", os.path.join(backup_path, "logs"))
        
        if os.path.exists("models"):
            shutil.copytree("models", os.path.join(backup_path, "models"))
        
        config_backup = {
            'timestamp': timestamp,
            'environment_vars': {k: v for k, v in os.environ.items() if 'API' not in k and 'KEY' not in k},
            'backup_path': backup_path
        }
        
        with open(os.path.join(backup_path, "config.json"), 'w') as f:
            json.dump(config_backup, f, indent=2)
        
        print(f"Backup created: {backup_path}")
        return backup_path
    
    def restore_backup(self, backup_path: str):
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        print(f"Restoring from backup: {backup_path}")
        
        if os.path.exists(os.path.join(backup_path, "token_cache.db")):
            os.makedirs("cache", exist_ok=True)
            shutil.copy2(os.path.join(backup_path, "token_cache.db"), "cache/")
        
        if os.path.exists(os.path.join(backup_path, "logs")):
            if os.path.exists("logs"):
                shutil.rmtree("logs")
            shutil.copytree(os.path.join(backup_path, "logs"), "logs")
        
        if os.path.exists(os.path.join(backup_path, "models")):
            if os.path.exists("models"):
                shutil.rmtree("models")
            shutil.copytree(os.path.join(backup_path, "models"), "models")
        
        print("Restore completed successfully")

backup_manager = BackupManager()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "backup":
            backup_manager.create_backup()
        elif sys.argv[1] == "restore" and len(sys.argv) > 2:
            backup_manager.restore_backup(sys.argv[2])
    else:
        print("Usage: python backup_manager.py [backup|restore <path>]")
