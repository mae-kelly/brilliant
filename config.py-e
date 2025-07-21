"""
Unified Configuration - Consolidates settings.yaml and environment variables
"""

import os
import yaml
from pathlib import Path

def load_config():
    """Load configuration from multiple sources"""
    config_path = Path(__file__).parent / "infrastructure" / "config" / "settings.yaml"
    
    if not config_path.exists():
        # Fallback to current directory
        config_path = Path("settings.yaml")
    
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
