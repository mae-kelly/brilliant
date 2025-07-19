import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class SecureConfig:
    @staticmethod
    def get_wallet_address() -> str:
        addr = os.getenv('WALLET_ADDRESS')
        if not addr or addr.startswith('0x1234'):
            raise ValueError("WALLET_ADDRESS not configured properly")
        return addr
    
    @staticmethod
    def get_private_key() -> str:
        key = os.getenv('PRIVATE_KEY')
        if not key or key.startswith('0x1234'):
            raise ValueError("PRIVATE_KEY not configured properly")
        return key
    
    @staticmethod
    def get_api_key(service: str = 'alchemy') -> str:
        key_map = {
            'alchemy': 'API_KEY',
            'etherscan': 'ETHERSCAN_API_KEY',
            'gecko': 'GECKO_API_KEY',
            'honeypot': 'HONEYPOT_API_KEY'
        }
        
        env_var = key_map.get(service, 'API_KEY')
        key = os.getenv(env_var)
        
        if not key or 'your_' in key.lower():
            raise ValueError(f"{env_var} not configured properly")
        return key
    
    @staticmethod
    def get_rpc_url(chain: str = 'ethereum') -> str:
        api_key = SecureConfig.get_api_key('alchemy')
        
        urls = {
            'ethereum': f'https://eth-mainnet.g.alchemy.com/v2/{api_key}',
            'arbitrum': f'https://arb-mainnet.g.alchemy.com/v2/{api_key}',
            'polygon': f'https://polygon-mainnet.g.alchemy.com/v2/{api_key}',
            'optimism': f'https://opt-mainnet.g.alchemy.com/v2/{api_key}'
        }
        
        return urls.get(chain, urls['ethereum'])
    
    @staticmethod
    def validate_all() -> bool:
        try:
            SecureConfig.get_wallet_address()
            SecureConfig.get_private_key()
            SecureConfig.get_api_key('alchemy')
            return True
        except ValueError as e:
            print(f"Configuration error: {e}")
            return False

config = SecureConfig()
