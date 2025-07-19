from safe_operations import logger, retry_on_failure, safe_execute, file_ops, net_ops

import os
import requests
from web3 import Web3
from typing import Dict, Optional
import time

class EnhancedHoneypotDetector:
    def __init__(self, rpc_url: str):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.honeypot_apis = [
            "https://api.honeypot.is/v2/IsHoneypot",
            "https://api.rugdoc.io/v1/scan"
        ]
        self.cache = {}
        self.cache_ttl = 300
    
    def check_multiple_sources(self, token_address: str) -> Dict[str, bool]:
        results = {}
        
        for api_url in self.honeypot_apis:
            try:
                if "honeypot.is" in api_url:
                    response = requests.get(f"{api_url}?address={token_address}", timeout=10)
                    data = response.json()
                    results['honeypot_is'] = not data.get('IsHoneypot', True)
                elif "rugdoc.io" in api_url:
                    response = requests.get(f"{api_url}/{token_address}", timeout=10)
                    data = response.json()
                    results['rugdoc'] = data.get('risk_level', 'high') == 'low'
            except Exception as e:
    logger.error(f"Error: {e}")
    continue
        
        return results
    
    def simulate_full_trade_cycle(self, token_address: str, test_amount: int = 10**15) -> bool:
        try:
            router_address = os.getenv("WALLET_ADDRESS", "os.getenv("WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")")
            weth_address = os.getenv("WALLET_ADDRESS", "os.getenv("WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")")
            
            router_abi = [{"inputs": [{"internalType": "uint256", "name": "amountOutMin", "type": "uint256"}, {"internalType": "address[]", "name": "path", "type": "address[]"}, {"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "deadline", "type": "uint256"}], "name": "swapExactETHForTokens", "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}], "stateMutability": "payable", "type": "function"}, {"inputs": [{"internalType": "uint256", "name": "amountIn", "type": "uint256"}, {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"}, {"internalType": "address[]", "name": "path", "type": "address[]"}, {"internalType": "address", "name": "to", "type": "address"}, {"internalType": "uint256", "name": "deadline", "type": "uint256"}], "name": "swapExactTokensForETH", "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}], "stateMutability": "nonpayable", "type": "function"}]
            
            router = self.web3.eth.contract(address=router_address, abi=router_abi)
            test_wallet = os.getenv("WALLET_ADDRESS", "os.getenv("WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")")
            deadline = int(time.time()) + 120
            
            buy_path = [weth_address, token_address]
            sell_path = [token_address, weth_address]
            
            buy_tx = router.functions.swapExactETHForTokens(
                0, buy_path, test_wallet, deadline
            ).build_transaction({
                'from': test_wallet,
                'value': test_amount,
                'gas': 300000
            })
            
            sell_tx = router.functions.swapExactTokensForETH(
                1000, 0, sell_path, test_wallet, deadline
            ).build_transaction({
                'from': test_wallet,
                'gas': 300000
            })
            
            self.web3.eth.call(buy_tx, 'latest')
            self.web3.eth.call(sell_tx, 'latest')
            
            return True
        except:
            return False
    
    def comprehensive_check(self, token_address: str) -> Dict[str, any]:
        cache_key = f"{token_address}_{int(time.time() // self.cache_ttl)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = {
            'is_safe': False,
            'external_apis': self.check_multiple_sources(token_address),
            'simulation_passed': self.simulate_full_trade_cycle(token_address),
            'checked_at': time.time()
        }
        
        api_results = list(result['external_apis'].values())
        api_consensus = sum(api_results) >= len(api_results) // 2 if api_results else False
        
        result['is_safe'] = api_consensus and result['simulation_passed']
        
        self.cache[cache_key] = result
        return result

enhanced_detector = EnhancedHoneypotDetector("https://mainnet.infura.io/v3/" + os.getenv("API_KEY", "")")
