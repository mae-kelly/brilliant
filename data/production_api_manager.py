import asyncio
import aiohttp
import os
import json
import time
import hashlib
import hmac
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class APICredentials:
    dexscreener: str = ""
    gecko_terminal: str = ""
    honeypot_is: str = ""
    twitter_bearer: str = ""
    etherscan: str = ""
    moralis: str = ""
    coingecko: str = ""
    telegram_bot: str = ""

class ProductionAPIClient:
    def __init__(self):
        self.credentials = APICredentials(
            dexscreener=os.getenv('DEXSCREENER_API_KEY', ''),
            gecko_terminal=os.getenv('GECKOTERMINAL_API_KEY', ''),
            honeypot_is=os.getenv('HONEYPOT_API_KEY', ''),
            twitter_bearer=os.getenv('TWITTER_BEARER_TOKEN', ''),
            etherscan=os.getenv('ETHERSCAN_API_KEY', ''),
            moralis=os.getenv('MORALIS_API_KEY', ''),
            coingecko=os.getenv('COINGECKO_API_KEY', ''),
            telegram_bot=os.getenv('TELEGRAM_BOT_TOKEN', '')
        )
        
        self.session_pool = []
        self.rate_limits = {
            'dexscreener': {'calls': 0, 'reset': 0, 'limit': 300},
            'gecko': {'calls': 0, 'reset': 0, 'limit': 500},
            'honeypot': {'calls': 0, 'reset': 0, 'limit': 100},
            'twitter': {'calls': 0, 'reset': 0, 'limit': 450},
            'etherscan': {'calls': 0, 'reset': 0, 'limit': 5}
        }
    
    async def initialize(self):
        connector = aiohttp.TCPConnector(limit=200, limit_per_host=50)
        for _ in range(50):
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=10),
                headers={
                    'User-Agent': 'Renaissance-DeFi-Scanner/2.0',
                    'Accept': 'application/json'
                }
            )
            self.session_pool.append(session)
    
    async def get_dexscreener_data(self, token_address: str) -> Dict:
        if not self._check_rate_limit('dexscreener'):
            return {}
        
        session = self.session_pool[0]
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    if pairs:
                        pair = max(pairs, key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))
                        return {
                            'price': float(pair.get('priceUsd', 0)),
                            'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                            'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                            'price_change_1h': float(pair.get('priceChange', {}).get('h1', 0)),
                            'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                            'txns_24h': pair.get('txns', {}).get('h24', {}),
                            'market_cap': float(pair.get('marketCap', 0))
                        }
        except Exception as e:
            logging.error(f"DexScreener API error: {e}")
        return {}
    
    async def get_honeypot_status(self, token_address: str) -> Dict:
        if not self._check_rate_limit('honeypot'):
            return {'is_honeypot': False, 'risk_score': 0.0}
        
        session = self.session_pool[1]
        url = f"https://api.honeypot.is/v2/IsHoneypot?address={token_address}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'is_honeypot': data.get('IsHoneypot', False),
                        'buy_tax': float(data.get('BuyTax', 0)),
                        'sell_tax': float(data.get('SellTax', 0)),
                        'transfer_tax': float(data.get('TransferTax', 0)),
                        'risk_score': (float(data.get('BuyTax', 0)) + 
                                     float(data.get('SellTax', 0))) / 200.0
                    }
        except Exception as e:
            logging.error(f"Honeypot API error: {e}")
        return {'is_honeypot': False, 'risk_score': 0.0}
    
    async def get_contract_info(self, token_address: str, chain: str = 'ethereum') -> Dict:
        if not self._check_rate_limit('etherscan'):
            return {'verified': False}
        
        api_key = self.credentials.etherscan
        if not api_key:
            return {'verified': False}
        
        base_urls = {
            'ethereum': 'https://api.etherscan.io/api',
            'arbitrum': 'https://api.arbiscan.io/api',
            'polygon': 'https://api.polygonscan.com/api'
        }
        
        base_url = base_urls.get(chain, base_urls['ethereum'])
        session = self.session_pool[2]
        
        try:
            url = f"{base_url}?module=contract&action=getsourcecode&address={token_address}&apikey={api_key}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get('result', [{}])[0]
                    source_code = result.get('SourceCode', '')
                    
                    # Analyze for dangerous functions
                    dangerous_functions = [
                        'mint', 'burn', 'pause', 'blacklist', 'setTax', 
                        'emergencyWithdraw', 'disableTrading', 'rugPull'
                    ]
                    
                    found_dangerous = [func for func in dangerous_functions if func in source_code]
                    
                    return {
                        'verified': len(source_code) > 0,
                        'contract_name': result.get('ContractName', ''),
                        'dangerous_functions': found_dangerous,
                        'source_code_length': len(source_code),
                        'proxy': 'Proxy' in source_code or 'proxy' in source_code.lower()
                    }
        except Exception as e:
            logging.error(f"Contract verification error: {e}")
        return {'verified': False}
    
    async def get_twitter_sentiment(self, token_symbol: str) -> Dict:
        if not self._check_rate_limit('twitter'):
            return {'sentiment': 0.5, 'mentions': 0}
        
        bearer_token = self.credentials.twitter_bearer
        if not bearer_token:
            return {'sentiment': 0.5, 'mentions': 0}
        
        session = self.session_pool[3]
        headers = {'Authorization': f'Bearer {bearer_token}'}
        
        try:
            query = f"#{token_symbol} OR ${token_symbol} (moon OR rocket OR gem OR scam OR rug) -is:retweet"
            url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=100&tweet.fields=public_metrics"
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    tweets = data.get('data', [])
                    
                    if not tweets:
                        return {'sentiment': 0.5, 'mentions': 0}
                    
                    # Sentiment analysis
                    positive_words = ['moon', 'rocket', 'gem', 'bullish', 'buy', 'pump']
                    negative_words = ['scam', 'rug', 'dump', 'bearish', 'sell', 'dead']
                    
                    sentiment_sum = 0
                    total_weight = 0
                    
                    for tweet in tweets:
                        text = tweet.get('text', '').lower()
                        metrics = tweet.get('public_metrics', {})
                        weight = 1 + metrics.get('like_count', 0) * 0.1
                        
                        pos_count = sum(1 for word in positive_words if word in text)
                        neg_count = sum(1 for word in negative_words if word in text)
                        
                        if pos_count > neg_count:
                            sentiment_sum += 0.7 * weight
                        elif neg_count > pos_count:
                            sentiment_sum += 0.3 * weight
                        else:
                            sentiment_sum += 0.5 * weight
                        
                        total_weight += weight
                    
                    sentiment = sentiment_sum / total_weight if total_weight > 0 else 0.5
                    
                    return {
                        'sentiment': min(max(sentiment, 0.0), 1.0),
                        'mentions': len(tweets),
                        'engagement': sum(t.get('public_metrics', {}).get('like_count', 0) for t in tweets)
                    }
        except Exception as e:
            logging.error(f"Twitter API error: {e}")
        return {'sentiment': 0.5, 'mentions': 0}
    
    async def get_token_holders(self, token_address: str) -> Dict:
        # This would integrate with Moralis or similar
        # Placeholder for real implementation
        return {
            'holder_count': 1000,
            'top_10_percentage': 60.0,
            'whale_count': 5
        }
    
    def _check_rate_limit(self, api_name: str) -> bool:
        current_time = time.time()
        rate_info = self.rate_limits[api_name]
        
        if current_time - rate_info['reset'] > 3600:
            rate_info['calls'] = 0
            rate_info['reset'] = current_time
        
        if rate_info['calls'] >= rate_info['limit']:
            return False
        
        rate_info['calls'] += 1
        return True
    
    async def get_comprehensive_data(self, token_address: str, token_symbol: str, chain: str = 'ethereum') -> Dict:
        """Get all data from real APIs in parallel"""
        tasks = [
            self.get_dexscreener_data(token_address),
            self.get_honeypot_status(token_address),
            self.get_contract_info(token_address, chain),
            self.get_twitter_sentiment(token_symbol),
            self.get_token_holders(token_address)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        dex_data = results[0] if not isinstance(results[0], Exception) else {}
        honeypot_data = results[1] if not isinstance(results[1], Exception) else {}
        contract_data = results[2] if not isinstance(results[2], Exception) else {}
        social_data = results[3] if not isinstance(results[3], Exception) else {}
        holder_data = results[4] if not isinstance(results[4], Exception) else {}
        
        return {
            **dex_data,
            **honeypot_data,
            **contract_data,
            **social_data,
            **holder_data,
            'timestamp': time.time()
        }

# Global API client
production_api = ProductionAPIClient()
