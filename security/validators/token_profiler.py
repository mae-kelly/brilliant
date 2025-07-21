import aiohttp
import logging
import sqlite3
import json
import time
import asyncio
from web3 import Web3
from abi import ERC20_ABI
import os
import yaml

class TokenProfiler:
    def __init__(self, chains):
        self.chains = chains
        self.conn = sqlite3.connect('token_cache.db')
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)
        self.etherscan_endpoints = {
            'arbitrum': "https://api.arbiscan.io/api",
            'polygon': "https://api.polygonscan.com/api", 
            'optimism': "https://api-optimistic.etherscan.io/api"
        }

    async def profile_token(self, chain, token_address):
        try:
            w3 = self.chains[chain]
            
            cached_profile = self.get_cached_profile(token_address)
            if cached_profile and not self.is_cache_expired(cached_profile):
                return cached_profile
            
            profile_tasks = [
                self.get_basic_token_info(w3, token_address),
                self.check_contract_verification(chain, token_address),
                self.get_liquidity_metrics(chain, token_address),
                self.analyze_trading_activity(chain, token_address)
            ]
            
            results = await asyncio.gather(*profile_tasks, return_exceptions=True)
            
            profile = {}
            for result in results:
                if isinstance(result, dict):
                    profile.update(result)
                elif isinstance(result, Exception):
                    logging.error(json.dumps({
                        'event': 'profile_component_error',
                        'chain': chain,
                        'token': token_address,
                        'error': str(result)
                    }))
            
            profile['chain'] = chain
            profile['token_address'] = token_address
            profile['last_updated'] = int(time.time())
            profile['blacklisted'] = profile.get('blacklisted', False)
            
            self.cache_profile(token_address, profile)
            
            return profile
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'token_profile_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {
                'blacklisted': True,
                'error': str(e),
                'last_updated': int(time.time())
            }

    async def get_basic_token_info(self, w3, token_address):
        try:
            token_contract = w3.eth.contract(address=token_address, abi=ERC20_ABI)
            
            info_calls = [
                ('symbol', token_contract.functions.symbol()),
                ('name', token_contract.functions.name()),
                ('decimals', token_contract.functions.decimals()),
                ('totalSupply', token_contract.functions.totalSupply())
            ]
            
            basic_info = {}
            for info_name, call_func in info_calls:
                try:
                    if info_name == 'totalSupply':
                        decimals = basic_info.get('decimals', 18)
                        raw_supply = call_func.call()
                        basic_info[info_name] = raw_supply / (10 ** decimals)
                    else:
                        basic_info[info_name] = call_func.call()
                except Exception as e:
                    logging.warning(f"Failed to get {info_name} for {token_address}: {e}")
                    basic_info[info_name] = None
            
            contract_code = w3.eth.get_code(token_address)
            basic_info['contract_size'] = len(contract_code)
            basic_info['is_contract'] = len(contract_code) > 0
            
            if basic_info['contract_size'] < 100:
                basic_info['blacklisted'] = True
                basic_info['blacklist_reason'] = 'Contract too small'
            
            return basic_info
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'basic_token_info_error',
                'token': token_address,
                'error': str(e)
            }))
            return {
                'blacklisted': True,
                'blacklist_reason': f'Failed to get basic info: {str(e)}'
            }

    async def check_contract_verification(self, chain, token_address):
        try:
            endpoint = self.etherscan_endpoints.get(chain)
            if not endpoint:
                return {'verified': False, 'verification_unknown': True}
            
            async with aiohttp.ClientSession() as session:
                params = {
                    'module': 'contract',
                    'action': 'getsourcecode',
                    'address': token_address,
                    'apikey': os.getenv('ETHERSCAN_API_KEY', 'demo')
                }
                
                async with session.get(endpoint, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get('result', [{}])[0]
                        
                        source_code = result.get('SourceCode', '')
                        contract_name = result.get('ContractName', '')
                        compiler_version = result.get('CompilerVersion', '')
                        
                        verification_info = {
                            'verified': bool(source_code and source_code != ''),
                            'contract_name': contract_name,
                            'compiler_version': compiler_version,
                            'source_code_length': len(source_code) if source_code else 0
                        }
                        
                        if verification_info['verified']:
                            verification_info.update(self.analyze_source_code(source_code))
                        
                        return verification_info
                    else:
                        return {'verified': False, 'verification_error': f'API error: {resp.status}'}
                        
        except Exception as e:
            logging.error(json.dumps({
                'event': 'verification_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {'verified': False, 'verification_error': str(e)}

    def analyze_source_code(self, source_code):
        try:
            source_lower = source_code.lower()
            
            suspicious_patterns = {
                'has_mint_function': any(pattern in source_lower for pattern in ['mint(', '_mint(', 'mint ']),
                'has_burn_function': any(pattern in source_lower for pattern in ['burn(', '_burn(', 'burn ']),
                'has_pause_function': any(pattern in source_lower for pattern in ['pause(', '_pause(', 'pause ']),
                'has_blacklist': any(pattern in source_lower for pattern in ['blacklist', '_blacklisted', 'isblacklisted']),
                'has_owner_functions': any(pattern in source_lower for pattern in ['onlyowner', 'owner()', '_owner']),
                'has_proxy_pattern': any(pattern in source_lower for pattern in ['proxy', 'implementation', 'delegate']),
                'has_selfdestruct': 'selfdestruct' in source_lower,
                'has_external_calls': any(pattern in source_lower for pattern in ['call(', 'delegatecall', 'staticcall'])
            }
            
            security_flags = {
                'uses_safemath': any(pattern in source_lower for pattern in ['safemath', 'safeerc20']),
                'has_reentrancy_guard': any(pattern in source_lower for pattern in ['reentrancyguard', 'nonreentrant']),
                'uses_openzeppelin': 'openzeppelin' in source_lower,
                'has_access_control': any(pattern in source_lower for pattern in ['accesscontrol', 'role', 'permission'])
            }
            
            code_quality = {
                'line_count': source_code.count('\n'),
                'function_count': source_code.count('function '),
                'comment_ratio': source_code.count('//') / max(source_code.count('\n'), 1)
            }
            
            risk_score = sum(suspicious_patterns.values()) * 0.1
            security_score = sum(security_flags.values()) * 0.2
            
            analysis = {
                'suspicious_patterns': suspicious_patterns,
                'security_flags': security_flags,
                'code_quality': code_quality,
                'risk_score': risk_score,
                'security_score': security_score,
                'overall_safety': security_score - risk_score
            }
            
            if risk_score > 0.5:
                analysis['blacklisted'] = True
                analysis['blacklist_reason'] = 'High risk patterns in source code'
            
            return analysis
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'source_code_analysis_error',
                'error': str(e)
            }))
            return {'analysis_error': str(e)}

    async def get_liquidity_metrics(self, chain, token_address):
        try:
            w3 = self.chains[chain]
            
            from abi import UNISWAP_V3_POOL_ABI
            
            try:
                pool_contract = w3.eth.contract(address=token_address, abi=UNISWAP_V3_POOL_ABI)
                liquidity = pool_contract.functions.liquidity().call()
                slot0 = pool_contract.functions.slot0().call()
                
                liquidity_metrics = {
                    'current_liquidity': liquidity,
                    'sqrt_price': slot0[0],
                    'current_tick': slot0[1],
                    'fee_tier': pool_contract.functions.fee().call(),
                    'is_pool': True
                }
                
                price = (slot0[0] / 2**96) ** 2
                liquidity_metrics['current_price'] = price
                
                if liquidity < self.settings['safety']['min_liquidity']:
                    liquidity_metrics['blacklisted'] = True
                    liquidity_metrics['blacklist_reason'] = f'Insufficient liquidity: {liquidity}'
                
            except:
                liquidity_metrics = {
                    'is_pool': False,
                    'current_liquidity': 0,
                    'liquidity_check_failed': True
                }
            
            return liquidity_metrics
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'liquidity_metrics_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {'liquidity_error': str(e)}

    async def analyze_trading_activity(self, chain, token_address):
        try:
            endpoint = self.etherscan_endpoints.get(chain)
            if not endpoint:
                return {'trading_activity_unknown': True}
            
            async with aiohttp.ClientSession() as session:
                params = {
                    'module': 'account',
                    'action': 'tokentx',
                    'contractaddress': token_address,
                    'page': 1,
                    'offset': 100,
                    'sort': 'desc',
                    'apikey': os.getenv('ETHERSCAN_API_KEY', 'demo')
                }
                
                async with session.get(endpoint, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        transactions = data.get('result', [])
                        
                        if isinstance(transactions, str):
                            return {'trading_activity': 0, 'recent_transactions': 0}
                        
                        current_time = int(time.time())
                        recent_txs = [tx for tx in transactions 
                                    if current_time - int(tx.get('timeStamp', 0)) < 3600]
                        
                        activity_metrics = {
                            'total_transactions': len(transactions),
                            'recent_transactions': len(recent_txs),
                            'unique_addresses': len(set(tx.get('from', '') for tx in transactions)),
                            'avg_tx_value': self.calculate_avg_tx_value(transactions),
                            'trading_velocity': len(recent_txs) / max(1, len(transactions))
                        }
                        
                        if len(transactions) < 10:
                            activity_metrics['low_activity_warning'] = True
                        
                        if len(recent_txs) == 0 and len(transactions) > 50:
                            activity_metrics['potentially_dead'] = True
                            activity_metrics['blacklisted'] = True
                            activity_metrics['blacklist_reason'] = 'No recent trading activity'
                        
                        return activity_metrics
                    else:
                        return {'trading_activity_error': f'API error: {resp.status}'}
                        
        except Exception as e:
            logging.error(json.dumps({
                'event': 'trading_activity_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {'trading_activity_error': str(e)}

    def calculate_avg_tx_value(self, transactions):
        try:
            if not transactions:
                return 0
            
            values = []
            for tx in transactions:
                try:
                    value = float(tx.get('value', 0))
                    decimals = int(tx.get('tokenDecimal', 18))
                    normalized_value = value / (10 ** decimals)
                    values.append(normalized_value)
                except:
                    continue
            
            return sum(values) / len(values) if values else 0
            
        except:
            return 0

    def get_cached_profile(self, token_address):
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT profile_data, last_updated 
                FROM token_profiles 
                WHERE address = ?
            """, (token_address,))
            
            result = cursor.fetchone()
            if result:
                profile_data, last_updated = result
                profile = json.loads(profile_data)
                profile['last_updated'] = last_updated
                return profile
            
            return None
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'cache_retrieval_error',
                'token': token_address,
                'error': str(e)
            }))
            return None

    def cache_profile(self, token_address, profile):
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_profiles (
                    address TEXT PRIMARY KEY,
                    profile_data TEXT,
                    last_updated INTEGER,
                    blacklisted BOOLEAN
                )
            """)
            
            profile_json = json.dumps(profile)
            
            cursor.execute("""
                INSERT OR REPLACE INTO token_profiles 
                (address, profile_data, last_updated, blacklisted)
                VALUES (?, ?, ?, ?)
            """, (token_address, profile_json, profile['last_updated'], 
                  profile.get('blacklisted', False)))
            
            self.conn.commit()
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'cache_storage_error',
                'token': token_address,
                'error': str(e)
            }))

    def is_cache_expired(self, profile, max_age=1800):
        try:
            last_updated = profile.get('last_updated', 0)
            current_time = int(time.time())
            return (current_time - last_updated) > max_age
        except:
            return True

    async def blacklist_token(self, token_address, reason=None):
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                UPDATE tokens SET blacklisted = ?, last_updated = ? 
                WHERE address = ?
            """, (True, int(time.time()), token_address))
            
            cursor.execute("""
                INSERT OR REPLACE INTO token_profiles 
                (address, profile_data, last_updated, blacklisted)
                VALUES (?, ?, ?, ?)
            """, (token_address, 
                  json.dumps({
                      'blacklisted': True, 
                      'blacklist_reason': reason or 'Manual blacklist',
                      'last_updated': int(time.time())
                  }), 
                  int(time.time()), True))
            
            self.conn.commit()
            
            logging.info(json.dumps({
                'event': 'token_blacklisted',
                'token': token_address,
                'reason': reason
            }))
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'blacklist_error',
                'token': token_address,
                'error': str(e)
            }))

    async def whitelist_token(self, token_address):
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                UPDATE tokens SET blacklisted = ?, last_updated = ? 
                WHERE address = ?
            """, (False, int(time.time()), token_address))
            
            cursor.execute("""
                UPDATE token_profiles SET blacklisted = ? 
                WHERE address = ?
            """, (False, token_address))
            
            self.conn.commit()
            
            logging.info(json.dumps({
                'event': 'token_whitelisted',
                'token': token_address
            }))
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'whitelist_error',
                'token': token_address,
                'error': str(e)
            }))

    def get_blacklist_summary(self):
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as total_blacklisted
                FROM token_profiles 
                WHERE blacklisted = 1
            """)
            total_blacklisted = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT profile_data
                FROM token_profiles 
                WHERE blacklisted = 1 
                ORDER BY last_updated DESC 
                LIMIT 10
            """)
            
            recent_blacklisted = []
            for row in cursor.fetchall():
                try:
                    profile = json.loads(row[0])
                    recent_blacklisted.append({
                        'token': profile.get('token_address', 'unknown'),
                        'reason': profile.get('blacklist_reason', 'unknown'),
                        'timestamp': profile.get('last_updated', 0)
                    })
                except:
                    continue
            
            return {
                'total_blacklisted': total_blacklisted,
                'recent_blacklisted': recent_blacklisted
            }
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'blacklist_summary_error',
                'error': str(e)
            }))
            return {'total_blacklisted': 0, 'recent_blacklisted': []}

    def cleanup_old_profiles(self, max_age_days=7):
        try:
            cursor = self.conn.cursor()
            cutoff_time = int(time.time()) - (max_age_days * 24 * 3600)
            
            cursor.execute("""
                DELETE FROM token_profiles 
                WHERE last_updated < ? AND blacklisted = 0
            """, (cutoff_time,))
            
            deleted_count = cursor.rowcount
            self.conn.commit()
            
            logging.info(json.dumps({
                'event': 'profile_cleanup_completed',
                'deleted_profiles': deleted_count,
                'cutoff_time': cutoff_time
            }))
            
            return deleted_count
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'profile_cleanup_error',
                'error': str(e)
            }))
            return 0

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass