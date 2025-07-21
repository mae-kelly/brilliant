from web3 import Web3
import aiohttp
import logging
from abi import abi, ERC20_ABI
import json
import asyncio
import yaml
import os
import time
from prometheus_client import Gauge, Counter

class RugpullAnalyzer:
    def __init__(self, chains):
        self.chains = chains
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)
        self.lp_lock_api = "https://api.unicrypt.network/v2/lock"
        self.dextools_api = "https://api.dextools.io/v1"
        self.rugpull_gauge = Gauge('rugpull_risk_score', 'Rugpull risk assessment', ['chain', 'token'])
        self.protection_counter = Counter('rugpull_protections_triggered', 'Rugpull protections activated', ['type'])

    async def analyze_token(self, chain, token_address):
        try:
            analysis_results = await asyncio.gather(
                self.check_lp_lock(chain, token_address),
                self.check_pause_function(chain, token_address),
                self.check_ownership(chain, token_address),
                self.check_mint_function(chain, token_address),
                self.analyze_holder_distribution(chain, token_address),
                self.check_contract_verification(chain, token_address),
                self.analyze_liquidity_stability(chain, token_address),
                return_exceptions=True
            )
            
            risk_factors = {}
            for i, result in enumerate(analysis_results):
                if isinstance(result, dict):
                    risk_factors.update(result)
                elif isinstance(result, Exception):
                    logging.error(json.dumps({
                        'event': 'rugpull_analysis_error',
                        'chain': chain,
                        'token': token_address,
                        'check_index': i,
                        'error': str(result)
                    }))
            
            overall_risk = self.calculate_overall_risk(risk_factors)
            self.rugpull_gauge.labels(chain=chain, token=token_address).set(overall_risk)
            
            is_safe = overall_risk < 0.3
            
            if not is_safe:
                self.protection_counter.labels(type='high_risk_detected').inc()
                
            logging.info(json.dumps({
                'event': 'rugpull_analysis_complete',
                'chain': chain,
                'token': token_address,
                'risk_score': overall_risk,
                'is_safe': is_safe,
                'risk_factors': risk_factors
            }))
            
            return is_safe
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'rugpull_analyzer_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return False

    async def check_lp_lock(self, chain, token_address):
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.lp_lock_api}/locks"
                params = {
                    'token': token_address,
                    'chain': self.get_chain_id(chain)
                }
                
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        locks = data.get('data', [])
                        
                        total_locked = 0
                        lock_duration = 0
                        
                        for lock in locks:
                            if lock.get('token', '').lower() == token_address.lower():
                                total_locked += float(lock.get('amount', 0))
                                unlock_time = int(lock.get('unlock_time', 0))
                                current_time = int(time.time())
                                duration = max(0, unlock_time - current_time)
                                lock_duration = max(lock_duration, duration)
                        
                        liquidity_locked_ratio = min(total_locked / 1000000, 1.0)
                        time_factor = min(lock_duration / (365 * 24 * 3600), 1.0)
                        
                        lp_lock_score = (liquidity_locked_ratio * 0.7 + time_factor * 0.3)
                        
                        return {'lp_lock_risk': 1 - lp_lock_score}
                    else:
                        return {'lp_lock_risk': 0.8}
                        
        except Exception as e:
            logging.error(json.dumps({
                'event': 'lp_lock_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {'lp_lock_risk': 0.7}

    async def check_pause_function(self, chain, token_address):
        try:
            w3 = self.chains[chain]
            contract_code = w3.eth.get_code(token_address)
            
            if len(contract_code) < 10:
                return {'pause_function_risk': 1.0}
            
            pause_signatures = [
                b'\x8d\xa5\xcb[\x00',  # pause()
                b';\xbc\xe0\x05\x00',  # _pause()
                b'\x02\xfe\x05\x31\x00', # emergencyStop()
                b'\x54\xfd\x4d\x50\x00'  # pauseContract()
            ]
            
            malicious_patterns = [
                b'\x31\x1f\x91\xc7\x00',  # onlyOwner with pause
                b'\x70\xa0\x82\x31\x00',  # blacklist function
                b'\xf2\xfb\xa2\xa2\x00',  # setMaxTransactionAmount
                b'\x44\x33\x70\xbb\x00'   # setTaxes
            ]
            
            code_hex = contract_code.hex().lower()
            
            pause_functions_found = sum(1 for sig in pause_signatures if sig.hex() in code_hex)
            malicious_functions_found = sum(1 for pattern in malicious_patterns if pattern.hex() in code_hex)
            
            pause_risk = min(pause_functions_found * 0.3, 1.0)
            malicious_risk = min(malicious_functions_found * 0.4, 1.0)
            
            total_risk = min(pause_risk + malicious_risk, 1.0)
            
            if total_risk > 0.5:
                self.protection_counter.labels(type='pause_function_detected').inc()
            
            return {'pause_function_risk': total_risk}
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'pause_function_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {'pause_function_risk': 0.5}

    async def check_ownership(self, chain, token_address):
        try:
            w3 = self.chains[chain]
            
            owner_abi = [{
                'constant': True,
                'inputs': [],
                'name': 'owner',
                'outputs': [{'name': '', 'type': 'address'}],
                'type': 'function'
            }]
            
            renounced_abi = [{
                'constant': True,
                'inputs': [],
                'name': 'renounceOwnership',
                'outputs': [],
                'type': 'function'
            }]
            
            try:
                contract = w3.eth.contract(address=token_address, abi=owner_abi)
                owner = contract.functions.owner().call()
                
                zero_address = '0x0000000000000000000000000000000000000000'
                dead_addresses = [
                    '0x000000000000000000000000000000000000dead',
                    '0x0000000000000000000000000000000000000001'
                ]
                
                if owner.lower() == zero_address or owner.lower() in [addr.lower() for addr in dead_addresses]:
                    ownership_risk = 0.0
                    self.protection_counter.labels(type='ownership_renounced').inc()
                else:
                    ownership_risk = 0.8
                    
            except:
                ownership_risk = 0.3
            
            contract_code = w3.eth.get_code(token_address)
            code_hex = contract_code.hex().lower()
            
            if 'renounceownership' in code_hex:
                ownership_risk *= 0.5
            
            return {'ownership_risk': ownership_risk}
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'ownership_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {'ownership_risk': 0.6}

    async def check_mint_function(self, chain, token_address):
        try:
            w3 = self.chains[chain]
            contract_code = w3.eth.get_code(token_address)
            code_hex = contract_code.hex().lower()
            
            mint_signatures = [
                'a0712d68',  # mint(uint256)
                '40c10f19',  # mint(address,uint256)
                '1249c58b',  # mintTo(address,uint256)
                '7d64bcb4'   # _mint(address,uint256)
            ]
            
            mint_functions_found = sum(1 for sig in mint_signatures if sig in code_hex)
            
            if mint_functions_found > 0:
                try:
                    total_supply_abi = [{
                        'constant': True,
                        'inputs': [],
                        'name': 'totalSupply',
                        'outputs': [{'name': '', 'type': 'uint256'}],
                        'type': 'function'
                    }]
                    
                    contract = w3.eth.contract(address=token_address, abi=total_supply_abi)
                    total_supply = contract.functions.totalSupply().call()
                    
                    max_supply_signatures = ['355274ea', '70a08231']
                    has_max_supply = any(sig in code_hex for sig in max_supply_signatures)
                    
                    if has_max_supply:
                        mint_risk = 0.3
                    else:
                        mint_risk = 0.7
                        
                    if total_supply == 0:
                        mint_risk = 1.0
                        
                except:
                    mint_risk = 0.6
            else:
                mint_risk = 0.0
            
            if mint_risk > 0.5:
                self.protection_counter.labels(type='mint_function_detected').inc()
            
            return {'mint_function_risk': mint_risk}
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'mint_function_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {'mint_function_risk': 0.4}

    async def analyze_holder_distribution(self, chain, token_address):
        try:
            async with aiohttp.ClientSession() as session:
                etherscan_endpoints = {
                    'arbitrum': f"https://api.arbiscan.io/api",
                    'polygon': f"https://api.polygonscan.com/api",
                    'optimism': f"https://api-optimistic.etherscan.io/api"
                }
                
                endpoint = etherscan_endpoints.get(chain)
                if not endpoint:
                    return {'holder_distribution_risk': 0.5}
                
                params = {
                    'module': 'token',
                    'action': 'tokenholderlist',
                    'contractaddress': token_address,
                    'page': 1,
                    'offset': 100,
                    'apikey': os.getenv('ETHERSCAN_API_KEY', 'demo')
                }
                
                async with session.get(endpoint, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        holders = data.get('result', [])
                        
                        if len(holders) < 10:
                            return {'holder_distribution_risk': 0.9}
                        
                        total_supply = sum(int(holder.get('TokenHolderQuantity', 0)) for holder in holders)
                        
                        if total_supply == 0:
                            return {'holder_distribution_risk': 1.0}
                        
                        top_10_holdings = sum(int(holder.get('TokenHolderQuantity', 0)) for holder in holders[:10])
                        concentration_ratio = top_10_holdings / total_supply
                        
                        dev_wallets = 0
                        for holder in holders[:20]:
                            balance = int(holder.get('TokenHolderQuantity', 0))
                            if balance > total_supply * 0.05:
                                dev_wallets += 1
                        
                        concentration_risk = min(concentration_ratio * 2, 1.0)
                        dev_wallet_risk = min(dev_wallets * 0.2, 1.0)
                        
                        distribution_risk = (concentration_risk + dev_wallet_risk) / 2
                        
                        if distribution_risk > 0.6:
                            self.protection_counter.labels(type='centralized_holdings').inc()
                        
                        return {'holder_distribution_risk': distribution_risk}
                    else:
                        return {'holder_distribution_risk': 0.5}
                        
        except Exception as e:
            logging.error(json.dumps({
                'event': 'holder_distribution_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {'holder_distribution_risk': 0.5}

    async def check_contract_verification(self, chain, token_address):
        try:
            async with aiohttp.ClientSession() as session:
                etherscan_endpoints = {
                    'arbitrum': f"https://api.arbiscan.io/api",
                    'polygon': f"https://api.polygonscan.com/api",
                    'optimism': f"https://api-optimistic.etherscan.io/api"
                }
                
                endpoint = etherscan_endpoints.get(chain)
                if not endpoint:
                    return {'verification_risk': 0.4}
                
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
                        
                        if not source_code or source_code == '':
                            verification_risk = 0.8
                            self.protection_counter.labels(type='unverified_contract').inc()
                        else:
                            verification_risk = 0.1
                            
                            suspicious_patterns = [
                                'selfdestruct',
                                'delegatecall',
                                'suicide',
                                'hidden',
                                'backdoor'
                            ]
                            
                            pattern_count = sum(1 for pattern in suspicious_patterns if pattern in source_code.lower())
                            verification_risk += min(pattern_count * 0.2, 0.5)
                        
                        return {'verification_risk': verification_risk}
                    else:
                        return {'verification_risk': 0.6}
                        
        except Exception as e:
            logging.error(json.dumps({
                'event': 'verification_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {'verification_risk': 0.5}

    async def analyze_liquidity_stability(self, chain, token_address):
        try:
            w3 = self.chains[chain]
            
            pool_abi = UNISWAP_V3_POOL_ABI
            current_block = w3.eth.block_number
            
            liquidity_history = []
            
            for i in range(5):
                try:
                    block_number = current_block - (i * 200)
                    
                    contract = w3.eth.contract(address=token_address, abi=pool_abi)
                    liquidity = contract.functions.liquidity().call(block_identifier=block_number)
                    liquidity_history.append(liquidity)
                    
                except:
                    continue
            
            if len(liquidity_history) < 2:
                return {'liquidity_stability_risk': 0.6}
            
            liquidity_changes = []
            for i in range(1, len(liquidity_history)):
                if liquidity_history[i-1] > 0:
                    change = abs(liquidity_history[i] - liquidity_history[i-1]) / liquidity_history[i-1]
                    liquidity_changes.append(change)
            
            if not liquidity_changes:
                return {'liquidity_stability_risk': 0.5}
            
            avg_volatility = sum(liquidity_changes) / len(liquidity_changes)
            stability_risk = min(avg_volatility * 5, 1.0)
            
            current_liquidity = liquidity_history[0]
            if current_liquidity < 100000:
                stability_risk = min(stability_risk + 0.4, 1.0)
            
            if stability_risk > 0.7:
                self.protection_counter.labels(type='unstable_liquidity').inc()
            
            return {'liquidity_stability_risk': stability_risk}
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'liquidity_stability_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {'liquidity_stability_risk': 0.5}

    def calculate_overall_risk(self, risk_factors):
        try:
            if not risk_factors:
                return 1.0
            
            weights = {
                'lp_lock_risk': 0.25,
                'pause_function_risk': 0.20,
                'ownership_risk': 0.15,
                'mint_function_risk': 0.15,
                'holder_distribution_risk': 0.15,
                'verification_risk': 0.05,
                'liquidity_stability_risk': 0.05
            }
            
            weighted_score = 0
            total_weight = 0
            
            for factor, risk_score in risk_factors.items():
                if factor in weights:
                    weight = weights[factor]
                    weighted_score += risk_score * weight
                    total_weight += weight
            
            if total_weight == 0:
                return 1.0
            
            final_score = weighted_score / total_weight
            
            if len(risk_factors) < 5:
                uncertainty_penalty = (5 - len(risk_factors)) * 0.1
                final_score = min(final_score + uncertainty_penalty, 1.0)
            
            return final_score
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'risk_calculation_error',
                'error': str(e)
            }))
            return 1.0

    def get_chain_id(self, chain):
        return {'arbitrum': 42161, 'polygon': 137, 'optimism': 10}.get(chain, 1)

    def get_risk_summary(self, chain, token_address):
        try:
            current_risk = self.rugpull_gauge.labels(chain=chain, token=token_address)._value._value
            
            risk_level = 'LOW'
            if current_risk > 0.6:
                risk_level = 'HIGH'
            elif current_risk > 0.3:
                risk_level = 'MEDIUM'
            
            return {
                'risk_score': current_risk,
                'risk_level': risk_level,
                'recommendation': 'AVOID' if risk_level == 'HIGH' else 'CAUTION' if risk_level == 'MEDIUM' else 'SAFE'
            }
            
        except:
            return {
                'risk_score': 1.0,
                'risk_level': 'UNKNOWN',
                'recommendation': 'AVOID'
            }