import aiohttp
import asyncio
import logging
import json
import numpy as np
import pandas as pd
from web3 import Web3
import yaml
import os
from prometheus_client import Gauge, Counter

class MempoolWatcher:
    def __init__(self, chains):
        self.chains = chains
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)
        self.blocknative_api = "https://api.blocknative.com/v0"
        self.flashbots_api = "https://relay.flashbots.net"
        self.mev_protection_gauge = Gauge('mev_protection_active', 'MEV protection status', ['chain'])
        self.frontrun_detected_counter = Counter('frontrun_attempts_detected', 'Frontrunning attempts detected', ['chain'])
        self.sandwich_detected_counter = Counter('sandwich_attacks_detected', 'Sandwich attacks detected', ['chain'])

    async def check_mempool(self, chain, token_address):
        try:
            tasks = [
                self.detect_frontrunning(chain, token_address),
                self.detect_sandwich_attacks(chain, token_address),
                self.check_mev_activity(chain, token_address),
                self.analyze_gas_competition(chain, token_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            frontrun_safe = results[0] if isinstance(results[0], bool) else False
            sandwich_safe = results[1] if isinstance(results[1], bool) else False
            mev_safe = results[2] if isinstance(results[2], bool) else False
            gas_safe = results[3] if isinstance(results[3], bool) else False
            
            overall_safe = all([frontrun_safe, sandwich_safe, mev_safe, gas_safe])
            
            if overall_safe:
                self.mev_protection_gauge.labels(chain=chain).set(1)
            else:
                self.mev_protection_gauge.labels(chain=chain).set(0)
                
            return overall_safe
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'mempool_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            self.mev_protection_gauge.labels(chain=chain).set(0)
            return False

    async def detect_frontrunning(self, chain, token_address):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f"Bearer {os.getenv('BLOCKNATIVE_API_KEY')}"}
                url = f"{self.blocknative_api}/mempool"
                
                params = {
                    'address': token_address,
                    'network': self.get_network_id(chain),
                    'status': 'pending'
                }
                
                async with session.get(url, headers=headers, params=params, timeout=5) as resp:
                    if resp.status != 200:
                        return True
                    
                    data = await resp.json()
                    pending_txs = data.get('transactions', [])
                    
                    if len(pending_txs) > self.settings['safety']['max_pending_txs']:
                        self.frontrun_detected_counter.labels(chain=chain).inc()
                        return False
                    
                    frontrun_indicators = self.analyze_frontrun_patterns(pending_txs)
                    
                    if frontrun_indicators > 0.7:
                        self.frontrun_detected_counter.labels(chain=chain).inc()
                        logging.warning(json.dumps({
                            'event': 'frontrun_detected',
                            'chain': chain,
                            'token': token_address,
                            'risk_score': frontrun_indicators
                        }))
                        return False
                    
                    return True
                    
        except Exception as e:
            logging.error(json.dumps({
                'event': 'frontrun_detection_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return True

    def analyze_frontrun_patterns(self, pending_txs):
        if not pending_txs:
            return 0.0
        
        try:
            gas_prices = [int(tx.get('gasPrice', '0'), 16) for tx in pending_txs if tx.get('gasPrice')]
            if not gas_prices:
                return 0.0
            
            gas_prices = np.array(gas_prices)
            gas_percentile_95 = np.percentile(gas_prices, 95)
            gas_mean = np.mean(gas_prices)
            
            high_gas_txs = sum(1 for price in gas_prices if price > gas_percentile_95)
            gas_competition_ratio = high_gas_txs / len(gas_prices)
            
            gas_spike_ratio = gas_percentile_95 / (gas_mean + 1)
            
            duplicate_functions = self.detect_duplicate_function_calls(pending_txs)
            
            risk_score = (gas_competition_ratio * 0.4 + 
                         min(gas_spike_ratio / 10, 1) * 0.4 + 
                         duplicate_functions * 0.2)
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'frontrun_analysis_error',
                'error': str(e)
            }))
            return 0.0

    def detect_duplicate_function_calls(self, pending_txs):
        try:
            function_signatures = {}
            for tx in pending_txs:
                input_data = tx.get('input', '')
                if len(input_data) >= 10:
                    func_sig = input_data[:10]
                    function_signatures[func_sig] = function_signatures.get(func_sig, 0) + 1
            
            if not function_signatures:
                return 0.0
            
            max_duplicates = max(function_signatures.values())
            return min(max_duplicates / 10, 1.0)
            
        except:
            return 0.0

    async def detect_sandwich_attacks(self, chain, token_address):
        try:
            w3 = self.chains[chain]
            latest_block = w3.eth.get_block('latest', full_transactions=True)
            
            recent_blocks = []
            for i in range(5):
                block_num = latest_block['number'] - i
                if block_num > 0:
                    block = w3.eth.get_block(block_num, full_transactions=True)
                    recent_blocks.append(block)
            
            sandwich_patterns = self.analyze_sandwich_patterns(recent_blocks, token_address)
            
            if sandwich_patterns > 0.5:
                self.sandwich_detected_counter.labels(chain=chain).inc()
                logging.warning(json.dumps({
                    'event': 'sandwich_attack_detected',
                    'chain': chain,
                    'token': token_address,
                    'pattern_score': sandwich_patterns
                }))
                return False
            
            return True
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'sandwich_detection_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return True

    def analyze_sandwich_patterns(self, blocks, token_address):
        try:
            suspicious_sequences = 0
            total_sequences = 0
            
            for block in blocks:
                transactions = block.get('transactions', [])
                token_txs = [tx for tx in transactions 
                           if token_address.lower() in str(tx.get('input', '')).lower()]
                
                if len(token_txs) < 3:
                    continue
                
                for i in range(len(token_txs) - 2):
                    sequence = token_txs[i:i+3]
                    if self.is_sandwich_sequence(sequence):
                        suspicious_sequences += 1
                    total_sequences += 1
            
            return suspicious_sequences / max(total_sequences, 1)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'sandwich_analysis_error',
                'error': str(e)
            }))
            return 0.0

    def is_sandwich_sequence(self, sequence):
        try:
            if len(sequence) != 3:
                return False
            
            gas_prices = [int(tx.get('gasPrice', '0'), 16) for tx in sequence]
            values = [int(tx.get('value', '0'), 16) for tx in sequence]
            
            high_gas_bookends = (gas_prices[0] > gas_prices[1] and 
                               gas_prices[2] > gas_prices[1])
            
            similar_values = (abs(values[0] - values[2]) / max(values[0], values[2], 1) < 0.1 
                            if values[0] > 0 and values[2] > 0 else False)
            
            return high_gas_bookends and similar_values
            
        except:
            return False

    async def check_mev_activity(self, chain, token_address):
        try:
            w3 = self.chains[chain]
            latest_block = w3.eth.get_block('latest', full_transactions=True)
            
            mev_indicators = {
                'flashloan_usage': 0,
                'arbitrage_patterns': 0,
                'liquidation_activity': 0,
                'atomic_swaps': 0
            }
            
            for tx in latest_block.get('transactions', []):
                input_data = tx.get('input', '')
                
                if self.detect_flashloan(input_data):
                    mev_indicators['flashloan_usage'] += 1
                
                if self.detect_arbitrage(input_data, token_address):
                    mev_indicators['arbitrage_patterns'] += 1
                
                if self.detect_liquidation(input_data):
                    mev_indicators['liquidation_activity'] += 1
                
                if self.detect_atomic_swap(input_data, token_address):
                    mev_indicators['atomic_swaps'] += 1
            
            total_mev_activity = sum(mev_indicators.values())
            mev_threshold = 5
            
            return total_mev_activity < mev_threshold
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'mev_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return True

    def detect_flashloan(self, input_data):
        flashloan_signatures = [
            '0x5cffe9de',  
            '0x1b5b6b52',  
            '0x8b5a8bfb'   
        ]
        return any(sig in input_data.lower() for sig in flashloan_signatures)

    def detect_arbitrage(self, input_data, token_address):
        try:
            token_occurrences = input_data.lower().count(token_address.lower()[2:])
            return token_occurrences > 3
        except:
            return False

    def detect_liquidation(self, input_data):
        liquidation_signatures = [
            '0x96cd4ddb',  
            '0x5c778605',  
            '0x7c025200'   
        ]
        return any(sig in input_data.lower() for sig in liquidation_signatures)

    def detect_atomic_swap(self, input_data, token_address):
        try:
            multicall_signatures = ['0xac9650d8', '0x1f00ca74']
            has_multicall = any(sig in input_data.lower() for sig in multicall_signatures)
            has_token = token_address.lower()[2:] in input_data.lower()
            return has_multicall and has_token
        except:
            return False

    async def analyze_gas_competition(self, chain, token_address):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f"Bearer {os.getenv('BLOCKNATIVE_API_KEY')}"}
                url = f"{self.blocknative_api}/gasprices"
                
                params = {'network': self.get_network_id(chain)}
                
                async with session.get(url, headers=headers, params=params, timeout=5) as resp:
                    if resp.status != 200:
                        return True
                    
                    data = await resp.json()
                    gas_prices = data.get('blockPrices', [{}])[0].get('estimatedPrices', [])
                    
                    if not gas_prices:
                        return True
                    
                    fastest_gas = next((p['price'] for p in gas_prices 
                                      if p['confidence'] == 99), 50)
                    standard_gas = next((p['price'] for p in gas_prices 
                                       if p['confidence'] == 95), 25)
                    
                    gas_premium = (fastest_gas - standard_gas) / standard_gas
                    
                    return gas_premium < 3.0
                    
        except Exception as e:
            logging.error(json.dumps({
                'event': 'gas_competition_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return True

    async def get_private_mempool_data(self, chain):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f"Bearer {os.getenv('FLASHBOTS_AUTH_KEY')}",
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'jsonrpc': '2.0',
                    'method': 'eth_getBlockByNumber',
                    'params': ['pending', True],
                    'id': 1
                }
                
                flashbots_url = f"{self.flashbots_api}/{chain}"
                
                async with session.post(flashbots_url, headers=headers, 
                                      json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('result', {}).get('transactions', [])
                    
            return []
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'private_mempool_error',
                'chain': chain,
                'error': str(e)
            }))
            return []

    def get_network_id(self, chain):
        network_mapping = {
            'arbitrum': 42161,
            'polygon': 137,
            'optimism': 10,
            'ethereum': 1
        }
        return network_mapping.get(chain, 1)

    async def estimate_mev_risk(self, chain, token_address, trade_size):
        try:
            mempool_safe = await self.check_mempool(chain, token_address)
            private_txs = await self.get_private_mempool_data(chain)
            
            competing_trades = sum(1 for tx in private_txs 
                                 if token_address.lower() in str(tx.get('input', '')).lower())
            
            risk_factors = {
                'mempool_safety': 1.0 if mempool_safe else 0.0,
                'competing_trades': max(0, 1 - competing_trades / 10),
                'trade_size_impact': max(0, 1 - trade_size / 10)
            }
            
            overall_risk = 1 - np.mean(list(risk_factors.values()))
            
            return {
                'safe_to_trade': overall_risk < 0.3,
                'risk_score': overall_risk,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'mev_risk_estimation_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return {
                'safe_to_trade': False,
                'risk_score': 1.0,
                'risk_factors': {}
            }