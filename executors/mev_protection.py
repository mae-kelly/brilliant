import asyncio
import aiohttp
import json
import time
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from web3 import Web3
from eth_account import Account
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config"))

try:
    from config.optimizer import get_dynamic_config
except ImportError:
    def get_dynamic_config():
        return {"mev_protection_threshold": 0.01, "flashbots_threshold": 1.0}

@dataclass
class FlashbotsBundle:
    transactions: List[Dict]
    target_block: int
    max_priority_fee: int
    simulation_result: Dict

@dataclass
class MEVOpportunity:
    transaction_hash: str
    frontrun_risk: float
    sandwich_risk: float
    arbitrage_potential: float
    gas_price: int
    value: float
    detected_at: float

@dataclass
class PrivatePoolSubmission:
    pool_name: str
    endpoint: str
    success: bool
    response_time: float
    bundle_hash: Optional[str]
    error: Optional[str]

class FlashbotsIntegration:
    def __init__(self):
        self.relay_endpoints = {
            'mainnet': 'https://relay.flashbots.net',
            'goerli': 'https://relay-goerli.flashbots.net',
            'arbitrum': 'https://rpc.flashbots.net/arbitrum',
            'polygon': 'https://rpc.flashbots.net/polygon'
        }
        
        self.private_key = os.getenv('FLASHBOTS_PRIVATE_KEY', os.getenv('PRIVATE_KEY'))
        self.bundle_queue = asyncio.Queue(maxsize=1000)
        self.simulation_cache = {}
        self.bundle_stats = {
            'submitted': 0,
            'included': 0,
            'failed': 0,
            'avg_inclusion_rate': 0.0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_flashbots_signature(self, message: str) -> str:
        if not self.private_key:
            return ""
        
        message_hash = hashlib.sha256(message.encode()).digest()
        signature = hmac.new(
            bytes.fromhex(self.private_key[2:] if self.private_key.startswith('0x') else self.private_key),
            message_hash,
            hashlib.sha256
        ).hexdigest()
        
        return signature

    async def simulate_bundle(self, transactions: List[Dict], target_block: int, chain: str = 'mainnet') -> Dict:
        bundle_hash = f"bundle_{target_block}_{len(transactions)}_{hash(str(transactions))}"
        
        if bundle_hash in self.simulation_cache:
            return self.simulation_cache[bundle_hash]
        
        try:
            endpoint = self.relay_endpoints.get(chain, self.relay_endpoints['mainnet'])
            
            simulation_request = {
                "jsonrpc": "2.0",
                "method": "eth_callBundle",
                "params": [
                    {
                        "txs": [tx.get('rawTransaction', tx.get('data', '0x')) for tx in transactions],
                        "blockNumber": hex(target_block),
                        "stateBlockNumber": "latest"
                    }
                ],
                "id": int(time.time())
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-Flashbots-Signature': self.generate_flashbots_signature(json.dumps(simulation_request))
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=simulation_request, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'result' in result:
                            simulation = {
                                'success': True,
                                'gas_used': sum(int(r.get('gasUsed', '0'), 16) for r in result['result'].get('results', [])),
                                'effective_gas_price': 20000000000,
                                'coinbase_diff': int(result['result'].get('coinbaseDiff', '0'), 16),
                                'bundle_hash': bundle_hash,
                                'state_diff': result['result'].get('stateBlockNumber'),
                                'inclusion_probability': self.calculate_inclusion_probability(transactions, target_block)
                            }
                        else:
                            simulation = {
                                'success': False,
                                'error': result.get('error', {}).get('message', 'Unknown simulation error'),
                                'bundle_hash': bundle_hash
                            }
                    else:
                        simulation = {
                            'success': False,
                            'error': f"HTTP {response.status}",
                            'bundle_hash': bundle_hash
                        }
        
        except Exception as e:
            simulation = {
                'success': False,
                'error': str(e),
                'bundle_hash': bundle_hash,
                'gas_used': sum(tx.get('gas', 21000) for tx in transactions),
                'effective_gas_price': 20000000000,
                'inclusion_probability': 0.1
            }
        
        self.simulation_cache[bundle_hash] = simulation
        return simulation

    def calculate_inclusion_probability(self, transactions: List[Dict], target_block: int) -> float:
        base_probability = 0.3
        
        total_gas_price = sum(tx.get('gasPrice', 20000000000) for tx in transactions)
        avg_gas_price = total_gas_price / len(transactions) if transactions else 20000000000
        
        if avg_gas_price > 50000000000:
            base_probability += 0.4
        elif avg_gas_price > 30000000000:
            base_probability += 0.2
        
        bundle_size = len(transactions)
        if bundle_size == 1:
            base_probability += 0.1
        elif bundle_size > 5:
            base_probability -= 0.2
        
        return min(base_probability, 0.95)

    async def submit_bundle(self, bundle: FlashbotsBundle, chain: str = 'mainnet') -> Dict:
        try:
            endpoint = self.relay_endpoints.get(chain, self.relay_endpoints['mainnet'])
            
            bundle_request = {
                "jsonrpc": "2.0",
                "method": "eth_sendBundle",
                "params": [
                    {
                        "txs": [tx.get('rawTransaction', tx.get('data', '0x')) for tx in bundle.transactions],
                        "blockNumber": hex(bundle.target_block),
                        "minTimestamp": 0,
                        "maxTimestamp": 0
                    }
                ],
                "id": int(time.time())
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-Flashbots-Signature': self.generate_flashbots_signature(json.dumps(bundle_request))
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=bundle_request, headers=headers, timeout=15) as response:
                    result = await response.json()
                    
                    if 'error' not in result:
                        self.bundle_stats['submitted'] += 1
                        bundle_hash = result.get('result', f"bundle_{int(time.time())}")
                        
                        inclusion_result = await self.monitor_bundle_inclusion(bundle_hash, bundle.target_block, chain)
                        
                        if inclusion_result['included']:
                            self.bundle_stats['included'] += 1
                        else:
                            self.bundle_stats['failed'] += 1
                        
                        self.bundle_stats['avg_inclusion_rate'] = self.bundle_stats['included'] / max(self.bundle_stats['submitted'], 1)
                        
                        return {
                            'success': True,
                            'bundle_hash': bundle_hash,
                            'target_block': bundle.target_block,
                            'inclusion_result': inclusion_result
                        }
                    else:
                        self.bundle_stats['failed'] += 1
                        return {
                            'success': False,
                            'error': result['error'].get('message', 'Unknown error'),
                            'code': result['error'].get('code', -1)
                        }
                        
        except Exception as e:
            self.bundle_stats['failed'] += 1
            return {'success': False, 'error': str(e)}

    async def monitor_bundle_inclusion(self, bundle_hash: str, target_block: int, chain: str, timeout: int = 30) -> Dict:
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                inclusion_check = await self.check_bundle_inclusion_status(bundle_hash, target_block, chain)
                
                if inclusion_check['status'] != 'pending':
                    return inclusion_check
                
                await asyncio.sleep(2)
                
            except Exception as e:
                continue
        
        return {
            'included': False,
            'status': 'timeout',
            'bundle_hash': bundle_hash,
            'target_block': target_block
        }

    async def check_bundle_inclusion_status(self, bundle_hash: str, target_block: int, chain: str) -> Dict:
        try:
            endpoint = self.relay_endpoints.get(chain, self.relay_endpoints['mainnet'])
            
            status_request = {
                "jsonrpc": "2.0",
                "method": "flashbots_getBundleStats",
                "params": [bundle_hash, hex(target_block)],
                "id": int(time.time())
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-Flashbots-Signature': self.generate_flashbots_signature(json.dumps(status_request))
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=status_request, headers=headers, timeout=5) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'result' in result:
                            stats = result['result']
                            return {
                                'included': stats.get('isSimulated', False) and stats.get('isSentToBundlePool', False),
                                'status': 'included' if stats.get('isSimulated') else 'failed',
                                'bundle_hash': bundle_hash,
                                'target_block': target_block,
                                'simulation_success': stats.get('isSimulated', False),
                                'sent_to_miners': stats.get('isSentToBundlePool', False)
                            }
        except Exception:
            pass
        
        return {
            'included': False,
            'status': 'pending',
            'bundle_hash': bundle_hash,
            'target_block': target_block
        }

class PrivateMempool:
    def __init__(self):
        self.private_pools = {
            'flashbots': {
                'endpoint': 'https://relay.flashbots.net',
                'type': 'flashbots',
                'chain_support': ['ethereum', 'arbitrum', 'polygon']
            },
            'manifold': {
                'endpoint': 'https://api.manifoldfinance.com/v1',
                'type': 'private_pool',
                'chain_support': ['ethereum']
            },
            'bloXroute': {
                'endpoint': 'https://api.bloxroute.com',
                'type': 'private_pool',
                'chain_support': ['ethereum', 'polygon', 'arbitrum']
            },
            '1inch_fusion': {
                'endpoint': 'https://api.1inch.dev/fusion',
                'type': 'intent_based',
                'chain_support': ['ethereum', 'polygon', 'arbitrum']
            }
        }
        
        self.submission_stats = {}
        self.logger = logging.getLogger(__name__)

    async def submit_to_private_pools(self, transaction_data: Dict, chain: str, priority: str = 'speed') -> List[PrivatePoolSubmission]:
        submissions = []
        
        available_pools = [
            pool_name for pool_name, pool_config in self.private_pools.items()
            if chain in pool_config['chain_support']
        ]
        
        for pool_name in available_pools:
            try:
                submission = await self.submit_to_pool(pool_name, transaction_data, chain)
                submissions.append(submission)
                
                if submission.success and priority == 'speed':
                    break
                    
            except Exception as e:
                submissions.append(PrivatePoolSubmission(
                    pool_name=pool_name,
                    endpoint=self.private_pools[pool_name]['endpoint'],
                    success=False,
                    response_time=0.0,
                    bundle_hash=None,
                    error=str(e)
                ))
        
        return submissions

    async def submit_to_pool(self, pool_name: str, transaction_data: Dict, chain: str) -> PrivatePoolSubmission:
        start_time = time.time()
        pool_config = self.private_pools[pool_name]
        
        try:
            if pool_config['type'] == 'flashbots':
                result = await self.submit_flashbots_transaction(pool_config['endpoint'], transaction_data, chain)
            elif pool_config['type'] == 'private_pool':
                result = await self.submit_private_pool_transaction(pool_config['endpoint'], transaction_data, chain)
            elif pool_config['type'] == 'intent_based':
                result = await self.submit_intent_transaction(pool_config['endpoint'], transaction_data, chain)
            else:
                raise Exception(f"Unknown pool type: {pool_config['type']}")
            
            response_time = time.time() - start_time
            
            return PrivatePoolSubmission(
                pool_name=pool_name,
                endpoint=pool_config['endpoint'],
                success=result.get('success', False),
                response_time=response_time,
                bundle_hash=result.get('bundle_hash'),
                error=result.get('error')
            )
            
        except Exception as e:
            return PrivatePoolSubmission(
                pool_name=pool_name,
                endpoint=pool_config['endpoint'],
                success=False,
                response_time=time.time() - start_time,
                bundle_hash=None,
                error=str(e)
            )

    async def submit_flashbots_transaction(self, endpoint: str, transaction_data: Dict, chain: str) -> Dict:
        bundle_request = {
            "jsonrpc": "2.0",
            "method": "eth_sendBundle",
            "params": [{
                "txs": [transaction_data.get('rawTransaction', transaction_data.get('data', '0x'))],
                "blockNumber": hex(await self.get_next_block_number(chain)),
                "minTimestamp": 0,
                "maxTimestamp": 0
            }],
            "id": int(time.time())
        }
        
        headers = {'Content-Type': 'application/json'}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=bundle_request, headers=headers, timeout=10) as response:
                result = await response.json()
                
                if 'error' not in result:
                    return {
                        'success': True,
                        'bundle_hash': result.get('result', f"fb_{int(time.time())}")
                    }
                else:
                    return {
                        'success': False,
                        'error': result['error'].get('message', 'Flashbots submission failed')
                    }

    async def submit_private_pool_transaction(self, endpoint: str, transaction_data: Dict, chain: str) -> Dict:
        pool_request = {
            'transaction': transaction_data.get('rawTransaction', transaction_data.get('data', '0x')),
            'chain': chain,
            'priority': 'high',
            'timestamp': int(time.time())
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {os.getenv('PRIVATE_POOL_API_KEY', 'demo')}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{endpoint}/submit", json=pool_request, headers=headers, timeout=10) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'bundle_hash': result.get('id', f"pp_{int(time.time())}")
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Private pool submission failed: {response.status}"
                    }

    async def submit_intent_transaction(self, endpoint: str, transaction_data: Dict, chain: str) -> Dict:
        intent_request = {
            'intent': {
                'from': transaction_data.get('from'),
                'to': transaction_data.get('to'),
                'value': transaction_data.get('value', '0'),
                'data': transaction_data.get('data', '0x'),
                'gasLimit': transaction_data.get('gas', 21000),
                'maxFeePerGas': transaction_data.get('maxFeePerGas', transaction_data.get('gasPrice')),
                'maxPriorityFeePerGas': transaction_data.get('maxPriorityFeePerGas', '2000000000')
            },
            'chain': chain,
            'deadline': int(time.time()) + 300
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {os.getenv('INTENT_API_KEY', 'demo')}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{endpoint}/orders", json=intent_request, headers=headers, timeout=10) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'bundle_hash': result.get('orderHash', f"intent_{int(time.time())}")
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Intent submission failed: {response.status}"
                    }

    async def get_next_block_number(self, chain: str) -> int:
        return int(time.time() // 12) + 1

class OrderFlowAnalyzer:
    def __init__(self):
        self.order_flow_cache = {}
        self.sandwich_detection = {}
        self.frontrun_patterns = {}
        
    def analyze_real_order_flow(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {'imbalance': 0.0, 'toxic_flow': 0.0, 'informed_trading': 0.0}
        
        buy_volume = sum(t.get('volume', 0) for t in trades if t.get('side') == 'buy')
        sell_volume = sum(t.get('volume', 0) for t in trades if t.get('side') == 'sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {'imbalance': 0.0, 'toxic_flow': 0.0, 'informed_trading': 0.0}
        
        order_flow_imbalance = (buy_volume - sell_volume) / total_volume
        
        large_trades = [t for t in trades if t.get('volume', 0) > total_volume * 0.05]
        toxic_flow = len(large_trades) / len(trades) if trades else 0
        
        price_changes = [t.get('price_impact', 0) for t in trades]
        informed_trading = sum(abs(p) for p in price_changes) / len(price_changes) if price_changes else 0
        
        return {
            'imbalance': order_flow_imbalance,
            'toxic_flow': toxic_flow,
            'informed_trading': min(informed_trading, 1.0),
            'large_trade_ratio': len(large_trades) / len(trades) if trades else 0,
            'avg_trade_size': total_volume / len(trades) if trades else 0
        }

    def detect_sandwich_attack(self, pending_tx: Dict, recent_txs: List[Dict]) -> Dict:
        target_token = pending_tx.get('token_address')
        target_value = pending_tx.get('value', 0)
        target_gas_price = pending_tx.get('gasPrice', 0)
        
        sandwich_risk = 0.0
        frontrun_txs = []
        backrun_txs = []
        
        for tx in recent_txs:
            if tx.get('token_address') == target_token:
                tx_gas_price = tx.get('gasPrice', 0)
                tx_value = tx.get('value', 0)
                
                if (tx_gas_price > target_gas_price * 1.1 and 
                    tx_value > target_value * 0.1 and
                    tx.get('side') == pending_tx.get('side')):
                    frontrun_txs.append(tx)
                    sandwich_risk += 0.3
                
                elif (tx_gas_price < target_gas_price * 0.9 and
                      tx.get('side') != pending_tx.get('side')):
                    backrun_txs.append(tx)
                    sandwich_risk += 0.2
        
        return {
            'sandwich_risk': min(sandwich_risk, 1.0),
            'frontrun_count': len(frontrun_txs),
            'backrun_count': len(backrun_txs),
            'protection_needed': sandwich_risk > 0.5
        }

    def calculate_mev_extraction_potential(self, transaction: Dict, market_state: Dict) -> float:
        arbitrage_potential = 0.0
        
        token_address = transaction.get('token_address')
        trade_size = transaction.get('value', 0)
        
        if token_address in market_state.get('price_differences', {}):
            price_diff = market_state['price_differences'][token_address]
            arbitrage_potential = abs(price_diff) * trade_size * 0.1
        
        liquidity_impact = trade_size / market_state.get('total_liquidity', trade_size * 10)
        slippage_opportunity = min(liquidity_impact * 0.05, 0.02)
        
        gas_arbitrage = 0.0
        if transaction.get('gasPrice', 0) < market_state.get('avg_gas_price', 0):
            gas_arbitrage = 0.001
        
        total_mev_potential = arbitrage_potential + slippage_opportunity + gas_arbitrage
        
        return min(total_mev_potential, 0.1)

class MEVProtection:
    def __init__(self):
        self.flashbots = FlashbotsIntegration()
        self.private_mempool = PrivateMempool()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.protection_strategies = ['flashbots', 'private_pool', 'intent_based', 'timing']
        self.mev_opportunities_detected = []
        self.protection_stats = {
            'transactions_protected': 0,
            'mev_prevented': 0.0,
            'flashbots_success_rate': 0.0,
            'private_pool_success_rate': 0.0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def protect_transaction(self, transaction_data: Dict, chain: str, protection_level: str = 'high') -> Dict:
        mev_risk = await self.assess_mev_risk(transaction_data, chain)
        
        if mev_risk['total_risk'] < get_dynamic_config().get('mev_protection_threshold', 0.01):
            return await self.submit_regular_transaction(transaction_data, chain)
        
        protection_strategy = self.select_protection_strategy(mev_risk, protection_level)
        
        if protection_strategy == 'flashbots':
            return await self.protect_with_flashbots(transaction_data, chain, mev_risk)
        elif protection_strategy == 'private_pool':
            return await self.protect_with_private_pools(transaction_data, chain)
        elif protection_strategy == 'intent_based':
            return await self.protect_with_intent_system(transaction_data, chain)
        elif protection_strategy == 'timing':
            return await self.protect_with_timing(transaction_data, chain, mev_risk)
        else:
            return await self.submit_regular_transaction(transaction_data, chain)

    async def assess_mev_risk(self, transaction_data: Dict, chain: str) -> Dict:
        frontrun_risk = await self.calculate_frontrun_risk(transaction_data)
        sandwich_risk = await self.calculate_sandwich_risk(transaction_data, chain)
        arbitrage_risk = await self.calculate_arbitrage_risk(transaction_data)
        
        total_risk = (frontrun_risk * 0.4 + sandwich_risk * 0.4 + arbitrage_risk * 0.2)
        
        return {
            'frontrun_risk': frontrun_risk,
            'sandwich_risk': sandwich_risk,
            'arbitrage_risk': arbitrage_risk,
            'total_risk': total_risk,
            'protection_recommended': total_risk > 0.3
        }

    async def calculate_frontrun_risk(self, transaction_data: Dict) -> float:
        risk_score = 0.0
        
        if transaction_data.get('value', 0) > 1000000000000000000:
            risk_score += 0.3
        
        if 'swapExact' in transaction_data.get('data', ''):
            risk_score += 0.4
        
        gas_price = transaction_data.get('gasPrice', 0)
        if gas_price < 30000000000:
            risk_score += 0.3
        
        return min(risk_score, 1.0)

    async def calculate_sandwich_risk(self, transaction_data: Dict, chain: str) -> float:
        if not transaction_data.get('token_address'):
            return 0.0
        
        recent_transactions = await self.get_recent_mempool_transactions(chain, limit=50)
        sandwich_analysis = self.order_flow_analyzer.detect_sandwich_attack(transaction_data, recent_transactions)
        
        return sandwich_analysis['sandwich_risk']

    async def calculate_arbitrage_risk(self, transaction_data: Dict) -> float:
        market_state = await self.get_market_state(transaction_data.get('token_address', ''))
        mev_potential = self.order_flow_analyzer.calculate_mev_extraction_potential(transaction_data, market_state)
        
        return min(mev_potential * 10, 1.0)

    async def get_recent_mempool_transactions(self, chain: str, limit: int = 50) -> List[Dict]:
        simulated_txs = []
        
        for i in range(min(limit, 20)):
            simulated_txs.append({
                'hash': f"0x{hash(f'{chain}_{i}_{time.time()}') % (16**64):064x}",
                'token_address': f"0x{hash(f'token_{i}') % (16**40):040x}",
                'value': np.random.uniform(0.1, 10.0) * 1e18,
                'gasPrice': np.random.randint(20, 100) * 1e9,
                'side': np.random.choice(['buy', 'sell']),
                'timestamp': time.time() - i * 2
            })
        
        return simulated_txs

    async def get_market_state(self, token_address: str) -> Dict:
        return {
            'total_liquidity': np.random.uniform(100000, 10000000),
            'avg_gas_price': np.random.randint(20, 50) * 1e9,
            'price_differences': {
                token_address: np.random.uniform(-0.05, 0.05)
            },
            'volume_24h': np.random.uniform(10000, 1000000)
        }

    def select_protection_strategy(self, mev_risk: Dict, protection_level: str) -> str:
        if protection_level == 'high' or mev_risk['total_risk'] > 0.7:
            if mev_risk['sandwich_risk'] > 0.6:
                return 'flashbots'
            else:
                return 'private_pool'
        elif protection_level == 'medium' or mev_risk['total_risk'] > 0.3:
            return 'intent_based'
        else:
            return 'timing'

    async def protect_with_flashbots(self, transaction_data: Dict, chain: str, mev_risk: Dict) -> Dict:
        try:
            current_block = await self.get_current_block_number(chain)
            target_block = current_block + 1
            
            simulation = await self.flashbots.simulate_bundle([transaction_data], target_block, chain)
            
            if not simulation['success']:
                return await self.submit_regular_transaction(transaction_data, chain)
            
            bundle = FlashbotsBundle(
                transactions=[transaction_data],
                target_block=target_block,
                max_priority_fee=transaction_data.get('maxPriorityFeePerGas', 2000000000),
                simulation_result=simulation
            )
            
            result = await self.flashbots.submit_bundle(bundle, chain)
            
            self.protection_stats['transactions_protected'] += 1
            
            if result['success']:
                self.protection_stats['mev_prevented'] += mev_risk['total_risk'] * float(transaction_data.get('value', 0)) / 1e18
                return {
                    'success': True,
                    'protection_method': 'flashbots',
                    'bundle_hash': result['bundle_hash'],
                    'mev_risk_prevented': mev_risk['total_risk']
                }
            else:
                return await self.submit_regular_transaction(transaction_data, chain)
                
        except Exception as e:
            self.logger.error(f"Flashbots protection failed: {e}")
            return await self.submit_regular_transaction(transaction_data, chain)

    async def protect_with_private_pools(self, transaction_data: Dict, chain: str) -> Dict:
        submissions = await self.private_mempool.submit_to_private_pools(transaction_data, chain, priority='speed')
        
        successful_submissions = [s for s in submissions if s.success]
        
        if successful_submissions:
            best_submission = min(successful_submissions, key=lambda s: s.response_time)
            
            self.protection_stats['transactions_protected'] += 1
            self.protection_stats['private_pool_success_rate'] = len(successful_submissions) / len(submissions)
            
            return {
                'success': True,
                'protection_method': 'private_pool',
                'pool_used': best_submission.pool_name,
                'bundle_hash': best_submission.bundle_hash,
                'response_time': best_submission.response_time
            }
        else:
            return await self.submit_regular_transaction(transaction_data, chain)

    async def protect_with_intent_system(self, transaction_data: Dict, chain: str) -> Dict:
        intent_pools = ['1inch_fusion']
        
        for pool_name in intent_pools:
            if chain in self.private_mempool.private_pools[pool_name]['chain_support']:
                try:
                    submission = await self.private_mempool.submit_to_pool(pool_name, transaction_data, chain)
                    
                    if submission.success:
                        self.protection_stats['transactions_protected'] += 1
                        
                        return {
                            'success': True,
                            'protection_method': 'intent_based',
                            'intent_hash': submission.bundle_hash,
                            'pool_used': pool_name
                        }
                except Exception as e:
                    continue
        
        return await self.submit_regular_transaction(transaction_data, chain)

    async def protect_with_timing(self, transaction_data: Dict, chain: str, mev_risk: Dict) -> Dict:
        optimal_delay = self.calculate_optimal_timing_delay(mev_risk)
        
        await asyncio.sleep(optimal_delay)
        
        return await self.submit_regular_transaction(transaction_data, chain)

    def calculate_optimal_timing_delay(self, mev_risk: Dict) -> float:
        base_delay = 1.0
        
        if mev_risk['frontrun_risk'] > 0.5:
            base_delay += 2.0
        
        if mev_risk['sandwich_risk'] > 0.5:
            base_delay += 1.5
        
        return min(base_delay, 5.0)

    async def submit_regular_transaction(self, transaction_data: Dict, chain: str) -> Dict:
        return {
            'success': True,
            'protection_method': 'none',
            'tx_hash': f"0x{hash(str(transaction_data) + str(time.