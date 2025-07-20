import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque
import networkx as nx

@dataclass
class WhaleWallet:
    address: str
    total_value_usd: float
    eth_balance: float
    token_count: int
    top_holdings: List[Dict]
    transaction_count: int
    first_transaction: int
    last_transaction: int
    cluster_id: Optional[int]
    risk_score: float
    whale_type: str
    connected_wallets: List[str]

@dataclass
class WhaleTransaction:
    wallet_address: str
    token_address: str
    transaction_type: str
    amount_usd: float
    token_amount: float
    price_impact: float
    timestamp: int
    block_number: int
    transaction_hash: str
    gas_used: int

@dataclass
class ClusterAnalysis:
    cluster_id: int
    wallet_count: int
    total_value_usd: float
    average_transaction_size: float
    common_tokens: List[str]
    activity_pattern: str
    coordination_score: float
    risk_level: str

class WhaleWalletAnalyzer:
    def __init__(self):
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        self.apis = {
            'alchemy': f"https://eth-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}",
            'moralis': 'https://deep-index.moralis.io/api/v2',
            'etherscan': f"https://api.etherscan.io/api?apikey={os.getenv('ETHERSCAN_API_KEY')}",
            'debank': 'https://openapi.debank.com/v1',
            'nansen': 'https://api.nansen.ai/v1'
        }
        
        self.whale_thresholds = {
            'eth_balance': 100.0,
            'total_value': 1000000.0,
            'transaction_size': 50000.0
        }
        
        self.known_whale_addresses = set()
        self.whale_clusters = {}
        self.transaction_history = defaultdict(deque)
        self.wallet_network = nx.Graph()
        
    async def initialize(self):
        connector = aiohttp.TCPConnector(limit=100)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        await self.load_known_whales()
        
    async def load_known_whales(self):
        known_whales_file = 'data/known_whale_addresses.txt'
        
        if os.path.exists(known_whales_file):
            with open(known_whales_file, 'r') as f:
                for line in f:
                    address = line.strip()
                    if address and address.startswith('0x'):
                        self.known_whale_addresses.add(address.lower())
        
        self.logger.info(f"Loaded {len(self.known_whale_addresses)} known whale addresses")
    
    async def discover_whale_wallets(self, token_address: str) -> List[WhaleWallet]:
        try:
            large_holders = await self.get_large_token_holders(token_address)
            whale_wallets = []
            
            for holder in large_holders:
                wallet = await self.analyze_wallet(holder['address'])
                if wallet and self.is_whale_wallet(wallet):
                    whale_wallets.append(wallet)
                    self.known_whale_addresses.add(wallet.address.lower())
            
            await self.cluster_whale_wallets(whale_wallets)
            
            return whale_wallets
            
        except Exception as e:
            self.logger.error(f"Error discovering whale wallets: {e}")
            return []
    
    async def get_large_token_holders(self, token_address: str, limit: int = 1000) -> List[Dict]:
        try:
            url = f"{self.apis['alchemy']}"
            payload = {
                "jsonrpc": "2.0",
                "method": "alchemy_getTokenBalances",
                "params": [token_address],
                "id": 1
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    holders = data.get('result', {}).get('tokenBalances', [])
                    
                    enriched_holders = []
                    for holder in holders[:limit]:
                        holder_data = await self.get_wallet_overview(holder['address'])
                        if holder_data:
                            enriched_holders.append(holder_data)
                    
                    enriched_holders.sort(key=lambda x: x.get('total_value_usd', 0), reverse=True)
                    return enriched_holders[:100]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting token holders: {e}")
            return []
    
    async def get_wallet_overview(self, wallet_address: str) -> Optional[Dict]:
        try:
            headers = {'X-API-Key': os.getenv('MORALIS_API_KEY')}
            url = f"{self.apis['moralis']}/wallets/{wallet_address}/tokens"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    total_value = 0
                    token_count = 0
                    top_holdings = []
                    
                    for token in data.get('result', []):
                        token_value = float(token.get('usd_value', 0))
                        total_value += token_value
                        token_count += 1
                        
                        if len(top_holdings) < 10:
                            top_holdings.append({
                                'token_address': token.get('token_address'),
                                'symbol': token.get('symbol'),
                                'balance': token.get('balance'),
                                'usd_value': token_value
                            })
                    
                    eth_balance = await self.get_eth_balance(wallet_address)
                    
                    return {
                        'address': wallet_address,
                        'total_value_usd': total_value,
                        'eth_balance': eth_balance,
                        'token_count': token_count,
                        'top_holdings': top_holdings
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting wallet overview: {e}")
            return None
    
    async def get_eth_balance(self, wallet_address: str) -> float:
        try:
            url = f"{self.apis['alchemy']}"
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_getBalance",
                "params": [wallet_address, "latest"],
                "id": 1
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    balance_wei = int(data.get('result', '0x0'), 16)
                    balance_eth = balance_wei / 10**18
                    return balance_eth
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    async def analyze_wallet(self, wallet_address: str) -> Optional[WhaleWallet]:
        try:
            overview = await self.get_wallet_overview(wallet_address)
            if not overview:
                return None
            
            transactions = await self.get_wallet_transactions(wallet_address)
            transaction_analysis = self.analyze_transaction_patterns(transactions)
            
            risk_score = self.calculate_risk_score(overview, transaction_analysis)
            whale_type = self.classify_whale_type(overview, transaction_analysis)
            connected_wallets = await self.find_connected_wallets(wallet_address)
            
            return WhaleWallet(
                address=wallet_address,
                total_value_usd=overview['total_value_usd'],
                eth_balance=overview['eth_balance'],
                token_count=overview['token_count'],
                top_holdings=overview['top_holdings'],
                transaction_count=len(transactions),
                first_transaction=min([tx['timestamp'] for tx in transactions]) if transactions else 0,
                last_transaction=max([tx['timestamp'] for tx in transactions]) if transactions else 0,
                cluster_id=None,
                risk_score=risk_score,
                whale_type=whale_type,
                connected_wallets=connected_wallets
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing wallet {wallet_address}: {e}")
            return None
    
    async def get_wallet_transactions(self, wallet_address: str, limit: int = 1000) -> List[Dict]:
        try:
            url = f"{self.apis['etherscan']}&module=account&action=txlist&address={wallet_address}&sort=desc&page=1&offset={limit}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    transactions = data.get('result', [])
                    
                    enriched_transactions = []
                    for tx in transactions:
                        enriched_tx = await self.enrich_transaction_data(tx)
                        if enriched_tx:
                            enriched_transactions.append(enriched_tx)
                    
                    return enriched_transactions
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting wallet transactions: {e}")
            return []
    
    async def enrich_transaction_data(self, tx: Dict) -> Optional[Dict]:
        try:
            value_eth = float(tx.get('value', '0')) / 10**18
            gas_price = float(tx.get('gasPrice', '0'))
            gas_used = int(tx.get('gasUsed', '0'))
            
            eth_price = await self.get_eth_price_at_timestamp(int(tx.get('timeStamp', '0')))
            value_usd = value_eth * eth_price
            
            return {
                'hash': tx.get('hash'),
                'block_number': int(tx.get('blockNumber', '0')),
                'timestamp': int(tx.get('timeStamp', '0')),
                'from_address': tx.get('from'),
                'to_address': tx.get('to'),
                'value_eth': value_eth,
                'value_usd': value_usd,
                'gas_price': gas_price,
                'gas_used': gas_used,
                'is_error': tx.get('isError') == '1',
                'method_id': tx.get('input', '')[:10] if tx.get('input') else None
            }
            
        except Exception as e:
            return None
    
    async def get_eth_price_at_timestamp(self, timestamp: int) -> float:
        current_time = int(time.time())
        if abs(current_time - timestamp) < 3600:
            try:
                url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['ethereum']['usd']
            except:
                pass
        
        return 2000.0
    
    def analyze_transaction_patterns(self, transactions: List[Dict]) -> Dict:
        if not transactions:
            return {
                'avg_transaction_size': 0,
                'transaction_frequency': 0,
                'activity_periods': [],
                'counterpart_diversity': 0,
                'gas_optimization': 0
            }
        
        transaction_sizes = [tx['value_usd'] for tx in transactions if tx['value_usd'] > 0]
        timestamps = [tx['timestamp'] for tx in transactions]
        counterparts = set()
        
        for tx in transactions:
            counterparts.add(tx['from_address'])
            counterparts.add(tx['to_address'])
        
        time_diffs = np.diff(sorted(timestamps))
        avg_frequency = np.mean(time_diffs) if len(time_diffs) > 0 else 0
        
        activity_periods = self.identify_activity_periods(timestamps)
        
        gas_prices = [tx['gas_price'] for tx in transactions if tx['gas_price'] > 0]
        gas_optimization_score = self.calculate_gas_optimization_score(gas_prices)
        
        return {
            'avg_transaction_size': np.mean(transaction_sizes) if transaction_sizes else 0,
            'transaction_frequency': avg_frequency,
            'activity_periods': activity_periods,
            'counterpart_diversity': len(counterparts),
            'gas_optimization': gas_optimization_score
        }
    
    def identify_activity_periods(self, timestamps: List[int]) -> List[Dict]:
        if len(timestamps) < 10:
            return []
        
        timestamps = sorted(timestamps)
        periods = []
        current_period_start = timestamps[0]
        current_period_txs = 1
        
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            
            if time_diff > 86400:
                if current_period_txs >= 5:
                    periods.append({
                        'start': current_period_start,
                        'end': timestamps[i-1],
                        'transaction_count': current_period_txs,
                        'duration': timestamps[i-1] - current_period_start
                    })
                
                current_period_start = timestamps[i]
                current_period_txs = 1
            else:
                current_period_txs += 1
        
        if current_period_txs >= 5:
            periods.append({
                'start': current_period_start,
                'end': timestamps[-1],
                'transaction_count': current_period_txs,
                'duration': timestamps[-1] - current_period_start
            })
        
        return periods
    
    def calculate_gas_optimization_score(self, gas_prices: List[float]) -> float:
        if not gas_prices:
            return 0.0
        
        gas_prices = np.array(gas_prices)
        gas_variance = np.var(gas_prices)
        gas_mean = np.mean(gas_prices)
        
        coefficient_of_variation = gas_variance / (gas_mean + 1e-8)
        optimization_score = 1.0 / (1.0 + coefficient_of_variation)
        
        return optimization_score
    
    def calculate_risk_score(self, overview: Dict, transaction_analysis: Dict) -> float:
        risk_factors = []
        
        total_value = overview.get('total_value_usd', 0)
        if total_value > 10000000:
            risk_factors.append(0.8)
        elif total_value > 1000000:
            risk_factors.append(0.6)
        elif total_value > 100000:
            risk_factors.append(0.4)
        else:
            risk_factors.append(0.2)
        
        avg_tx_size = transaction_analysis.get('avg_transaction_size', 0)
        if avg_tx_size > 1000000:
            risk_factors.append(0.9)
        elif avg_tx_size > 100000:
            risk_factors.append(0.7)
        elif avg_tx_size > 10000:
            risk_factors.append(0.5)
        else:
            risk_factors.append(0.3)
        
        activity_periods = transaction_analysis.get('activity_periods', [])
        if len(activity_periods) > 5:
            risk_factors.append(0.6)
        elif len(activity_periods) > 2:
            risk_factors.append(0.4)
        else:
            risk_factors.append(0.2)
        
        counterpart_diversity = transaction_analysis.get('counterpart_diversity', 0)
        if counterpart_diversity < 10:
            risk_factors.append(0.8)
        elif counterpart_diversity < 50:
            risk_factors.append(0.5)
        else:
            risk_factors.append(0.2)
        
        return np.mean(risk_factors)
    
    def classify_whale_type(self, overview: Dict, transaction_analysis: Dict) -> str:
        total_value = overview.get('total_value_usd', 0)
        avg_tx_size = transaction_analysis.get('avg_transaction_size', 0)
        token_count = overview.get('token_count', 0)
        
        if total_value > 50000000 and avg_tx_size > 1000000:
            return 'institutional'
        elif token_count > 100 and avg_tx_size > 100000:
            return 'fund_manager'
        elif avg_tx_size > 500000:
            return 'large_trader'
        elif token_count < 10 and total_value > 1000000:
            return 'hodler'
        elif len(transaction_analysis.get('activity_periods', [])) > 5:
            return 'active_trader'
        else:
            return 'unknown'
    
    async def find_connected_wallets(self, wallet_address: str) -> List[str]:
        try:
            transactions = await self.get_wallet_transactions(wallet_address, limit=500)
            connected_addresses = set()
            
            for tx in transactions:
                from_addr = tx.get('from_address', '').lower()
                to_addr = tx.get('to_address', '').lower()
                
                if from_addr and from_addr != wallet_address.lower():
                    connected_addresses.add(from_addr)
                if to_addr and to_addr != wallet_address.lower():
                    connected_addresses.add(to_addr)
            
            significant_connections = []
            for addr in connected_addresses:
                connection_strength = await self.calculate_connection_strength(wallet_address, addr)
                if connection_strength > 0.3:
                    significant_connections.append(addr)
            
            return significant_connections[:20]
            
        except Exception as e:
            self.logger.error(f"Error finding connected wallets: {e}")
            return []
    
    async def calculate_connection_strength(self, wallet1: str, wallet2: str) -> float:
        try:
            transactions1 = await self.get_wallet_transactions(wallet1, limit=200)
            
            shared_transactions = 0
            total_value = 0
            
            for tx in transactions1:
                if (tx.get('from_address', '').lower() == wallet2.lower() or 
                    tx.get('to_address', '').lower() == wallet2.lower()):
                    shared_transactions += 1
                    total_value += tx.get('value_usd', 0)
            
            if shared_transactions == 0:
                return 0.0
            
            frequency_score = min(shared_transactions / 10, 1.0)
            value_score = min(total_value / 100000, 1.0)
            
            return (frequency_score + value_score) / 2
            
        except Exception as e:
            return 0.0
    
    async def cluster_whale_wallets(self, whale_wallets: List[WhaleWallet]):
        if len(whale_wallets) < 3:
            return
        
        features = []
        addresses = []
        
        for wallet in whale_wallets:
            features.append([
                wallet.total_value_usd,
                wallet.eth_balance,
                wallet.token_count,
                wallet.transaction_count,
                wallet.risk_score,
                len(wallet.connected_wallets)
            ])
            addresses.append(wallet.address)
        
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        clustering = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = clustering.fit_predict(features_scaled)
        
        for i, wallet in enumerate(whale_wallets):
            wallet.cluster_id = int(cluster_labels[i]) if cluster_labels[i] != -1 else None
        
        await self.analyze_clusters(whale_wallets)
    
    async def analyze_clusters(self, whale_wallets: List[WhaleWallet]):
        clusters = defaultdict(list)
        
        for wallet in whale_wallets:
            if wallet.cluster_id is not None:
                clusters[wallet.cluster_id].append(wallet)
        
        for cluster_id, wallets in clusters.items():
            if len(wallets) < 2:
                continue
            
            total_value = sum(w.total_value_usd for w in wallets)
            avg_tx_count = np.mean([w.transaction_count for w in wallets])
            
            common_holdings = self.find_common_holdings(wallets)
            coordination_score = await self.calculate_coordination_score(wallets)
            
            risk_level = 'high' if coordination_score > 0.7 else 'medium' if coordination_score > 0.4 else 'low'
            
            cluster_analysis = ClusterAnalysis(
                cluster_id=cluster_id,
                wallet_count=len(wallets),
                total_value_usd=total_value,
                average_transaction_size=avg_tx_count,
                common_tokens=[holding['symbol'] for holding in common_holdings],
                activity_pattern='coordinated' if coordination_score > 0.6 else 'independent',
                coordination_score=coordination_score,
                risk_level=risk_level
            )
            
            self.whale_clusters[cluster_id] = cluster_analysis
    
    def find_common_holdings(self, wallets: List[WhaleWallet]) -> List[Dict]:
        token_counts = defaultdict(int)
        token_info = {}
        
        for wallet in wallets:
            for holding in wallet.top_holdings:
                token_address = holding['token_address']
                token_counts[token_address] += 1
                token_info[token_address] = holding
        
        common_threshold = len(wallets) * 0.6
        common_holdings = []
        
        for token_address, count in token_counts.items():
            if count >= common_threshold:
                common_holdings.append(token_info[token_address])
        
        return common_holdings
    
    async def calculate_coordination_score(self, wallets: List[WhaleWallet]) -> float:
        if len(wallets) < 2:
            return 0.0
        
        coordination_factors = []
        
        transaction_times = []
        for wallet in wallets:
            if wallet.last_transaction > 0:
                transaction_times.append(wallet.last_transaction)
        
        if len(transaction_times) > 1:
            time_variance = np.var(transaction_times)
            time_coordination = 1.0 / (1.0 + time_variance / 86400)
            coordination_factors.append(time_coordination)
        
        common_holdings = self.find_common_holdings(wallets)
        holding_coordination = len(common_holdings) / 10
        coordination_factors.append(min(holding_coordination, 1.0))
        
        if len(coordination_factors) == 0:
            return 0.0
        
        return np.mean(coordination_factors)
    
    def is_whale_wallet(self, wallet: WhaleWallet) -> bool:
        return (wallet.total_value_usd >= self.whale_thresholds['total_value'] or
                wallet.eth_balance >= self.whale_thresholds['eth_balance'])
    
    async def monitor_whale_activity(self, token_address: str) -> List[WhaleTransaction]:
        whale_transactions = []
        
        for whale_address in list(self.known_whale_addresses)[:50]:
            try:
                recent_txs = await self.get_recent_token_transactions(whale_address, token_address)
                
                for tx in recent_txs:
                    whale_tx = await self.create_whale_transaction(tx, whale_address)
                    if whale_tx:
                        whale_transactions.append(whale_tx)
                
            except Exception as e:
                continue
        
        return whale_transactions
    
    async def get_recent_token_transactions(self, wallet_address: str, token_address: str) -> List[Dict]:
        try:
            url = f"{self.apis['etherscan']}&module=account&action=tokentx&contractaddress={token_address}&address={wallet_address}&sort=desc&page=1&offset=50"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    transactions = data.get('result', [])
                    
                    recent_txs = []
                    current_time = int(time.time())
                    
                    for tx in transactions:
                        tx_time = int(tx.get('timeStamp', '0'))
                        if current_time - tx_time <= 3600:
                            recent_txs.append(tx)
                    
                    return recent_txs
            
            return []
            
        except Exception as e:
            return []
    
    async def create_whale_transaction(self, tx_data: Dict, whale_address: str) -> Optional[WhaleTransaction]:
        try:
            token_amount = float(tx_data.get('value', '0'))
            decimals = int(tx_data.get('tokenDecimal', '18'))
            token_amount = token_amount / (10 ** decimals)
            
            token_price = await self.get_token_price(tx_data.get('contractAddress'))
            amount_usd = token_amount * token_price
            
            if amount_usd < self.whale_thresholds['transaction_size']:
                return None
            
            transaction_type = 'buy' if tx_data.get('to', '').lower() == whale_address.lower() else 'sell'
            
            price_impact = await self.estimate_price_impact(
                tx_data.get('contractAddress'), 
                amount_usd, 
                transaction_type
            )
            
            return WhaleTransaction(
                wallet_address=whale_address,
                token_address=tx_data.get('contractAddress'),
                transaction_type=transaction_type,
                amount_usd=amount_usd,
                token_amount=token_amount,
                price_impact=price_impact,
                timestamp=int(tx_data.get('timeStamp', '0')),
                block_number=int(tx_data.get('blockNumber', '0')),
                transaction_hash=tx_data.get('hash'),
                gas_used=int(tx_data.get('gasUsed', '0'))
            )
            
        except Exception as e:
            return None
    
    async def get_token_price(self, token_address: str) -> float:
        try:
            url = f"https://api.coingecko.com/api/v3/simple/token_price/ethereum?contract_addresses={token_address}&vs_currencies=usd"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get(token_address.lower(), {}).get('usd', 0.0)
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    async def estimate_price_impact(self, token_address: str, amount_usd: float, transaction_type: str) -> float:
        try:
            liquidity = await self.get_token_liquidity(token_address)
            
            if liquidity == 0:
                return 0.0
            
            impact_ratio = amount_usd / liquidity
            
            base_impact = np.sqrt(impact_ratio) * 0.5
            
            if transaction_type == 'sell':
                base_impact *= 1.2
            
            return min(base_impact, 0.5)
            
        except Exception as e:
            return 0.0
    
    async def get_token_liquidity(self, token_address: str) -> float:
        return 1000000.0
    
    async def close(self):
        if self.session:
            await self.session.close()

whale_analyzer = WhaleWalletAnalyzer()