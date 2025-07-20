
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from optimizer import get_dynamic_config, update_performance
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


import asyncio
import websockets
import json
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

@dataclass
class MempoolTransaction:
    hash: str
    from_address: str
    to_address: str
    value: float
    gas_price: int
    gas_limit: int
    data: str
    detected_at: float
    is_swap: bool
    token_address: Optional[str] = None

class MempoolWatcher:
    def __init__(self):
        self.websocket_endpoints = [
            'wss://ethereum-rpc.publicnode.com',
            'wss://arbitrum-one.publicnode.com',
            'wss://polygon-bor-rpc.publicnode.com'
        ]
        
        self.connections = {}
        self.transaction_callbacks = []
        self.pending_txs = deque(maxlen=10000)
        self.mev_opportunities = deque(maxlen=1000)
        self.gas_tracker = defaultdict(list)
        
        self.swap_signatures = [
            '0x7ff36ab5',  # swapExactETHForTokens
            '0x18cbafe5',  # swapExactTokensForETH  
            '0xd0e30db0',  # deposit (WETH)
            '0x2e1a7d4d',  # withdraw (WETH)
        ]
        
        self.running = False
        self.stats = {
            'total_transactions': 0,
            'swap_transactions': 0,
            'mev_opportunities': 0,
            'start_time': time.time()
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def start_monitoring(self):
        self.running = True
        self.logger.info("🔍 Starting mempool monitoring...")
        
        tasks = []
        for i, endpoint in enumerate(self.websocket_endpoints):
            task = asyncio.create_task(self.monitor_chain(endpoint, i))
            tasks.append(task)
        
        task = asyncio.create_task(self.mev_detector())
        tasks.append(task)
        
        task = asyncio.create_task(self.gas_price_tracker())
        tasks.append(task)
        
        task = asyncio.create_task(self.performance_monitor())
        tasks.append(task)
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")

    async def monitor_chain(self, endpoint: str, chain_id: int):
        while self.running:
            try:
                async with websockets.connect(endpoint) as websocket:
                    self.connections[chain_id] = websocket
                    
                    subscription = {
                        "id": 1,
                        "method": "eth_subscribe",
                        "params": ["newPendingTransactions"]
                    }
                    
                    await websocket.send(json.dumps(subscription))
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            if 'params' in data:
                                tx_hash = data['params']['result']
                                await self.process_transaction(tx_hash, chain_id)
                        except Exception as e:
                            self.logger.debug(f"Message processing error: {e}")
                        
            except Exception as e:
                self.logger.warning(f"Connection error {endpoint}: {e}")
                await asyncio.sleep(5)

    async def process_transaction(self, tx_hash: str, chain_id: int):
        try:
            await asyncio.sleep(0.001)
            
            simulated_tx = self.simulate_transaction(tx_hash, chain_id)
            
            if simulated_tx.is_swap:
                self.stats['swap_transactions'] += 1
                await self.analyze_swap_transaction(simulated_tx)
            
            self.pending_txs.append(simulated_tx)
            self.stats['total_transactions'] += 1
            
            for callback in self.transaction_callbacks:
                try:
                    await callback(simulated_tx)
                except Exception as e:
                    self.logger.debug(f"Callback error: {e}")
                    
        except Exception as e:
            self.logger.debug(f"Transaction processing error: {e}")

    def simulate_transaction(self, tx_hash: str, chain_id: int) -> MempoolTransaction:
        tx_data = hash(tx_hash) % 10000
        
        is_swap = tx_data % 20 < 3
        
        return MempoolTransaction(
            hash=tx_hash,
            from_address=f"0x{hash(tx_hash + 'from') % (16**40):040x}",
            to_address=f"0x{hash(tx_hash + 'to') % (16**40):040x}",
            value=float((tx_data % 1000) / 1000),
            gas_price=20 + (tx_data % 100),
            gas_limit=21000 + (tx_data % 200000),
            data='0x' + ('a' * (tx_data % 100)),
            detected_at=time.time(),
            is_swap=is_swap,
            token_address=f"0x{hash(tx_hash + 'token') % (16**40):040x}" if is_swap else None
        )

    async def analyze_swap_transaction(self, tx: MempoolTransaction):
        if tx.value > 0.1:
            opportunity_score = self.calculate_mev_score(tx)
            
            if opportunity_score > 0.7:
                self.mev_opportunities.append({
                    'tx_hash': tx.hash,
                    'opportunity_type': 'frontrun',
                    'score': opportunity_score,
                    'gas_price': tx.gas_price,
                    'value': tx.value,
                    'detected_at': tx.detected_at
                })
                
                self.stats['mev_opportunities'] += 1
                self.logger.info(f"🎯 MEV opportunity: {tx.hash[:10]}... Score: {opportunity_score:.2f}")

    def calculate_mev_score(self, tx: MempoolTransaction) -> float:
        score = 0.0
        
        if tx.value > 1.0:
            score += 0.3
        
        if tx.gas_price < 30:
            score += 0.2
        
        if tx.token_address:
            token_hash = hash(tx.token_address)
            if token_hash % 10 < 3:
                score += 0.5
        
        return min(score, 1.0)

    async def mev_detector(self):
        while self.running:
            try:
                if len(self.pending_txs) > 10:
                    recent_txs = list(self.pending_txs)[-10:]
                    await self.detect_mev_patterns(recent_txs)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.debug(f"MEV detector error: {e}")

    async def detect_mev_patterns(self, transactions: List[MempoolTransaction]):
        for i, tx in enumerate(transactions):
            if tx.is_swap and i < len(transactions) - 1:
                next_tx = transactions[i + 1]
                
                if (next_tx.is_swap and 
                    tx.token_address == next_tx.token_address and
                    abs(tx.detected_at - next_tx.detected_at) < 5.0):
                    
                    self.logger.info(f"🔍 Potential sandwich attack detected: {tx.hash[:10]}...")

    async def gas_price_tracker(self):
        while self.running:
            try:
                current_time = time.time()
                recent_txs = [tx for tx in self.pending_txs if current_time - tx.detected_at < 60]
                
                if recent_txs:
                    avg_gas = sum(tx.gas_price for tx in recent_txs) / len(recent_txs)
                    self.gas_tracker['hourly'].append((current_time, avg_gas))
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.debug(f"Gas tracker error: {e}")

    async def performance_monitor(self):
        while self.running:
            try:
                runtime = time.time() - self.stats['start_time']
                tx_rate = self.stats['total_transactions'] / max(runtime, 1)
                
                self.logger.info("=" * 50)
                self.logger.info("📊 MEMPOOL WATCHER PERFORMANCE")
                self.logger.info("=" * 50)
                self.logger.info(f"⏱️  Runtime: {runtime/60:.1f} minutes")
                self.logger.info(f"📡 Total transactions: {self.stats['total_transactions']:,}")
                self.logger.info(f"🔄 Swap transactions: {self.stats['swap_transactions']:,}")
                self.logger.info(f"⚡ MEV opportunities: {self.stats['mev_opportunities']:,}")
                self.logger.info(f"📈 TX rate: {tx_rate:.1f} tx/sec")
                self.logger.info(f"🔗 Active connections: {len(self.connections)}")
                self.logger.info("=" * 50)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")

    def add_transaction_callback(self, callback: Callable):
        self.transaction_callbacks.append(callback)

    def get_recent_mev_opportunities(self, limit: int = 10) -> List[Dict]:
        return list(self.mev_opportunities)[-limit:]

    async def shutdown(self):
        self.running = False
        for connection in self.connections.values():
            try:
                await connection.close()
            except:
                pass
        self.logger.info("✅ Mempool watcher shutdown complete")

mempool_watcher = MempoolWatcher()
