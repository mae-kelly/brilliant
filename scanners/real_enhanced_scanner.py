import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque
import logging
import numpy as np

# Import real data components
try:
    from data.real_websocket_feeds import real_data_engine
    from analyzers.real_blockchain_analyzer import real_blockchain_analyzer
    from analyzers.real_honeypot_detector import real_honeypot_detector
except ImportError as e:
    print(f"Import warning: {e}")

@dataclass
class RealTokenDetection:
    address: str
    chain: str
    dex: str
    price: float
    volume_24h: float
    liquidity_usd: float
    price_change_1h: float
    momentum_score: float
    velocity: float
    safety_score: float
    verified: bool
    detected_at: float

class RealEnhancedScanner:
    def __init__(self):
        self.target_tokens_per_day = 10000
        self.discovered_tokens = set()
        self.momentum_signals = asyncio.Queue(maxsize=10000)
        self.workers = []
        
        self.stats = {
            'tokens_scanned': 0,
            'signals_generated': 0,
            'real_data_points': 0,
            'start_time': time.time()
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize real enhanced scanner"""
        self.logger.info("🔥 Initializing REAL Enhanced Scanner...")
        
        # Initialize real data components
        try:
            await real_data_engine.initialize()
            await real_honeypot_detector.initialize()
            self.logger.info("✅ Real data components initialized")
        except Exception as e:
            self.logger.warning(f"Real data initialization issue: {e}")
        
        # Start workers
        for i in range(10):
            task = asyncio.create_task(self.real_data_processor(i))
            self.workers.append(task)
        
        for i in range(5):
            task = asyncio.create_task(self.real_momentum_analyzer(i))
            self.workers.append(task)
        
        task = asyncio.create_task(self.performance_monitor())
        self.workers.append(task)
        
        self.logger.info(f"✅ Started {len(self.workers)} real data workers")

    async def real_data_processor(self, worker_id: int):
        """Process real data from feeds"""
        while True:
            try:
                # Get real tokens from data engine
                for chain in ['ethereum', 'arbitrum', 'polygon', 'optimism']:
                    await self.scan_chain_real_data(chain)
                
                await asyncio.sleep(3)
                
            except Exception as e:
                self.logger.debug(f"Real data processor {worker_id} error: {e}")
                await asyncio.sleep(10)

    async def scan_chain_real_data(self, chain: str):
        """Scan chain for real token data"""
        try:
            # Get tokens with recent activity
            tokens_to_analyze = []
            
            for token_key, cache in real_data_engine.live_tokens.items():
                if token_key.startswith(f"{chain}_") and cache['prices']:
                    token_address = token_key.split('_', 1)[1]
                    
                    if len(cache['prices']) >= 3 and token_address not in self.discovered_tokens:
                        real_token_data = await real_data_engine.get_real_token_data(token_address, chain)
                        if real_token_data:
                            tokens_to_analyze.append(real_token_data)
            
            # Analyze each token
            for token_data in tokens_to_analyze[:20]:  # Limit to prevent overload
                await self.analyze_real_token(token_data)
                self.stats['real_data_points'] += 1
                
        except Exception as e:
            self.logger.debug(f"Chain scan error {chain}: {e}")

    async def analyze_real_token(self, token_data: Dict):
        """Analyze real token for trading signals"""
        try:
            token_address = token_data['address']
            chain = token_data['chain']
            
            if token_address in self.discovered_tokens:
                return
            
            # Calculate real momentum
            price_history = token_data['price_history']
            volume_history = token_data['volume_history']
            
            if len(price_history) < 3:
                return
            
            momentum_score = self.calculate_real_momentum(price_history, volume_history)
            velocity = self.calculate_real_velocity(price_history)
            price_change = self.calculate_real_price_change(price_history)
            
            # Get blockchain info
            try:
                blockchain_info = await real_blockchain_analyzer.get_real_token_info(token_address, chain)
                liquidity_info = await real_blockchain_analyzer.analyze_real_liquidity(token_address, chain)
            except:
                # Fallback data
                blockchain_info = {'symbol': 'UNKNOWN', 'verified': False}
                liquidity_info = {'liquidity_usd': 1000}
            
            # Safety check
            try:
                safety_check = await real_honeypot_detector.check_real_honeypot(token_address, chain)
                safety_score = 1.0 - safety_check['risk_score']
            except:
                safety_score = 0.5  # Neutral if check fails
            
            # Signal criteria
            if (momentum_score > 0.7 and 
                abs(price_change) > 5 and 
                safety_score > 0.6 and
                liquidity_info.get('liquidity_usd', 0) > 20000):
                
                detection = RealTokenDetection(
                    address=token_address,
                    chain=chain,
                    dex='real_dex',
                    price=token_data['current_price'],
                    volume_24h=sum(volume_history) if volume_history else 0,
                    liquidity_usd=liquidity_info.get('liquidity_usd', 0),
                    price_change_1h=price_change,
                    momentum_score=momentum_score,
                    velocity=velocity,
                    safety_score=safety_score,
                    verified=blockchain_info.get('verified', False),
                    detected_at=time.time()
                )
                
                try:
                    self.momentum_signals.put_nowait(detection)
                    self.discovered_tokens.add(token_address)
                    self.stats['signals_generated'] += 1
                    
                    self.logger.info(
                        f"🎯 REAL SIGNAL: {blockchain_info.get('symbol', 'UNKNOWN')} "
                        f"Momentum: {momentum_score:.3f} "
                        f"Change: {price_change:.2f}% "
                        f"Safety: {safety_score:.3f}"
                    )
                except:
                    pass
            
            self.stats['tokens_scanned'] += 1
            
        except Exception as e:
            self.logger.debug(f"Real token analysis error: {e}")

    def calculate_real_momentum(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate momentum from real price data"""
        try:
            if len(prices) < 2:
                return 0.0
            
            prices_arr = np.array(prices)
            volumes_arr = np.array(volumes) if volumes else np.ones_like(prices)
            
            # Price momentum
            returns = np.diff(prices_arr) / (prices_arr[:-1] + 1e-10)
            price_momentum = np.mean(returns) * np.sqrt(len(returns))
            
            # Volume confirmation
            if len(volumes_arr) > 1:
                volume_momentum = (volumes_arr[-1] - np.mean(volumes_arr[:-1])) / (np.mean(volumes_arr[:-1]) + 1)
                combined = 0.7 * price_momentum + 0.3 * volume_momentum
            else:
                combined = price_momentum
            
            return max(0, np.tanh(combined * 3))
            
        except Exception:
            return 0.0

    def calculate_real_velocity(self, prices: List[float]) -> float:
        """Calculate velocity from real prices"""
        try:
            if len(prices) < 2:
                return 0.0
            
            velocity = (prices[-1] - prices[0]) / (len(prices) * (prices[0] + 1e-10))
            return abs(velocity)
            
        except Exception:
            return 0.0

    def calculate_real_price_change(self, prices: List[float]) -> float:
        """Calculate price change percentage"""
        try:
            if len(prices) < 2:
                return 0.0
            
            return ((prices[-1] - prices[0]) / prices[0]) * 100
            
        except Exception:
            return 0.0

    async def real_momentum_analyzer(self, worker_id: int):
        """Analyze momentum signals"""
        while True:
            try:
                detection = await self.momentum_signals.get()
                
                # Additional analysis
                enhanced_score = await self.enhance_with_real_data(detection)
                detection.momentum_score = enhanced_score
                
                if enhanced_score > 0.8:
                    self.logger.info(
                        f"🔥 HIGH CONFIDENCE: {detection.address[:8]}... "
                        f"Score: {enhanced_score:.3f} Chain: {detection.chain}"
                    )
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                await asyncio.sleep(1)

    async def enhance_with_real_data(self, detection: RealTokenDetection) -> float:
        """Enhance score with additional real data"""
        try:
            # Get fresh data
            fresh_data = await real_data_engine.get_real_token_data(detection.address, detection.chain)
            
            if fresh_data and fresh_data['price_history']:
                recent_prices = fresh_data['price_history'][-5:]
                if len(recent_prices) >= 3:
                    recent_momentum = self.calculate_real_momentum(recent_prices, [])
                    
                    if recent_momentum > detection.momentum_score:
                        return min(detection.momentum_score * 1.1, 1.0)
            
            return detection.momentum_score
            
        except Exception:
            return detection.momentum_score

    async def get_real_signals(self, max_signals: int = 10) -> List[RealTokenDetection]:
        """Get real trading signals"""
        signals = []
        for _ in range(max_signals):
            try:
                signal = await asyncio.wait_for(self.momentum_signals.get(), timeout=0.1)
                signals.append(signal)
            except asyncio.TimeoutError:
                break
        return signals

    async def performance_monitor(self):
        """Monitor real scanner performance"""
        while True:
            try:
                runtime = time.time() - self.stats['start_time']
                tokens_per_hour = self.stats['tokens_scanned'] / (runtime / 3600) if runtime > 0 else 0
                daily_projection = tokens_per_hour * 24
                
                self.logger.info("=" * 60)
                self.logger.info("🔥 REAL ENHANCED SCANNER PERFORMANCE")
                self.logger.info("=" * 60)
                self.logger.info(f"⏱️  Runtime: {runtime/60:.1f} minutes")
                self.logger.info(f"🔍 Real tokens scanned: {self.stats['tokens_scanned']:,}")
                self.logger.info(f"📊 Real data points: {self.stats['real_data_points']:,}")
                self.logger.info(f"📈 Real signals: {self.stats['signals_generated']:,}")
                self.logger.info(f"🚀 Rate: {tokens_per_hour:.0f} tokens/hour")
                self.logger.info(f"🎯 Daily projection: {daily_projection:.0f} tokens/day")
                self.logger.info(f"🏆 Target: {daily_projection/self.target_tokens_per_day*100:.1f}%")
                self.logger.info(f"💾 Live tokens: {len(real_data_engine.live_tokens):,}")
                self.logger.info("=" * 60)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                await asyncio.sleep(120)

    async def shutdown(self):
        """Shutdown scanner"""
        self.logger.info("🛑 Shutting down real enhanced scanner...")
        
        for worker in self.workers:
            worker.cancel()
        
        try:
            await real_data_engine.shutdown()
            await real_honeypot_detector.shutdown()
        except:
            pass
        
        self.logger.info("✅ Real scanner shutdown complete")

# Global scanner instance
real_enhanced_scanner = RealEnhancedScanner()
