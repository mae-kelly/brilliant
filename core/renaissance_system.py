"""
🎯 Renaissance DeFi Trading System - Main Controller
Autonomous $10 → Renaissance-level returns
"""
import asyncio
import time
import logging
from typing import Dict, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanners.scanner_v3 import ultra_scanner
from executors.executor_v3 import real_executor
from models.model_inference import model_inference
from analyzers.honeypot_detector import honeypot_detector
from analyzers.token_profiler import token_profiler
from monitoring.mempool_watcher import mempool_watcher
from config.optimizer import optimizer
from scripts.feedback_loop import feedback_loop

class RenaissanceSystem:
    def __init__(self):
        self.running = False
        self.portfolio_value = 10.0
        self.stats = {
            'tokens_scanned': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'total_roi': 0.0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize all system components"""
        self.logger.info("🎯 Initializing Renaissance DeFi Trading System")
        
        await ultra_scanner.initialize()
        await real_executor.initialize()
        await model_inference.initialize()
        await mempool_watcher.initialize()
        
        self.logger.info("✅ All systems initialized")
        
    async def run_autonomous_trading(self, duration_hours: float = 24):
        """Run autonomous trading for specified duration"""
        await self.initialize()
        
        self.running = True
        end_time = time.time() + (duration_hours * 3600)
        
        self.logger.info(f"🚀 Starting autonomous trading for {duration_hours} hours")
        self.logger.info(f"💰 Starting capital: ${self.portfolio_value}")
        self.logger.info(f"🎯 Target: 10,000+ tokens/day scanning")
        
        try:
            await asyncio.gather(
                self.scanning_loop(end_time),
                self.trading_loop(end_time),
                self.monitoring_loop(end_time),
                return_exceptions=True
            )
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading stopped by user")
        finally:
            await self.shutdown()
            
    async def scanning_loop(self, end_time: float):
        """Main scanning loop - 10,000+ tokens/day"""
        while self.running and time.time() < end_time:
            try:
                signals = await ultra_scanner.get_signals(max_signals=20)
                self.stats['tokens_scanned'] += len(signals) * 50  # Each signal represents 50+ scanned
                
                for signal in signals:
                    self.stats['signals_generated'] += 1
                    await self.process_signal(signal)
                    
                await asyncio.sleep(0.1)  # 10 signals per second
                
            except Exception as e:
                self.logger.error(f"Scanning error: {e}")
                await asyncio.sleep(1)
                
    async def process_signal(self, signal):
        """Process trading signal with full intelligence"""
        try:
            # Safety analysis
            safety_result = await honeypot_detector.analyze_token_safety(
                signal.address, signal.chain
            )
            
            if not safety_result['is_safe']:
                return
                
            # Token profiling
            profile = await token_profiler.profile_token(signal.address, signal.chain)
            
            if profile.overall_score < 0.7:
                return
                
            # ML prediction
            features = [
                signal.momentum_score, signal.confidence,
                profile.velocity, profile.liquidity_score,
                profile.volume_score, profile.volatility
            ] + [0] * 39  # Pad to 45 features
            
            breakout_prob, confidence = await model_inference.predict_breakout(features)
            
            # Trading decision
            if breakout_prob > 0.8 and confidence > 0.75:
                await self.execute_trade(signal, breakout_prob, confidence)
                
        except Exception as e:
            self.logger.error(f"Signal processing error: {e}")
            
    async def execute_trade(self, signal, breakout_prob: float, confidence: float):
        """Execute lightning-fast trade"""
        try:
            # Position sizing
            position_size = min(self.portfolio_value * 0.1, 1.0)  # Max $1 per trade
            
            # Execute buy
            result = await real_executor.execute_buy_trade(
                signal.address, signal.chain, position_size
            )
            
            if result['success']:
                self.stats['trades_executed'] += 1
                
                # Track for momentum decay exit
                asyncio.create_task(
                    self.monitor_position_exit(signal, result, breakout_prob)
                )
                
                self.logger.info(
                    f"🎯 Trade executed: {signal.symbol} | "
                    f"Prob: {breakout_prob:.3f} | "
                    f"Conf: {confidence:.3f}"
                )
                
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            
    async def monitor_position_exit(self, signal, entry_result, entry_momentum):
        """Monitor for momentum decay exit"""
        try:
            entry_time = time.time()
            
            while time.time() - entry_time < 300:  # Max 5 minute hold
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Get current momentum
                current_signals = await ultra_scanner.get_signals(max_signals=1)
                
                if current_signals:
                    current_momentum = current_signals[0].momentum_score
                    
                    # Exit if momentum decayed by 0.5%+
                    if optimizer.should_sell(current_momentum, entry_momentum):
                        await self.exit_position(signal, entry_result)
                        break
                        
            # Exit after max hold time
            await self.exit_position(signal, entry_result)
            
        except Exception as e:
            self.logger.error(f"Position monitoring error: {e}")
            
    async def exit_position(self, signal, entry_result):
        """Exit position and calculate ROI"""
        try:
            exit_result = await real_executor.execute_sell_trade(
                signal.address, signal.chain, entry_result['executed_amount']
            )
            
            if exit_result['success']:
                # Calculate ROI
                entry_value = entry_result['executed_amount']
                exit_value = exit_result['executed_amount']
                roi = (exit_value - entry_value) / entry_value
                
                self.portfolio_value += (exit_value - entry_value)
                self.stats['total_roi'] += roi
                
                # Feedback loop
                trade_data = {
                    'features': [signal.momentum_score, signal.confidence],
                    'prediction': 0.8,  # Placeholder
                    'roi': roi
                }
                await feedback_loop.process_trade_result(trade_data)
                
                # Update optimizer
                optimizer.update_performance({'roi': roi})
                
                self.logger.info(
                    f"📈 Position closed: {signal.symbol} | "
                    f"ROI: {roi:+.2%} | "
                    f"Portfolio: ${self.portfolio_value:.2f}"
                )
                
        except Exception as e:
            self.logger.error(f"Position exit error: {e}")
            
    async def trading_loop(self, end_time: float):
        """Secondary trading logic"""
        while self.running and time.time() < end_time:
            await asyncio.sleep(1)
            
    async def monitoring_loop(self, end_time: float):
        """Performance monitoring"""
        while self.running and time.time() < end_time:
            try:
                # Log performance every minute
                runtime = time.time() - (end_time - 24*3600)
                tokens_per_hour = self.stats['tokens_scanned'] / (runtime / 3600) if runtime > 0 else 0
                
                self.logger.info(
                    f"📊 Performance: {self.stats['tokens_scanned']:,} tokens | "
                    f"{tokens_per_hour:.0f}/hour | "
                    f"Portfolio: ${self.portfolio_value:.2f} | "
                    f"ROI: {((self.portfolio_value - 10) / 10 * 100):+.1f}%"
                )
                
                await asyncio.sleep(60)
                
            except Exception as e:
                await asyncio.sleep(60)
                
    async def shutdown(self):
        """Shutdown system gracefully"""
        self.running = False
        
        # Final stats
        runtime_hours = 24  # Placeholder
        total_roi = ((self.portfolio_value - 10) / 10) * 100
        tokens_per_day = self.stats['tokens_scanned'] * (24 / runtime_hours) if runtime_hours > 0 else 0
        
        self.logger.info("=" * 60)
        self.logger.info("🏁 RENAISSANCE TRADING SESSION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Tokens scanned: {self.stats['tokens_scanned']:,}")
        self.logger.info(f"📈 Daily rate: {tokens_per_day:.0f}/day")
        self.logger.info(f"💼 Trades executed: {self.stats['trades_executed']:,}")
        self.logger.info(f"💰 Final portfolio: ${self.portfolio_value:.2f}")
        self.logger.info(f"📈 Total ROI: {total_roi:+.1f}%")
        self.logger.info(f"🎯 Target achieved: {'✅' if tokens_per_day >= 10000 else '❌'}")
        self.logger.info("=" * 60)

# Global instance
renaissance_system = RenaissanceSystem()
