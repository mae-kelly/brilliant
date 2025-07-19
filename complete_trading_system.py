
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

#!/usr/bin/env python3
"""
COMPLETE RENAISSANCE TRADING SYSTEM
Integrates all components for autonomous trading
"""

import asyncio
import time
import logging
from websocket_scanner_working import WorkingWebSocketScanner
from enhanced_momentum_analyzer import EnhancedMomentumAnalyzer
from intelligent_trade_executor import IntelligentTradeExecutor

class RenaissanceTradingSystem:
    def __init__(self):
        # Core components
        self.scanner = WorkingWebSocketScanner()
        self.analyzer = EnhancedMomentumAnalyzer()
        self.executor = IntelligentTradeExecutor()
        
        # System state
        self.running = False
        self.start_time = None
        
        # Performance tracking
        self.tokens_analyzed = 0
        self.signals_generated = 0
        self.trades_executed = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize all system components"""
        self.logger.info("🚀 Initializing Renaissance Trading System...")
        
        # Initialize scanner
        await self.scanner.initialize()
        
        self.logger.info("✅ All components initialized")
        
    async def start_autonomous_trading(self):
        """Start autonomous trading loop"""
        self.running = True
        self.start_time = time.time()
        
        self.logger.info("🎯 Starting autonomous trading...")
        self.logger.info(f"💰 Starting portfolio: ${self.executor.portfolio_value:.2f}")
        
        # Start main trading loop
        trading_task = asyncio.create_task(self.trading_loop())
        monitoring_task = asyncio.create_task(self.monitoring_loop())
        
        try:
            await asyncio.gather(trading_task, monitoring_task)
        except KeyboardInterrupt:
            await self.shutdown()
            
    async def trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Get momentum signals from scanner
                raw_signals = await self.scanner.get_signals(max_signals=5)
                
                for raw_signal in raw_signals:
                    # Enhanced momentum analysis
                    token_key = f"{raw_signal.chain}_{raw_signal.address}"
                    token_data = self.scanner.token_data.get(token_key, {})
                    
                    if 'prices' in token_data and len(token_data['prices']) >= 10:
                        # Prepare data for enhanced analysis
                        enhanced_token_data = {
                            'address': raw_signal.address,
                            'chain': raw_signal.chain,
                            'dex': raw_signal.dex,
                            'liquidity_usd': raw_signal.liquidity_usd
                        }
                        
                        price_history = list(token_data['prices'])
                        volume_history = list(token_data['volumes'])
                        
                        # Enhanced analysis
                        enhanced_signal = await self.analyzer.analyze_enhanced_momentum(
                            enhanced_token_data, price_history, volume_history
                        )
                        
                        if enhanced_signal:
                            self.signals_generated += 1
                            self.logger.info(
                                f"🎯 Enhanced signal: {enhanced_signal.address[:8]}... "
                                f"Quality: {enhanced_signal.signal_quality} "
                                f"Score: {enhanced_signal.momentum_score:.3f} "
                                f"Confidence: {enhanced_signal.confidence:.3f}"
                            )
                            
                            # Execute trade
                            trade_result = await self.executor.execute_trade_strategy(enhanced_signal)
                            
                            if trade_result:
                                self.trades_executed += 1
                                self.logger.info(
                                    f"💼 Trade completed: "
                                    f"ROI: {trade_result.roi_percent:.2f}% "
                                    f"Profit: ${trade_result.profit_usd:.2f}"
                                )
                
                # Brief pause between cycles
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)
                
    async def monitoring_loop(self):
        """System monitoring and reporting loop"""
        while self.running:
            try:
                # Get system stats
                uptime = time.time() - self.start_time if self.start_time else 0
                scanner_stats = await self.scanner.get_discovery_stats() if hasattr(self.scanner, 'get_discovery_stats') else {}
                executor_stats = self.executor.get_performance_stats()
                
                # System performance report
                self.logger.info("=" * 60)
                self.logger.info("📊 RENAISSANCE TRADING SYSTEM STATUS")
                self.logger.info("=" * 60)
                self.logger.info(f"⏱️  Uptime: {uptime/60:.1f} minutes")
                self.logger.info(f"🔍 Tokens discovered: {scanner_stats.get('discovered_tokens', 0)}")
                self.logger.info(f"🧠 Signals generated: {self.signals_generated}")
                self.logger.info(f"💼 Trades executed: {self.trades_executed}")
                self.logger.info(f"💰 Portfolio value: ${executor_stats['portfolio_value']:.2f}")
                self.logger.info(f"📈 Total ROI: {executor_stats['roi_percent']:.2f}%")
                self.logger.info(f"🎯 Win rate: {executor_stats['win_rate']:.1f}%")
                self.logger.info(f"💵 Total profit: ${executor_stats['total_profit']:.2f}")
                self.logger.info(f"🔄 Active trades: {executor_stats['active_trades']}")
                self.logger.info("=" * 60)
                
                await asyncio.sleep(30)  # Report every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def shutdown(self):
        """Gracefully shutdown the system"""
        self.logger.info("🛑 Shutting down Renaissance Trading System...")
        self.running = False
        
        # Shutdown components
        await self.scanner.shutdown()
        
        # Final performance report
        final_stats = self.executor.get_performance_stats()
        self.logger.info("📊 FINAL PERFORMANCE REPORT")
        self.logger.info(f"Total trades: {final_stats['total_trades']}")
        self.logger.info(f"Win rate: {final_stats['win_rate']:.1f}%")
        self.logger.info(f"Final portfolio: ${final_stats['portfolio_value']:.2f}")
        self.logger.info(f"Total ROI: {final_stats['roi_percent']:.2f}%")
        
        self.logger.info("✅ Shutdown complete")

async def main():
    """Main entry point"""
    system = RenaissanceTradingSystem()
    
    try:
        await system.initialize()
        await system.start_autonomous_trading()
    except KeyboardInterrupt:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
