import asyncio
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

try:
    from universal_dex_executor import universal_executor
    from live_data_streams_fixed import live_streams
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available")

@dataclass
class ProductionSignal:
    token_address: str
    chain: str
    price: float
    price_change: float
    volume: float
    momentum_score: float
    confidence: float
    timestamp: float

class FixedProductionTradingSystem:
    def __init__(self):
        self.running = False
        self.start_time = None
        
        self.signals_processed = 0
        self.trades_executed = 0
        self.total_profit = 0.0
        self.portfolio_value = 10.0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        self.logger.info("🚀 Initializing Fixed Production Trading System...")
        
        if hasattr(universal_executor, 'validate_connection'):
            connection_ok = universal_executor.validate_connection()
            self.logger.info(f"📡 Web3 connection: {'✅' if connection_ok else '⚠️ Simulation mode'}")
        
        await live_streams.initialize()
        
        self.logger.info("✅ Fixed production system initialized")
        return True

    async def start_trading(self):
        self.running = True
        self.start_time = time.time()
        
        self.logger.info("🎯 Starting fixed production trading...")
        
        trading_task = asyncio.create_task(self.trading_loop())
        monitoring_task = asyncio.create_task(self.monitoring_loop())
        
        try:
            await asyncio.gather(trading_task, monitoring_task)
        except KeyboardInterrupt:
            await self.shutdown()

    async def trading_loop(self):
        while self.running:
            try:
                raw_signals = await live_streams.get_signals(max_signals=3)
                
                for raw_signal in raw_signals:
                    production_signal = await self.analyze_signal(raw_signal)
                    
                    if production_signal and production_signal.confidence > 0.7:
                        await self.execute_trade(production_signal)
                        
                self.signals_processed += len(raw_signals)
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)

    async def analyze_signal(self, raw_signal: Dict) -> Optional[ProductionSignal]:
        try:
            token_address = raw_signal['token_address']
            chain = raw_signal['chain']
            price = raw_signal['current_price']
            price_change = raw_signal['price_change']
            volume = raw_signal['volume']
            
            momentum_score = abs(price_change) * min(volume / 10000, 2.0)
            
            confidence = 0.5
            if volume > 50000:
                confidence += 0.2
            if abs(price_change) > 0.1:
                confidence += 0.2
            if momentum_score > 0.1:
                confidence += 0.1
                
            confidence = min(confidence, 1.0)
            
            if confidence > 0.6 and momentum_score > 0.05:
                return ProductionSignal(
                    token_address=token_address,
                    chain=chain,
                    price=price,
                    price_change=price_change,
                    volume=volume,
                    momentum_score=momentum_score,
                    confidence=confidence,
                    timestamp=time.time()
                )
                
        except Exception as e:
            self.logger.error(f"Signal analysis error: {e}")
            
        return None

    async def execute_trade(self, signal: ProductionSignal):
        try:
            if signal.price_change > 0:
                position_size = min(1.0, self.portfolio_value * 0.1)
                
                self.logger.info(
                    f"🔥 BUY SIGNAL: {signal.token_address[:8]}... "
                    f"Change: {signal.price_change:.3f} "
                    f"Confidence: {signal.confidence:.3f} "
                    f"Size: ${position_size:.2f}"
                )
                
                tx_hash = universal_executor.execute_trade(
                    signal.token_address, 'buy', position_size
                )
                
                if tx_hash:
                    self.trades_executed += 1
                    self.logger.info(f"✅ Trade executed: {tx_hash[:10]}...")
                    
                    await asyncio.sleep(30)
                    
                    sell_tx = universal_executor.execute_trade(
                        signal.token_address, 'sell', position_size
                    )
                    
                    if sell_tx:
                        profit = position_size * signal.price_change * 0.8
                        self.total_profit += profit
                        self.portfolio_value += profit
                        
                        self.logger.info(
                            f"💰 SELL completed: Profit ${profit:.2f} "
                            f"Portfolio: ${self.portfolio_value:.2f}"
                        )
                    
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")

    async def monitoring_loop(self):
        while self.running:
            try:
                uptime = time.time() - self.start_time if self.start_time else 0
                
                self.logger.info("=" * 50)
                self.logger.info("📊 FIXED PRODUCTION SYSTEM STATUS")
                self.logger.info("=" * 50)
                self.logger.info(f"⏱️  Uptime: {uptime/60:.1f} minutes")
                self.logger.info(f"🔍 Signals processed: {self.signals_processed}")
                self.logger.info(f"💼 Trades executed: {self.trades_executed}")
                self.logger.info(f"💰 Portfolio value: ${self.portfolio_value:.2f}")
                self.logger.info(f"📈 Total profit: ${self.total_profit:.2f}")
                self.logger.info(f"🎯 ROI: {((self.portfolio_value - 10.0) / 10.0) * 100:.2f}%")
                self.logger.info("=" * 50)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)

    async def shutdown(self):
        self.logger.info("🛑 Shutting down fixed production system...")
        self.running = False
        
        await live_streams.shutdown()
        
        final_roi = ((self.portfolio_value - 10.0) / 10.0) * 100
        
        self.logger.info("📊 FINAL FIXED PRODUCTION REPORT")
        self.logger.info(f"Signals processed: {self.signals_processed}")
        self.logger.info(f"Trades executed: {self.trades_executed}")
        self.logger.info(f"Final portfolio: ${self.portfolio_value:.2f}")
        self.logger.info(f"Total profit: ${self.total_profit:.2f}")
        self.logger.info(f"Final ROI: {final_roi:.2f}%")
        
        self.logger.info("✅ Fixed production shutdown complete")

fixed_system = FixedProductionTradingSystem()

async def main():
    try:
        success = await fixed_system.initialize()
        if success:
            await fixed_system.start_trading()
        else:
            print("❌ System initialization failed")
    except KeyboardInterrupt:
        await fixed_system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
