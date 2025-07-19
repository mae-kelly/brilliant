#!/usr/bin/env python3
"""
Test the complete optimized trading system
"""

import asyncio
import time
from complete_trading_system import RenaissanceTradingSystem

async def test_complete_system():
    print("🧪 Testing Complete Renaissance Trading System")
    print("=" * 50)
    
    system = RenaissanceTradingSystem()
    
    try:
        # Initialize
        print("🚀 Initializing complete system...")
        await system.initialize()
        
        # Run for 2 minutes
        print("⏳ Running system test for 2 minutes...")
        
        # Start the system (but limit to 2 minutes)
        start_time = time.time()
        system.running = True
        system.start_time = start_time
        
        # Run trading loop for test duration
        while time.time() - start_time < 120:  # 2 minutes
            # Get and process signals
            raw_signals = await system.scanner.get_signals(max_signals=3)
            
            if raw_signals:
                print(f"📊 Processing {len(raw_signals)} signals...")
                
                for raw_signal in raw_signals[:2]:  # Test with first 2
                    # Test enhanced analysis
                    token_key = f"{raw_signal.chain}_{raw_signal.address}"
                    token_data = system.scanner.token_data.get(token_key, {})
                    
                    if 'prices' in token_data and len(token_data['prices']) >= 5:
                        enhanced_token_data = {
                            'address': raw_signal.address,
                            'chain': raw_signal.chain,
                            'dex': raw_signal.dex,
                            'liquidity_usd': raw_signal.liquidity_usd
                        }
                        
                        enhanced_signal = await system.analyzer.analyze_enhanced_momentum(
                            enhanced_token_data, 
                            list(token_data['prices']), 
                            list(token_data['volumes'])
                        )
                        
                        if enhanced_signal:
                            print(f"  🎯 Enhanced signal: {enhanced_signal.address[:8]}... "
                                  f"Quality: {enhanced_signal.signal_quality} "
                                  f"Score: {enhanced_signal.momentum_score:.3f}")
                            
                            # Test trade execution
                            trade_result = await system.executor.execute_trade_strategy(enhanced_signal)
                            
                            if trade_result:
                                print(f"    💼 Trade: ROI {trade_result.roi_percent:.2f}% "
                                      f"Profit ${trade_result.profit_usd:.2f}")
            
            await asyncio.sleep(5)
        
        # Final report
        stats = system.executor.get_performance_stats()
        print("\n📊 TEST RESULTS:")
        print(f"Signals generated: {system.signals_generated}")
        print(f"Trades executed: {system.trades_executed}")
        print(f"Portfolio value: ${stats['portfolio_value']:.2f}")
        print(f"Total ROI: {stats['roi_percent']:.2f}%")
        print(f"Win rate: {stats['win_rate']:.1f}%")
        
        if system.trades_executed > 0:
            print("✅ Complete system is working!")
        else:
            print("⚠️ System working but no trades executed (normal for short test)")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(test_complete_system())
