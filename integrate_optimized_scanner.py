#!/usr/bin/env python3
"""
Integration wrapper for the optimized scanner
"""

import asyncio
from websocket_scanner_optimized import OptimizedWebSocketScanner

class ScannerIntegration:
    def __init__(self):
        self.scanner = None
        self.running = False
        
    async def initialize(self):
        """Initialize scanner"""
        if not self.scanner:
            self.scanner = OptimizedWebSocketScanner()
            await self.scanner.initialize()
            self.running = True
            
    async def scan_for_momentum(self):
        """Get momentum signals - compatible with existing pipeline"""
        if not self.running:
            await self.initialize()
            
        signals = await self.scanner.get_signals(max_signals=10)
        
        # Convert to format expected by existing pipeline
        momentum_tokens = []
        for signal in signals:
            token_data = {
                'token': signal.address,
                'chain': signal.chain,
                'dex': signal.dex,
                'price_now': signal.price,
                'momentum_score': signal.momentum_score,
                'confidence': signal.confidence,
                'volume_24h': signal.volume_24h,
                'liquidity_usd': signal.liquidity_usd,
                'detected_at': signal.detected_at
            }
            momentum_tokens.append(token_data)
            
        return momentum_tokens
        
    async def shutdown(self):
        """Shutdown scanner"""
        if self.scanner:
            await self.scanner.shutdown()
        self.running = False

# Global integration instance
scanner_integration = ScannerIntegration()

# Compatibility functions
async def scan_tokens():
    """Compatibility function for existing pipeline"""
    return await scanner_integration.scan_for_momentum()

if __name__ == "__main__":
    async def main():
        integration = ScannerIntegration()
        
        try:
            print("🚀 Starting optimized scanner integration...")
            
            while True:
                signals = await integration.scan_for_momentum()
                
                if signals:
                    print(f"📊 Pipeline received {len(signals)} signals:")
                    for signal in signals:
                        print(f"  🎯 {signal['token'][:8]}... "
                              f"Score: {signal['momentum_score']:.3f}")
                else:
                    print("⏳ No signals, continuing scan...")
                    
                await asyncio.sleep(3)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            await integration.shutdown()
            
    asyncio.run(main())
