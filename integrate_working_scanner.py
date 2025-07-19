#!/usr/bin/env python3
"""
Integration wrapper for working scanner
"""

import asyncio
from websocket_scanner_working import WorkingWebSocketScanner

class WorkingScannerIntegration:
    def __init__(self):
        self.scanner = None
        self.running = False
        
    async def initialize(self):
        """Initialize scanner"""
        if not self.scanner:
            self.scanner = WorkingWebSocketScanner()
            await self.scanner.initialize()
            self.running = True
            print("✅ Working scanner initialized and discovering tokens...")
            
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
                'detected_at': signal.detected_at,
                'price_change_1h': signal.price_change_1h
            }
            momentum_tokens.append(token_data)
            
        return momentum_tokens
        
    async def get_discovery_stats(self):
        """Get token discovery statistics"""
        if not self.scanner:
            return {'discovered_tokens': 0, 'processed_tokens': 0, 'api_calls': 0}
            
        return {
            'discovered_tokens': len(self.scanner.discovered_tokens),
            'processed_tokens': self.scanner.tokens_processed,
            'api_calls': self.scanner.api_calls_made,
            'active_tokens': len(self.scanner.token_data)
        }
        
    async def shutdown(self):
        """Shutdown scanner"""
        if self.scanner:
            await self.scanner.shutdown()
        self.running = False

# Global integration instance
working_scanner_integration = WorkingScannerIntegration()

# Compatibility functions
async def scan_tokens():
    """Compatibility function for existing pipeline"""
    return await working_scanner_integration.scan_for_momentum()

if __name__ == "__main__":
    async def main():
        integration = WorkingScannerIntegration()
        
        try:
            print("🚀 Starting working scanner integration...")
            
            while True:
                # Get discovery stats
                stats = await integration.get_discovery_stats()
                print(f"📊 Discovery stats: {stats['discovered_tokens']} tokens, "
                      f"{stats['api_calls']} API calls")
                
                # Get momentum signals
                signals = await integration.scan_for_momentum()
                
                if signals:
                    print(f"📊 Pipeline received {len(signals)} signals:")
                    for signal in signals[:3]:
                        print(f"  🎯 {signal['token'][:8]}... "
                              f"Chain: {signal['chain']} "
                              f"Score: {signal['momentum_score']:.3f} "
                              f"Liquidity: ${signal['liquidity_usd']:,.0f}")
                else:
                    print("⏳ No signals yet, continuing discovery...")
                    
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            await integration.shutdown()
            
    asyncio.run(main())
