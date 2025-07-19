#!/usr/bin/env python3
"""
Integration script to replace old scanner with WebSocket scanner
"""

import asyncio
import sys
import os
from websocket_scanner_v5 import initialize_scanner, scanner
from websocket_config import config_manager

class ScannerIntegration:
    def __init__(self):
        self.scanner = None
        self.running = False
        
    async def initialize(self):
        """Initialize WebSocket scanner"""
        print("🚀 Initializing Renaissance WebSocket Scanner...")
        self.scanner = await initialize_scanner()
        self.running = True
        print("✅ WebSocket scanner initialized successfully")
        
    async def scan_for_momentum(self):
        """Main scanning interface - compatible with existing pipeline"""
        if not self.scanner:
            await self.initialize()
            
        momentum_signals = []
        
        # Get up to 10 momentum signals
        for _ in range(10):
            signal = await self.scanner.get_momentum_signal()
            if signal:
                # Convert to format expected by existing pipeline
                token_data = {
                    'token': signal.token_address,
                    'dex': signal.dex,
                    'chain': signal.chain,
                    'price_now': signal.price_current,
                    'momentum_score': signal.momentum_score,
                    'breakout_strength': signal.breakout_strength,
                    'confidence': signal.confidence,
                    'volume_spike': signal.volume_spike_factor,
                    'liquidity_usd': signal.liquidity_usd,
                    'honeypot_risk': signal.honeypot_risk,
                    'rug_risk': signal.rug_risk,
                    'detected_at': signal.detected_at
                }
                momentum_signals.append(token_data)
            else:
                break
                
        return momentum_signals
        
    async def shutdown(self):
        """Shutdown scanner"""
        if self.scanner:
            await self.scanner.shutdown()
        self.running = False

# Global integration instance
scanner_integration = ScannerIntegration()

# Compatibility functions for existing pipeline
async def scan_tokens():
    """Compatibility function for existing scanner interface"""
    return await scanner_integration.scan_for_momentum()

def initialize_websocket_scanner():
    """Initialize WebSocket scanner for use in pipeline"""
    return scanner_integration

if __name__ == "__main__":
    async def main():
        integration = ScannerIntegration()
        await integration.initialize()
        
        print("🔍 Starting momentum detection...")
        
        try:
            while True:
                signals = await integration.scan_for_momentum()
                
                if signals:
                    print(f"📊 Found {len(signals)} momentum signals:")
                    for signal in signals:
                        print(f"  🎯 {signal['token'][:8]}... "
                              f"Score: {signal['momentum_score']:.3f} "
                              f"Confidence: {signal['confidence']:.3f}")
                else:
                    print("⏳ No signals detected, continuing scan...")
                    
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            await integration.shutdown()
            
    asyncio.run(main())
