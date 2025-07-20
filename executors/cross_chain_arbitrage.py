
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

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class ArbitrageOpportunity:
    token_address: str
    source_chain: str
    target_chain: str
    source_price: float
    target_price: float
    profit_potential: float
    bridge_cost: float
    net_profit: float
    execution_time: int
    confidence_score: float

class CrossChainArbitrage:
    def __init__(self):
        self.bridge_costs = {
            ('ethereum', 'arbitrum'): 0.002,
            ('ethereum', 'optimism'): 0.003,
            ('ethereum', 'polygon'): 0.001,
            ('arbitrum', 'optimism'): 0.001,
            ('arbitrum', 'polygon'): 0.0015,
            ('optimism', 'polygon'): 0.0015
        }
        
        self.bridge_times = {
            ('ethereum', 'arbitrum'): 15,
            ('ethereum', 'optimism'): 20,
            ('ethereum', 'polygon'): 30,
            ('arbitrum', 'optimism'): 10,
            ('arbitrum', 'polygon'): 25,
            ('optimism', 'polygon'): 25
        }
        
        self.chain_scanners = {}
        self.price_feeds = {}
        self.opportunities = asyncio.Queue(maxsize=1000)

    async def initialize(self, scanners: Dict):
        self.chain_scanners = scanners
        
        for chain in ['ethereum', 'arbitrum', 'optimism', 'polygon']:
            task = asyncio.create_task(self.monitor_chain_prices(chain))
            
        task = asyncio.create_task(self.detect_arbitrage_opportunities())

    async def monitor_chain_prices(self, chain: str):
        while True:
            try:
                if chain in self.chain_scanners:
                    tokens = await self.chain_scanners[chain].get_recent_tokens(100)
                    
                    for token in tokens:
                        key = f"{chain}_{token.address}"
                        self.price_feeds[key] = {
                            'price': token.price,
                            'volume': token.volume_24h,
                            'liquidity': token.liquidity_usd,
                            'timestamp': time.time()
                        }
                
                await asyncio.sleep(10)
                
            except Exception as e:
                await asyncio.sleep(30)

    async def detect_arbitrage_opportunities(self):
        while True:
            try:
                await self.scan_cross_chain_prices()
                await asyncio.sleep(5)
                
            except Exception as e:
                await asyncio.sleep(10)

    async def scan_cross_chain_prices(self):
        chains = ['ethereum', 'arbitrum', 'optimism', 'polygon']
        token_addresses = set()
        
        for key in self.price_feeds.keys():
            chain, address = key.split('_', 1)
            token_addresses.add(address)
        
        for token_address in list(token_addresses)[:100]:
            prices_by_chain = {}
            
            for chain in chains:
                key = f"{chain}_{token_address}"
                if key in self.price_feeds:
                    data = self.price_feeds[key]
                    if time.time() - data['timestamp'] < 120:
                        prices_by_chain[chain] = data
            
            if len(prices_by_chain) >= 2:
                opportunities = self.calculate_arbitrage_opportunities(token_address, prices_by_chain)
                
                for opp in opportunities:
                    if opp.net_profit > 0.05:
                        try:
                            self.opportunities.put_nowait(opp)
                        except:
                            pass

    def calculate_arbitrage_opportunities(self, token_address: str, prices_by_chain: Dict) -> List[ArbitrageOpportunity]:
        opportunities = []
        chains = list(prices_by_chain.keys())
        
        for i, source_chain in enumerate(chains):
            for target_chain in chains[i+1:]:
                source_data = prices_by_chain[source_chain]
                target_data = prices_by_chain[target_chain]
                
                source_price = source_data['price']
                target_price = target_data['price']
                
                price_diff = abs(target_price - source_price)
                price_ratio = price_diff / min(source_price, target_price) if min(source_price, target_price) > 0 else 0
                
                if price_ratio > 0.02:
                    bridge_pair = (source_chain, target_chain) if source_price < target_price else (target_chain, source_chain)
                    bridge_cost = self.bridge_costs.get(bridge_pair, 0.005)
                    bridge_time = self.bridge_times.get(bridge_pair, 30)
                    
                    profit_potential = price_ratio
                    net_profit = profit_potential - bridge_cost
                    
                    confidence_score = self.calculate_confidence_score(source_data, target_data, price_ratio)
                    
                    if net_profit > 0:
                        opportunity = ArbitrageOpportunity(
                            token_address=token_address,
                            source_chain=source_chain if source_price < target_price else target_chain,
                            target_chain=target_chain if source_price < target_price else source_chain,
                            source_price=min(source_price, target_price),
                            target_price=max(source_price, target_price),
                            profit_potential=profit_potential,
                            bridge_cost=bridge_cost,
                            net_profit=net_profit,
                            execution_time=bridge_time,
                            confidence_score=confidence_score
                        )
                        
                        opportunities.append(opportunity)
        
        return opportunities

    def calculate_confidence_score(self, source_data: Dict, target_data: Dict, price_ratio: float) -> float:
        source_liquidity = source_data.get('liquidity', 0)
        target_liquidity = target_data.get('liquidity', 0)
        source_volume = source_data.get('volume', 0)
        target_volume = target_data.get('volume', 0)
        
        liquidity_score = min(min(source_liquidity, target_liquidity) / 50000, 1.0)
        volume_score = min(min(source_volume, target_volume) / 10000, 1.0)
        price_score = min(price_ratio * 10, 1.0)
        
        confidence = (liquidity_score * 0.4 + volume_score * 0.3 + price_score * 0.3)
        
        return confidence

    async def get_opportunities(self, min_profit: float = 0.03) -> List[ArbitrageOpportunity]:
        opportunities = []
        
        for _ in range(20):
            try:
                opp = await asyncio.wait_for(self.opportunities.get(), timeout=0.1)
                if opp.net_profit >= min_profit:
                    opportunities.append(opp)
            except asyncio.TimeoutError:
                break
        
        return sorted(opportunities, key=lambda x: x.net_profit, reverse=True)

cross_chain_arbitrage = CrossChainArbitrage()
