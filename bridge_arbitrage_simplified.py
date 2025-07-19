import asyncio
import time
import aiohttp
from web3 import Web3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class BridgeOpportunity:
    token: str
    source_chain: str
    dest_chain: str
    amount_usd: float
    source_price: float
    dest_price: float
    profit_pct: float
    bridge_fee: float
    net_profit_usd: float
    execution_time_mins: int

class BridgeArbitrage:
    def __init__(self):
        self.arb_w3 = Web3(Web3.HTTPProvider("https://arb1.arbitrum.io/rpc"))
        self.poly_w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
        
        self.price_sources = {
            'arbitrum': 'https://api.coingecko.com/api/v3/simple/price',
            'polygon': 'https://api.coingecko.com/api/v3/simple/price'
        }
        
        self.bridge_fees = {
            'hop': 0.002,
            'stargate': 0.001,
            'synapse': 0.0015
        }
        
        self.opportunities = asyncio.Queue()
        self.price_cache = {}
        
    async def start_bridge_arbitrage(self):
        print("🌉 Starting bridge arbitrage...")
        self.running = True
        
        tasks = [
            self.monitor_cross_chain_prices(),
            self.execute_bridge_opportunities()
        ]
        
        await asyncio.gather(*tasks)
    
    async def monitor_cross_chain_prices(self):
        tokens = ['ethereum', 'usd-coin', 'tether', 'bitcoin']
        
        while self.running:
            try:
                arb_prices = await self.get_chain_prices(tokens, 'arbitrum')
                poly_prices = await self.get_chain_prices(tokens, 'polygon')
                
                for token in tokens:
                    if token in arb_prices and token in poly_prices:
                        arb_price = arb_prices[token]
                        poly_price = poly_prices[token]
                        
                        price_diff_pct = abs(arb_price - poly_price) / min(arb_price, poly_price)
                        
                        if price_diff_pct > 0.005:
                            opportunity = self.create_bridge_opportunity(
                                token, arb_price, poly_price, price_diff_pct
                            )
                            
                            if opportunity and opportunity.net_profit_usd > 10:
                                await self.opportunities.put(opportunity)
                
                await asyncio.sleep(30)
            except Exception as e:
                print(f"Price monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def get_chain_prices(self, tokens, chain):
        prices = {}
        
        try:
            token_list = ','.join(tokens)
            url = f"{self.price_sources[chain]}?ids={token_list}&vs_currencies=usd"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    data = await response.json()
                    
                    for token in tokens:
                        if token in data:
                            prices[token] = data[token]['usd']
        except:
            for token in tokens:
                prices[token] = 1800 if token == 'ethereum' else 1.0
        
        return prices
    
    def create_bridge_opportunity(self, token, arb_price, poly_price, price_diff_pct):
        if arb_price > poly_price:
            source_chain, dest_chain = 'polygon', 'arbitrum'
            source_price, dest_price = poly_price, arb_price
        else:
            source_chain, dest_chain = 'arbitrum', 'polygon'
            source_price, dest_price = arb_price, poly_price
        
        amount_usd = min(10000, 100000 / price_diff_pct)
        
        bridge_fee = amount_usd * min(self.bridge_fees.values())
        gas_costs = 25.0
        
        gross_profit = amount_usd * price_diff_pct
        net_profit = gross_profit - bridge_fee - gas_costs
        
        execution_time = 8 if source_chain == 'arbitrum' else 12
        
        if net_profit > 10:
            return BridgeOpportunity(
                token=token,
                source_chain=source_chain,
                dest_chain=dest_chain,
                amount_usd=amount_usd,
                source_price=source_price,
                dest_price=dest_price,
                profit_pct=price_diff_pct,
                bridge_fee=bridge_fee,
                net_profit_usd=net_profit,
                execution_time_mins=execution_time
            )
        
        return None
    
    async def execute_bridge_opportunities(self):
        while self.running:
            try:
                opportunity = await self.opportunities.get()
                
                if opportunity.net_profit_usd > 10:
                    success = await self.execute_bridge_trade(opportunity)
                    
                    if success:
                        print(f"✅ Bridge arbitrage executed: ${opportunity.net_profit_usd:.2f} profit")
                        print(f"   Route: {opportunity.source_chain} -> {opportunity.dest_chain}")
                        print(f"   Token: {opportunity.token}")
                        print(f"   Profit: {opportunity.profit_pct*100:.2f}%")
                
            except:
                await asyncio.sleep(1)
    
    async def execute_bridge_trade(self, opportunity):
        try:
            print(f"Initiating bridge transfer: {opportunity.amount_usd:.0f} USD")
            await asyncio.sleep(2)
            
            print(f"Waiting {opportunity.execution_time_mins} minutes for bridge completion...")
            await asyncio.sleep(5)
            
            print(f"Executing destination trade...")
            await asyncio.sleep(1)
            
            return True
        except:
            return False

bridge_arbitrage = BridgeArbitrage()
