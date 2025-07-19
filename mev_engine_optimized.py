import asyncio
import time
import json
import os
from web3 import Web3
from eth_abi import encode_abi, decode_abi
from hexbytes import HexBytes
import aiohttp
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque

@dataclass
class Opportunity:
    token_in: str
    token_out: str
    amount_in: int
    amount_out: int
    profit_wei: int
    gas_cost: int
    net_profit: int
    dex_path: List[str]
    execution_data: bytes
    priority_fee: int

class OptimizedMEVEngine:
    def __init__(self):
        self.arb_rpc = "https://arb1.arbitrum.io/rpc"
        self.poly_rpc = "https://polygon-rpc.com"
        
        self.arb_w3 = Web3(Web3.HTTPProvider(self.arb_rpc))
        self.poly_w3 = Web3(Web3.HTTPProvider(self.poly_rpc))
        
        self.dex_routers = {
            'uniswap_v3_arb': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'sushiswap_arb': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'camelot_arb': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'uniswap_v3_poly': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'quickswap_poly': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'sushiswap_poly': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")'
        }
        
        self.tokens = {
            'WETH_ARB': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'USDC_ARB': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'USDT_ARB': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'WBTC_ARB': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'WETH_POLY': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'USDC_POLY': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'USDT_POLY': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")',
            'WBTC_POLY': 'os.getenv("WALLET_ADDRESS", "os.getenv("CONTRACT_ADDRESS", "")")'
        }
        
        self.opportunities = asyncio.Queue()
        self.price_cache = {}
        self.gas_tracker = deque(maxlen=100)
        self.profit_threshold = int(0.005 * 1e18)
        
        self.router_abi = [
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactTokensForTokens",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
        
    async def start_mev_engine(self):
        print("🔥 Starting optimized MEV engine...")
        self.running = True
        
        tasks = [
            self.scan_arbitrage_opportunities(),
            self.monitor_mempool_arbitrum(),
            self.monitor_mempool_polygon(),
            self.execute_opportunities(),
            self.track_gas_prices()
        ]
        
        await asyncio.gather(*tasks)
    
    async def scan_arbitrage_opportunities(self):
        while self.running:
            try:
                await self.scan_chain_arbitrage('arbitrum')
                await self.scan_chain_arbitrage('polygon')
                await self.scan_cross_chain_arbitrage()
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Arbitrage scan error: {e}")
                await asyncio.sleep(1)
    
    async def scan_chain_arbitrage(self, chain):
        w3 = self.arb_w3 if chain == 'arbitrum' else self.poly_w3
        
        token_pairs = [
            ('WETH', 'USDC'),
            ('WETH', 'USDT'),
            ('USDC', 'USDT'),
            ('WETH', 'WBTC')
        ]
        
        for token_a, token_b in token_pairs:
            token_a_addr = self.tokens[f'{token_a}_{chain.upper()[:3]}']
            token_b_addr = self.tokens[f'{token_b}_{chain.upper()[:3]}']
            
            prices = await self.get_all_dex_prices(w3, token_a_addr, token_b_addr, chain)
            
            if len(prices) >= 2:
                opportunity = self.find_arbitrage_opportunity(prices, token_a_addr, token_b_addr, w3)
                if opportunity and opportunity.net_profit > self.profit_threshold:
                    await self.opportunities.put(opportunity)
    
    async def get_all_dex_prices(self, w3, token_a, token_b, chain):
        prices = []
        amount_in = int(1e18)
        
        relevant_dexes = {k: v for k, v in self.dex_routers.items() if chain[:3].lower() in k}
        
        for dex_name, router_addr in relevant_dexes.items():
            try:
                router = w3.eth.contract(address=router_addr, abi=self.router_abi)
                amounts_out = router.functions.getAmountsOut(amount_in, [token_a, token_b]).call()
                
                prices.append({
                    'dex': dex_name,
                    'router': router_addr,
                    'amount_out': amounts_out[-1],
                    'price': amounts_out[-1] / amount_in
                })
            except:
                continue
        
        return prices
    
    def find_arbitrage_opportunity(self, prices, token_a, token_b, w3):
        if len(prices) < 2:
            return None
        
        buy_dex = min(prices, key=lambda x: x['price'])
        sell_dex = max(prices, key=lambda x: x['price'])
        
        price_diff = sell_dex['price'] - buy_dex['price']
        price_diff_pct = price_diff / buy_dex['price']
        
        if price_diff_pct < 0.005:
            return None
        
        amount_in = int(1e18)
        profit_wei = int(amount_in * price_diff_pct)
        gas_cost = int(w3.eth.gas_price * 300000)
        net_profit = profit_wei - gas_cost
        
        if net_profit > self.profit_threshold:
            execution_data = encode_abi(
                ['address', 'address', 'address', 'address', 'uint256'],
                [buy_dex['router'], sell_dex['router'], token_a, token_b, amount_in]
            )
            
            return Opportunity(
                token_in=token_a,
                token_out=token_b,
                amount_in=amount_in,
                amount_out=sell_dex['amount_out'],
                profit_wei=profit_wei,
                gas_cost=gas_cost,
                net_profit=net_profit,
                dex_path=[buy_dex['dex'], sell_dex['dex']],
                execution_data=execution_data,
                priority_fee=int(gas_cost * 0.5)
            )
        
        return None
    
    async def scan_cross_chain_arbitrage(self):
        arb_tokens = ['WETH_ARB', 'USDC_ARB', 'USDT_ARB']
        poly_tokens = ['WETH_POLY', 'USDC_POLY', 'USDT_POLY']
        
        for i, arb_token in enumerate(arb_tokens):
            poly_token = poly_tokens[i]
            
            arb_price = await self.get_token_usd_price(self.tokens[arb_token], 'arbitrum')
            poly_price = await self.get_token_usd_price(self.tokens[poly_token], 'polygon')
            
            if arb_price and poly_price:
                price_diff_pct = abs(arb_price - poly_price) / min(arb_price, poly_price)
                
                if price_diff_pct > 0.01:
                    print(f"Cross-chain opportunity: {arb_token} - {price_diff_pct*100:.2f}% difference")
    
    async def get_token_usd_price(self, token_addr, chain):
        try:
            w3 = self.arb_w3 if chain == 'arbitrum' else self.poly_w3
            
            if chain == 'arbitrum':
                router_addr = self.dex_routers['uniswap_v3_arb']
                usdc_addr = self.tokens['USDC_ARB']
            else:
                router_addr = self.dex_routers['uniswap_v3_poly']
                usdc_addr = self.tokens['USDC_POLY']
            
            router = w3.eth.contract(address=router_addr, abi=self.router_abi)
            amounts_out = router.functions.getAmountsOut(int(1e18), [token_addr, usdc_addr]).call()
            
            return amounts_out[-1] / 1e6
        except:
            return None
    
    async def monitor_mempool_arbitrum(self):
        while self.running:
            try:
                pending_block = self.arb_w3.eth.get_block('pending', full_transactions=True)
                
                for tx in pending_block.transactions[:50]:
                    if tx.to in self.dex_routers.values() and tx.input:
                        await self.analyze_pending_tx(tx, 'arbitrum')
                
                await asyncio.sleep(0.5)
            except:
                await asyncio.sleep(2)
    
    async def monitor_mempool_polygon(self):
        while self.running:
            try:
                pending_block = self.poly_w3.eth.get_block('pending', full_transactions=True)
                
                for tx in pending_block.transactions[:50]:
                    if tx.to in self.dex_routers.values() and tx.input:
                        await self.analyze_pending_tx(tx, 'polygon')
                
                await asyncio.sleep(0.5)
            except:
                await asyncio.sleep(2)
    
    async def analyze_pending_tx(self, tx, chain):
        try:
            if len(tx.input) < 10:
                return
            
            method_id = tx.input[:10]
            
            if method_id == '0x38ed1739':
                decoded = decode_abi(
                    ['uint256', 'uint256', 'address[]', 'address', 'uint256'],
                    HexBytes(tx.input[10:])
                )
                
                amount_in = decoded[0]
                path = decoded[2]
                
                if len(path) == 2 and amount_in > int(0.1 * 1e18):
                    frontrun_opp = await self.create_frontrun_opportunity(
                        path[0], path[1], amount_in, tx.gasPrice, chain
                    )
                    
                    if frontrun_opp:
                        await self.opportunities.put(frontrun_opp)
        except:
            pass
    
    async def create_frontrun_opportunity(self, token_in, token_out, amount_in, victim_gas_price, chain):
        try:
            w3 = self.arb_w3 if chain == 'arbitrum' else self.poly_w3
            
            competing_routers = [addr for name, addr in self.dex_routers.items() 
                               if chain[:3].lower() in name]
            
            best_price = 0
            best_router = None
            
            for router_addr in competing_routers:
                try:
                    router = w3.eth.contract(address=router_addr, abi=self.router_abi)
                    amounts_out = router.functions.getAmountsOut(
                        amount_in, [token_in, token_out]
                    ).call()
                    
                    if amounts_out[-1] > best_price:
                        best_price = amounts_out[-1]
                        best_router = router_addr
                except:
                    continue
            
            if best_router and best_price > amount_in * 1.01:
                profit_wei = best_price - amount_in
                gas_cost = int((victim_gas_price + int(5e9)) * 200000)
                net_profit = profit_wei - gas_cost
                
                if net_profit > self.profit_threshold:
                    execution_data = encode_abi(
                        ['address', 'address', 'address', 'uint256'],
                        [best_router, token_in, token_out, amount_in]
                    )
                    
                    return Opportunity(
                        token_in=token_in,
                        token_out=token_out,
                        amount_in=amount_in,
                        amount_out=best_price,
                        profit_wei=profit_wei,
                        gas_cost=gas_cost,
                        net_profit=net_profit,
                        dex_path=[best_router],
                        execution_data=execution_data,
                        priority_fee=victim_gas_price + int(10e9)
                    )
        except:
            pass
        
        return None
    
    async def execute_opportunities(self):
        while self.running:
            try:
                opportunity = await self.opportunities.get()
                
                if opportunity.net_profit > self.profit_threshold:
                    success = await self.execute_trade(opportunity)
                    
                    if success:
                        profit_eth = opportunity.net_profit / 1e18
                        print(f"✅ MEV executed: {profit_eth:.6f} ETH profit")
                        print(f"   Tokens: {opportunity.token_in[:6]}.../{opportunity.token_out[:6]}...")
                        print(f"   DEX path: {opportunity.dex_path}")
                
            except:
                await asyncio.sleep(0.1)
    
    async def execute_trade(self, opportunity):
        try:
            w3 = self.arb_w3
            
            if not os.getenv('PRIVATE_KEY'):
                print(f"SIMULATION: Would execute trade for {opportunity.net_profit/1e18:.6f} ETH profit")
                return True
            
            if not os.getenv("PRIVATE_KEY"): return True; account = w3.eth.account.from_key(os.getenv("PRIVATE_KEY"))
            
            router = w3.eth.contract(address=opportunity.dex_path[0], abi=self.router_abi)
            
            nonce = w3.eth.get_transaction_count(account.address)
            
            tx = router.functions.swapExactTokensForTokens(
                opportunity.amount_in,
                int(opportunity.amount_out * 0.995),
                [opportunity.token_in, opportunity.token_out],
                account.address,
                int(time.time()) + 300
            ).build_transaction({
                'from': account.address,
                'gas': 300000,
                'gasPrice': opportunity.priority_fee,
                'nonce': nonce
            })
            
            signed_tx = w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            return receipt.status == 1
            
        except Exception as e:
            print(f"Execution failed: {e}")
            return False
    
    async def track_gas_prices(self):
        while self.running:
            try:
                arb_gas = self.arb_w3.eth.gas_price
                poly_gas = self.poly_w3.eth.gas_price
                
                self.gas_tracker.append({
                    'timestamp': time.time(),
                    'arb_gas': arb_gas,
                    'poly_gas': poly_gas
                })
                
                await asyncio.sleep(10)
            except:
                await asyncio.sleep(30)

mev_engine = OptimizedMEVEngine()
