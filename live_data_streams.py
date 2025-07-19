
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import websockets
import json
import aiohttp
import time
from typing import Dict, List, Optional
from collections import deque, defaultdict
import requests
from gql import gql, Client
from gql.transport.websockets import WebsocketsTransport

class LiveDataStreams:
    def __init__(self):
        self.price_feeds = defaultdict(lambda: deque(maxlen=1000))
        self.volume_feeds = defaultdict(lambda: deque(maxlen=1000))
        self.mempool_data = deque(maxlen=10000)
        
        self.uniswap_v2_subgraph = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"
        self.uniswap_v3_subgraph = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
        
        self.websocket_endpoints = {
            'ethereum': f"wss://eth-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}",
            'arbitrum': f"wss://arb-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}",
            'polygon': f"wss://polygon-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}"
        }
        
        self.session = None
        self.connections = {}
        self.running = False

    async def initialize(self):
        self.session = aiohttp.ClientSession()
        self.running = True
        
        tasks = [
            asyncio.create_task(self.subscribe_uniswap_pairs()),
            asyncio.create_task(self.monitor_mempool('ethereum')),
            asyncio.create_task(self.monitor_mempool('arbitrum')),
            asyncio.create_task(self.price_stream_processor()),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def subscribe_uniswap_pairs(self):
        transport = WebsocketsTransport(url="wss://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2")
        client = Client(transport=transport, fetch_schema_from_transport=True)
        
        subscription = gql("""
            subscription {
                swaps(first: 50, orderBy: timestamp, orderDirection: desc) {
                    pair {
                        id
                        token0 { symbol }
                        token1 { symbol }
                    }
                    amount0In
                    amount1In
                    amount0Out
                    amount1Out
                    amountUSD
                    timestamp
                }
            }
        """)
        
        while self.running:
            try:
                async for result in client.subscribe(subscription):
                    await self.process_swap_data(result['swaps'])
            except Exception as e:
                await asyncio.sleep(5)

    async def process_swap_data(self, swaps: List[Dict]):
        for swap in swaps:
            pair_id = swap['pair']['id']
            amount_usd = float(swap['amountUSD'])
            timestamp = int(swap['timestamp'])
            
            self.volume_feeds[pair_id].append({
                'amount_usd': amount_usd,
                'timestamp': timestamp
            })
            
            if amount_usd > 1000:
                token0_amount = float(swap['amount0In']) + float(swap['amount0Out'])
                token1_amount = float(swap['amount1In']) + float(swap['amount1Out'])
                
                if token0_amount > 0:
                    price = amount_usd / token0_amount
                else:
                    price = amount_usd / token1_amount if token1_amount > 0 else 0
                
                self.price_feeds[pair_id].append({
                    'price': price,
                    'timestamp': timestamp,
                    'volume': amount_usd
                })

    async def monitor_mempool(self, network: str):
        uri = self.websocket_endpoints.get(network)
        if not uri:
            return
            
        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.connections[network] = websocket
                    
                    subscribe_msg = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "eth_subscribe",
                        "params": ["newPendingTransactions"]
                    }
                    
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    async for message in websocket:
                        data = json.loads(message)
                        if 'params' in data:
                            tx_hash = data['params']['result']
                            await self.fetch_transaction_details(tx_hash, network)
                            
            except Exception as e:
                await asyncio.sleep(5)

    async def fetch_transaction_details(self, tx_hash: str, network: str):
        rpc_call = {
            "jsonrpc": "2.0",
            "method": "eth_getTransactionByHash",
            "params": [tx_hash],
            "id": 1
        }
        
        try:
            async with self.session.post(
                self.websocket_endpoints[network].replace('wss://', 'https://'),
                json=rpc_call
            ) as response:
                result = await response.json()
                
                if result.get('result'):
                    tx = result['result']
                    self.mempool_data.append({
                        'hash': tx_hash,
                        'to': tx.get('to'),
                        'value': int(tx.get('value', '0'), 16),
                        'gasPrice': int(tx.get('gasPrice', '0'), 16),
                        'timestamp': time.time(),
                        'network': network
                    })
        except:
            pass

    async def price_stream_processor(self):
        while self.running:
            current_time = time.time()
            
            for pair_id, price_data in self.price_feeds.items():
                if len(price_data) >= 10:
                    recent_prices = [p['price'] for p in list(price_data)[-10:]]
                    recent_volumes = [p['volume'] for p in list(price_data)[-10:]]
                    
                    price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    volume_surge = sum(recent_volumes[-3:]) / (sum(recent_volumes[:-3]) + 1)
                    
                    if price_change > 0.09 and volume_surge > 2.0:
                        await self.signal_momentum_detected(pair_id, {
                            'price_change': price_change,
                            'volume_surge': volume_surge,
                            'current_price': recent_prices[-1],
                            'timestamp': current_time
                        })
            
            await asyncio.sleep(1)

    async def signal_momentum_detected(self, pair_id: str, signal_data: Dict):
        print(f"MOMENTUM DETECTED: {pair_id} - Change: {signal_data['price_change']:.3f}")

    def get_pair_data(self, pair_id: str) -> Dict:
        return {
            'prices': list(self.price_feeds[pair_id]),
            'volumes': list(self.volume_feeds[pair_id])
        }

    async def shutdown(self):
        self.running = False
        if self.session:
            await self.session.close()
        for connection in self.connections.values():
            await connection.close()

live_streams = LiveDataStreams()
