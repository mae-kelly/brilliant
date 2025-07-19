import asyncio
import aiohttp
import time
import numpy as np
from typing import List, Dict, Set
from dataclasses import dataclass
from collections import deque
import json

@dataclass
class GraphQLToken:
    address: str
    chain: str
    price: float
    volume_24h: float
    liquidity: float
    tx_count: int
    price_change: float
    created_at: int

class GraphQLScanner:
    def __init__(self):
        self.subgraphs = {
            'uniswap_v3': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
            'uniswap_v2': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2',
            'sushiswap_arbitrum': 'https://api.thegraph.com/subgraphs/name/sushi-v2/sushiswap-arbitrum',
            'sushiswap_polygon': 'https://api.thegraph.com/subgraphs/name/sushi-v2/sushiswap-polygon',
            'camelot': 'https://api.thegraph.com/subgraphs/name/camelot-labs/camelot-amm',
            'quickswap': 'https://api.thegraph.com/subgraphs/name/quickswap/quickswap'
        }
        
        self.batch_size = 1000
        self.tokens_per_hour = 0
        self.session_pool = []
        self.discovered_tokens = set()
        
    async def initialize(self):
        connector = aiohttp.TCPConnector(limit=100)
        for _ in range(20):
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=3),
                headers={'Content-Type': 'application/json'}
            )
            self.session_pool.append(session)
    
    async def scan_all_subgraphs(self):
        tasks = []
        for name, url in self.subgraphs.items():
            for i in range(5):
                task = asyncio.create_task(self.scan_subgraph(name, url, i))
                tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def scan_subgraph(self, name: str, url: str, worker_id: int):
        session = self.session_pool[worker_id % len(self.session_pool)]
        skip = 0
        
        while True:
            try:
                query = self.build_batch_query(skip, self.batch_size)
                
                async with session.post(url, json={'query': query}) as response:
                    if response.status == 200:
                        data = await response.json()
                        tokens = await self.process_graphql_response(data, name)
                        
                        if len(tokens) < self.batch_size:
                            break
                        
                        skip += self.batch_size
                        self.tokens_per_hour += len(tokens)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                await asyncio.sleep(1)
                continue
    
    def build_batch_query(self, skip: int, first: int) -> str:
        return f'''
        {{
          tokens(
            first: {first}
            skip: {skip}
            orderBy: volumeUSD
            orderDirection: desc
            where: {{
              volumeUSD_gt: "1000"
              liquidity_gt: "10000"
            }}
          ) {{
            id
            symbol
            name
            decimals
            volumeUSD
            liquidity
            txCount
            derivedETH
            tokenDayData(
              first: 1
              orderBy: date
              orderDirection: desc
            ) {{
              priceUSD
              volumeUSD
              open
              high
              low
              close
            }}
          }}
        }}
        '''
    
    async def process_graphql_response(self, data: dict, subgraph: str) -> List[GraphQLToken]:
        tokens = []
        
        try:
            for token_data in data.get('data', {}).get('tokens', []):
                token_address = token_data['id']
                
                if token_address in self.discovered_tokens:
                    continue
                
                day_data = token_data.get('tokenDayData', [])
                if not day_data:
                    continue
                
                latest_data = day_data[0]
                price = float(latest_data.get('priceUSD', 0))
                volume = float(token_data.get('volumeUSD', 0))
                liquidity = float(token_data.get('liquidity', 0))
                
                if price > 0 and volume > 1000 and liquidity > 10000:
                    price_change = self.calculate_price_change(latest_data)
                    
                    if abs(price_change) > 5:
                        token = GraphQLToken(
                            address=token_address,
                            chain=self.get_chain_from_subgraph(subgraph),
                            price=price,
                            volume_24h=volume,
                            liquidity=liquidity,
                            tx_count=int(token_data.get('txCount', 0)),
                            price_change=price_change,
                            created_at=int(time.time())
                        )
                        
                        tokens.append(token)
                        self.discovered_tokens.add(token_address)
        
        except Exception as e:
            pass
        
        return tokens
    
    def calculate_price_change(self, day_data: dict) -> float:
        try:
            open_price = float(day_data.get('open', 0))
            close_price = float(day_data.get('close', 0))
            
            if open_price > 0:
                return ((close_price - open_price) / open_price) * 100
        except:
            pass
        
        return 0.0
    
    def get_chain_from_subgraph(self, subgraph: str) -> str:
        if 'arbitrum' in subgraph:
            return 'arbitrum'
        elif 'polygon' in subgraph:
            return 'polygon'
        elif 'camelot' in subgraph:
            return 'arbitrum'
        elif 'quickswap' in subgraph:
            return 'polygon'
        else:
            return 'ethereum'
    
    async def get_tokens_per_hour(self) -> int:
        return self.tokens_per_hour

graphql_scanner = GraphQLScanner()
