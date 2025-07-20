"""
PRODUCTION GraphQL Scanner - Real subgraph queries
NO SIMULATION - Complete GraphQL implementation
"""
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class RealToken:
    address: str
    chain: str
    symbol: str
    name: str
    price_usd: float
    volume_24h_usd: float
    liquidity_usd: float
    price_change_24h: float
    created_timestamp: int
    tx_count: int

class RealGraphQLScanner:
    """PRODUCTION GraphQL scanner with complete implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # REAL subgraph endpoints
        self.subgraphs = {
            'ethereum_uniswap_v3': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
            'ethereum_uniswap_v2': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2',
            'arbitrum_uniswap_v3': 'https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-minimal',
            'arbitrum_camelot': 'https://api.thegraph.com/subgraphs/name/camelot-labs/camelot-amm',
            'polygon_uniswap_v3': 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon',
            'polygon_quickswap': 'https://api.thegraph.com/subgraphs/name/sameepsi/quickswap06'
        }
        
        self.discovered_tokens = set()
        self.rate_limits = {}
        
    async def initialize(self):
        """Initialize scanner with session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        self.logger.info("✅ Real GraphQL scanner initialized")
    
    async def scan_all_subgraphs(self) -> List[RealToken]:
        """Scan all subgraphs for tokens with momentum"""
        all_tokens = []
        
        tasks = []
        for subgraph_name, url in self.subgraphs.items():
            task = asyncio.create_task(self.scan_subgraph_complete(subgraph_name, url))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_tokens.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Subgraph scan failed: {result}")
        
        return all_tokens
    
    async def scan_subgraph_complete(self, subgraph_name: str, url: str) -> List[RealToken]:
        """Complete subgraph scanning implementation"""
        tokens = []
        skip = 0
        batch_size = 1000
        
        while True:
            try:
                # Rate limiting
                await self._check_rate_limit(subgraph_name)
                
                # Build complete query
                query = self._build_token_query(skip, batch_size)
                
                async with self.session.post(
                    url,
                    json={'query': query},
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    
                    if response.status != 200:
                        self.logger.error(f"GraphQL error {response.status} for {subgraph_name}")
                        break
                    
                    data = await response.json()
                    
                    if 'errors' in data:
                        self.logger.error(f"GraphQL errors: {data['errors']}")
                        break
                    
                    batch_tokens = await self._process_token_data(data, subgraph_name)
                    tokens.extend(batch_tokens)
                    
                    # Check if we got fewer tokens than requested (end of data)
                    if len(batch_tokens) < batch_size:
                        break
                    
                    skip += batch_size
                    
                    # Prevent infinite loops
                    if skip > 100000:
                        break
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error scanning {subgraph_name}: {e}")
                break
        
        self.logger.info(f"✅ Scanned {subgraph_name}: {len(tokens)} tokens")
        return tokens
    
    def _build_token_query(self, skip: int, first: int) -> str:
        """Build complete GraphQL query"""
        return f'''
        {{
          tokens(
            first: {first}
            skip: {skip}
            orderBy: volumeUSD
            orderDirection: desc
            where: {{
              volumeUSD_gt: "1000"
              txCount_gt: "10"
            }}
          ) {{
            id
            symbol
            name
            decimals
            totalSupply
            volumeUSD
            txCount
            derivedETH
            tokenDayData(
              first: 7
              orderBy: date
              orderDirection: desc
            ) {{
              id
              date
              priceUSD
              volumeUSD
              open
              high
              low
              close
              totalValueLocked
              totalValueLockedUSD
            }}
            whitelistPools {{
              id
              token0 {{
                id
                symbol
              }}
              token1 {{
                id
                symbol
              }}
              feeTier
              liquidity
              volumeUSD
              totalValueLockedUSD
            }}
          }}
        }}
        '''
    
    async def _process_token_data(self, data: Dict, subgraph_name: str) -> List[RealToken]:
        """Process GraphQL response into RealToken objects"""
        tokens = []
        
        try:
            token_data_list = data.get('data', {}).get('tokens', [])
            
            for token_data in token_data_list:
                token_address = token_data['id']
                
                # Skip if already discovered
                if token_address in self.discovered_tokens:
                    continue
                
                # Get chain from subgraph name
                chain = self._extract_chain_from_subgraph(subgraph_name)
                
                # Process day data
                day_data = token_data.get('tokenDayData', [])
                if not day_data:
                    continue
                
                latest_day = day_data[0]
                price_usd = float(latest_day.get('priceUSD', 0))
                volume_24h = float(latest_day.get('volumeUSD', 0))
                liquidity_usd = float(latest_day.get('totalValueLockedUSD', 0))
                
                # Calculate 24h price change
                price_change_24h = 0.0
                if len(day_data) > 1:
                    prev_price = float(day_data[1].get('priceUSD', 0))
                    if prev_price > 0:
                        price_change_24h = ((price_usd - prev_price) / prev_price) * 100
                
                # Filter for momentum (5%+ change)
                if abs(price_change_24h) < 5:
                    continue
                
                # Validate data quality
                if price_usd <= 0 or volume_24h < 1000 or liquidity_usd < 10000:
                    continue
                
                token = RealToken(
                    address=token_address,
                    chain=chain,
                    symbol=token_data.get('symbol', ''),
                    name=token_data.get('name', ''),
                    price_usd=price_usd,
                    volume_24h_usd=volume_24h,
                    liquidity_usd=liquidity_usd,
                    price_change_24h=price_change_24h,
                    created_timestamp=int(latest_day.get('date', 0)),
                    tx_count=int(token_data.get('txCount', 0))
                )
                
                tokens.append(token)
                self.discovered_tokens.add(token_address)
                
        except Exception as e:
            self.logger.error(f"Error processing token data: {e}")
        
        return tokens
    
    def _extract_chain_from_subgraph(self, subgraph_name: str) -> str:
        """Extract chain name from subgraph identifier"""
        if 'ethereum' in subgraph_name:
            return 'ethereum'
        elif 'arbitrum' in subgraph_name:
            return 'arbitrum'
        elif 'polygon' in subgraph_name:
            return 'polygon'
        elif 'optimism' in subgraph_name:
            return 'optimism'
        else:
            return 'ethereum'  # Default
    
    async def _check_rate_limit(self, subgraph_name: str):
        """Implement rate limiting for subgraph requests"""
        now = time.time()
        
        if subgraph_name not in self.rate_limits:
            self.rate_limits[subgraph_name] = []
        
        # Remove old timestamps (older than 1 minute)
        self.rate_limits[subgraph_name] = [
            ts for ts in self.rate_limits[subgraph_name] 
            if now - ts < 60
        ]
        
        # Check if we're over the limit (60 requests per minute)
        if len(self.rate_limits[subgraph_name]) >= 60:
            sleep_time = 60 - (now - self.rate_limits[subgraph_name][0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.rate_limits[subgraph_name].append(now)
    
    async def get_token_historical_data(self, token_address: str, chain: str, days: int = 7) -> List[Dict]:
        """Get historical price data for a specific token"""
        subgraph_url = self._get_subgraph_for_chain(chain)
        if not subgraph_url:
            return []
        
        query = f'''
        {{
          tokenDayDatas(
            first: {days}
            orderBy: date
            orderDirection: desc
            where: {{
              token: "{token_address.lower()}"
            }}
          ) {{
            id
            date
            priceUSD
            volumeUSD
            totalValueLockedUSD
            open
            high
            low
            close
          }}
        }}
        '''
        
        try:
            async with self.session.post(
                subgraph_url,
                json={'query': query}
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get('tokenDayDatas', [])
                
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
        
        return []
    
    def _get_subgraph_for_chain(self, chain: str) -> Optional[str]:
        """Get primary subgraph URL for chain"""
        chain_subgraphs = {
            'ethereum': 'ethereum_uniswap_v3',
            'arbitrum': 'arbitrum_uniswap_v3',
            'polygon': 'polygon_uniswap_v3'
        }
        
        subgraph_key = chain_subgraphs.get(chain)
        return self.subgraphs.get(subgraph_key) if subgraph_key else None
    
    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()

# Global instance
real_graphql_scanner = RealGraphQLScanner()
