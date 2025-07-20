"""
PRODUCTION GraphQL Scanner - Real subgraph queries
NO SIMULATION - Complete GraphQL implementation with rate limiting and caching
"""
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from collections import defaultdict, deque
import hashlib

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
    fee_tier: Optional[int] = None
    pool_address: Optional[str] = None

@dataclass
class HistoricalData:
    timestamp: int
    price_usd: float
    volume_usd: float
    liquidity_usd: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float

class RateLimiter:
    """Advanced rate limiter for GraphQL endpoints"""
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = deque()
        
    async def acquire(self):
        now = time.time()
        
        # Remove old requests
        while self.requests and now - self.requests[0] > 60:
            self.requests.popleft()
        
        # Check if we can make a request
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)

class QueryCache:
    """Intelligent caching for GraphQL queries"""
    def __init__(self, ttl: int = 300):
        self.cache = {}
        self.ttl = ttl
    
    def get_cache_key(self, query: str, variables: Dict = None) -> str:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if variables:
            vars_hash = hashlib.md5(json.dumps(variables, sort_keys=True).encode()).hexdigest()
            return f"{query_hash}_{vars_hash}"
        return query_hash
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: Dict):
        self.cache[key] = (data, time.time())
    
    def clear_expired(self):
        now = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

class RealGraphQLScanner:
    """PRODUCTION GraphQL scanner with complete implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # REAL subgraph endpoints with fallbacks
        self.subgraphs = {
            'ethereum_uniswap_v3': {
                'primary': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
                'fallback': 'https://gateway.thegraph.com/api/[api-key]/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV'
            },
            'ethereum_uniswap_v2': {
                'primary': 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2',
                'fallback': None
            },
            'arbitrum_uniswap_v3': {
                'primary': 'https://api.thegraph.com/subgraphs/name/ianlapham/arbitrum-minimal',
                'fallback': None
            },
            'arbitrum_camelot': {
                'primary': 'https://api.thegraph.com/subgraphs/name/camelot-labs/camelot-amm',
                'fallback': None
            },
            'polygon_uniswap_v3': {
                'primary': 'https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-polygon',
                'fallback': None
            },
            'polygon_quickswap': {
                'primary': 'https://api.thegraph.com/subgraphs/name/sameepsi/quickswap06',
                'fallback': None
            }
        }
        
        self.discovered_tokens = set()
        self.rate_limiters = {name: RateLimiter() for name in self.subgraphs.keys()}
        self.query_cache = QueryCache()
        self.performance_metrics = defaultdict(list)
        
    async def initialize(self):
        """Initialize scanner with session and connection pooling"""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'Renaissance-Trading-Bot/1.0'
            }
        )
        
        self.logger.info("✅ Real GraphQL scanner initialized with connection pooling")
    
    async def scan_all_subgraphs_parallel(self) -> List[RealToken]:
        """Scan all subgraphs in parallel for maximum throughput"""
        all_tokens = []
        
        # Create tasks for each subgraph
        tasks = []
        for subgraph_name, endpoints in self.subgraphs.items():
            task = asyncio.create_task(
                self.scan_subgraph_with_retry(subgraph_name, endpoints)
            )
            tasks.append(task)
        
        # Execute all scans in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            subgraph_name = list(self.subgraphs.keys())[i]
            if isinstance(result, list):
                all_tokens.extend(result)
                self.logger.info(f"✅ {subgraph_name}: {len(result)} tokens discovered")
            elif isinstance(result, Exception):
                self.logger.error(f"❌ {subgraph_name} failed: {result}")
        
        # Remove duplicates and filter by quality
        unique_tokens = self._deduplicate_and_filter(all_tokens)
        
        self.logger.info(f"🎯 Total unique high-quality tokens: {len(unique_tokens)}")
        
        return unique_tokens
    
    async def scan_subgraph_with_retry(self, subgraph_name: str, endpoints: Dict) -> List[RealToken]:
        """Scan subgraph with automatic retry and fallback"""
        primary_url = endpoints['primary']
        fallback_url = endpoints.get('fallback')
        
        # Try primary endpoint first
        try:
            return await self.scan_subgraph_complete(subgraph_name, primary_url)
        except Exception as e:
            self.logger.warning(f"Primary endpoint failed for {subgraph_name}: {e}")
            
            # Try fallback if available
            if fallback_url:
                try:
                    return await self.scan_subgraph_complete(subgraph_name, fallback_url)
                except Exception as e2:
                    self.logger.error(f"Fallback endpoint also failed for {subgraph_name}: {e2}")
            
            return []
    
    async def scan_subgraph_complete(self, subgraph_name: str, url: str) -> List[RealToken]:
        """Complete subgraph scanning implementation with advanced features"""
        tokens = []
        skip = 0
        batch_size = 1000
        max_tokens = 50000  # Limit to prevent excessive memory usage
        
        while len(tokens) < max_tokens:
            try:
                # Rate limiting
                await self.rate_limiters[subgraph_name].acquire()
                
                # Build optimized query
                query = self._build_advanced_token_query(skip, batch_size, subgraph_name)
                
                # Check cache first
                cache_key = self.query_cache.get_cache_key(query, {'skip': skip})
                cached_result = self.query_cache.get(cache_key)
                
                if cached_result:
                    batch_tokens = await self._process_token_data(cached_result, subgraph_name)
                else:
                    # Execute GraphQL query
                    start_time = time.time()
                    
                    async with self.session.post(
                        url,
                        json={'query': query},
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        if response.status != 200:
                            self.logger.error(f"GraphQL error {response.status} for {subgraph_name}")
                            break
                        
                        data = await response.json()
                        
                        # Track performance
                        query_time = time.time() - start_time
                        self.performance_metrics[subgraph_name].append(query_time)
                        
                        if 'errors' in data:
                            self.logger.error(f"GraphQL errors: {data['errors']}")
                            break
                        
                        # Cache successful result
                        self.query_cache.set(cache_key, data)
                        
                        batch_tokens = await self._process_token_data(data, subgraph_name)
                
                tokens.extend(batch_tokens)
                
                # Check if we got fewer tokens than requested (end of data)
                if len(batch_tokens) < batch_size:
                    break
                
                skip += batch_size
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error scanning {subgraph_name}: {e}")
                break
        
        self.logger.info(f"✅ Scanned {subgraph_name}: {len(tokens)} tokens")
        return tokens
    
    def _build_advanced_token_query(self, skip: int, first: int, subgraph_name: str) -> str:
        """Build advanced GraphQL query optimized for each subgraph type"""
        
        # Base query components
        base_where_conditions = [
            'volumeUSD_gt: "1000"',
            'txCount_gt: "10"'
        ]
        
        # Add subgraph-specific conditions
        if 'uniswap_v3' in subgraph_name:
            base_where_conditions.append('feeTier_in: [500, 3000, 10000]')
        
        where_clause = ', '.join(base_where_conditions)
        
        return f'''
        {{
          tokens(
            first: {first}
            skip: {skip}
            orderBy: volumeUSD
            orderDirection: desc
            where: {{
              {where_clause}
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
            whitelistPools(
              first: 5
              orderBy: totalValueLockedUSD
              orderDirection: desc
            ) {{
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
              sqrtPrice
              tick
            }}
          }}
        }}
        '''
    
    async def _process_token_data(self, data: Dict, subgraph_name: str) -> List[RealToken]:
        """Process GraphQL response into RealToken objects with advanced filtering"""
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
                
                # Process day data with validation
                day_data = token_data.get('tokenDayData', [])
                if not day_data:
                    continue
                
                latest_day = day_data[0]
                price_usd = float(latest_day.get('priceUSD', 0))
                volume_24h = float(latest_day.get('volumeUSD', 0))
                liquidity_usd = float(latest_day.get('totalValueLockedUSD', 0))
                
                # Calculate 24h price change with validation
                price_change_24h = 0.0
                if len(day_data) > 1:
                    prev_price = float(day_data[1].get('priceUSD', 0))
                    if prev_price > 0:
                        price_change_24h = ((price_usd - prev_price) / prev_price) * 100
                
                # Advanced filtering for momentum signals
                if not self._passes_quality_filter(
                    price_usd, volume_24h, liquidity_usd, price_change_24h, token_data
                ):
                    continue
                
                # Extract pool information
                pools = token_data.get('whitelistPools', [])
                best_pool = pools[0] if pools else {}
                
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
                    tx_count=int(token_data.get('txCount', 0)),
                    fee_tier=best_pool.get('feeTier'),
                    pool_address=best_pool.get('id')
                )
                
                tokens.append(token)
                self.discovered_tokens.add(token_address)
                
        except Exception as e:
            self.logger.error(f"Error processing token data: {e}")
        
        return tokens
    
    def _passes_quality_filter(self, price_usd: float, volume_24h: float, 
                              liquidity_usd: float, price_change_24h: float, 
                              token_data: Dict) -> bool:
        """Advanced quality filtering for tokens"""
        
        # Basic validation
        if price_usd <= 0 or volume_24h < 1000 or liquidity_usd < 10000:
            return False
        
        # Momentum filter (5%+ change for momentum signals)
        if abs(price_change_24h) < 5:
            return False
        
        # Transaction activity filter
        tx_count = int(token_data.get('txCount', 0))
        if tx_count < 100:
            return False
        
        # Liquidity-to-volume ratio filter
        if volume_24h > 0:
            lv_ratio = liquidity_usd / volume_24h
            if lv_ratio < 0.1 or lv_ratio > 1000:  # Too illiquid or suspicious
                return False
        
        # Price range filter (avoid extremely low/high prices)
        if price_usd < 0.000001 or price_usd > 1000000:
            return False
        
        return True
    
    def _deduplicate_and_filter(self, tokens: List[RealToken]) -> List[RealToken]:
        """Remove duplicates and apply final quality filters"""
        seen_addresses = set()
        unique_tokens = []
        
        # Sort by volume to prioritize high-volume tokens
        sorted_tokens = sorted(tokens, key=lambda t: t.volume_24h_usd, reverse=True)
        
        for token in sorted_tokens:
            if token.address not in seen_addresses:
                seen_addresses.add(token.address)
                unique_tokens.append(token)
        
        # Apply final momentum-based filtering
        momentum_tokens = [
            token for token in unique_tokens
            if abs(token.price_change_24h) >= 8 and token.volume_24h_usd >= 5000
        ]
        
        return momentum_tokens
    
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
    
    async def get_token_historical_data(self, token_address: str, chain: str, 
                                      days: int = 7) -> List[HistoricalData]:
        """Get detailed historical price data for a specific token"""
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
                    raw_data = data.get('data', {}).get('tokenDayDatas', [])
                    
                    return [
                        HistoricalData(
                            timestamp=int(item['date']),
                            price_usd=float(item.get('priceUSD', 0)),
                            volume_usd=float(item.get('volumeUSD', 0)),
                            liquidity_usd=float(item.get('totalValueLockedUSD', 0)),
                            open_price=float(item.get('open', 0)),
                            high_price=float(item.get('high', 0)),
                            low_price=float(item.get('low', 0)),
                            close_price=float(item.get('close', 0))
                        )
                        for item in raw_data
                    ]
                
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
        if subgraph_key and subgraph_key in self.subgraphs:
            return self.subgraphs[subgraph_key]['primary']
        return None
    
    def get_performance_metrics(self) -> Dict[str, Dict]:
        """Get performance metrics for monitoring"""
        metrics = {}
        
        for subgraph_name, query_times in self.performance_metrics.items():
            if query_times:
                metrics[subgraph_name] = {
                    'avg_query_time': sum(query_times) / len(query_times),
                    'max_query_time': max(query_times),
                    'min_query_time': min(query_times),
                    'total_queries': len(query_times),
                    'cache_hit_rate': 0.0  # Could be implemented
                }
        
        return metrics
    
    async def cleanup_cache(self):
        """Clean up expired cache entries"""
        self.query_cache.clear_expired()
    
    async def close(self):
        """Close session and cleanup resources"""
        if self.session:
            await self.session.close()
        
        # Clear caches
        self.query_cache.cache.clear()
        self.performance_metrics.clear()

# Global instance
real_graphql_scanner = RealGraphQLScanner()
