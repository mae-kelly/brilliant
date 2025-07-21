import numpy as np
import pandas as pd
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import List, Dict, Any
import time
import logging
import json
from dataclasses import dataclass
import pickle
import mmap

@dataclass
class TokenBatch:
    """Optimized token batch for vectorized processing"""
    addresses: List[str]
    features: np.ndarray  # Vectorized features
    metadata: List[Dict]
    batch_id: str
    timestamp: float

class VectorizedMLProcessor:
    """Ultra-fast vectorized ML processing for 10,000+ tokens"""
    
    def __init__(self, model_path: str, batch_size: int = 256):
        self.batch_size = batch_size
        self.model_path = model_path
        
        # Load optimized TFLite model
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=mp.cpu_count()
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Pre-allocate arrays for maximum performance
        self.input_buffer = np.zeros(
            (self.batch_size, self.input_details[0]['shape'][1]), 
            dtype=np.float32
        )
        self.output_buffer = np.zeros(
            (self.batch_size, self.output_details[0]['shape'][1]), 
            dtype=np.float32
        )
        
        logging.info(f"Vectorized processor initialized: batch_size={batch_size}")
    
    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Ultra-fast batch prediction using vectorized operations"""
        batch_size = features_batch.shape[0]
        
        if batch_size <= self.batch_size:
            # Single batch
            return self._predict_single_batch(features_batch)
        else:
            # Multiple batches
            results = []
            for i in range(0, batch_size, self.batch_size):
                end_idx = min(i + self.batch_size, batch_size)
                batch_features = features_batch[i:end_idx]
                batch_results = self._predict_single_batch(batch_features)
                results.append(batch_results)
            
            return np.concatenate(results, axis=0)
    
    def _predict_single_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """Single batch prediction with optimized memory usage"""
        batch_size = features_batch.shape[0]
        
        # Use pre-allocated buffer to avoid memory allocation
        self.input_buffer[:batch_size] = features_batch
        
        # Run inference
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            self.input_buffer[:batch_size]
        )
        self.interpreter.invoke()
        
        # Get results
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output[:batch_size].copy()

class MemoryMappedCache:
    """High-performance memory-mapped cache for token data"""
    
    def __init__(self, cache_file: str, max_size: int = 1000000):
        self.cache_file = cache_file
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        
        # Try to load existing cache
        try:
            with open(cache_file, 'rb') as f:
                self.cache = pickle.load(f)
        except FileNotFoundError:
            pass
        
        logging.info(f"Memory-mapped cache initialized: {len(self.cache)} items")
    
    def get(self, key: str) -> Any:
        """Get item from cache with LRU tracking"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with automatic eviction"""
        # Evict old items if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru_items(self.max_size // 10)  # Evict 10%
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict_lru_items(self, num_items: int):
        """Evict least recently used items"""
        if not self.access_times:
            return
        
        # Sort by access time
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        
        # Remove oldest items
        for key, _ in sorted_items[:num_items]:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def save(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")

class AsyncTokenScanner:
    """Ultra-fast async token scanner with connection pooling"""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.session_pool = []
        self.semaphore = asyncio.Semaphore(max_connections)
        
        # Connection pool configuration
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        await self.connector.close()
    
    async def scan_tokens_ultra_fast(self, chains: List[str], 
                                   target_count: int = 10000) -> List[TokenBatch]:
        """Ultra-fast token scanning across multiple chains"""
        
        # Create scanning tasks for all chains simultaneously
        scan_tasks = []
        tokens_per_chain = target_count // len(chains)
        
        for chain in chains:
            task = self._scan_single_chain(chain, tokens_per_chain)
            scan_tasks.append(task)
        
        # Execute all scans concurrently
        chain_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Combine results into batches
        all_tokens = []
        for result in chain_results:
            if isinstance(result, list):
                all_tokens.extend(result)
        
        # Create optimized batches
        batches = self._create_optimized_batches(all_tokens)
        
        logging.info(f"Ultra-fast scan completed: {len(all_tokens)} tokens in {len(batches)} batches")
        return batches
    
    async def _scan_single_chain(self, chain: str, target_count: int) -> List[Dict]:
        """Scan single chain with maximum concurrency"""
        
        # Multiple subgraph endpoints for redundancy
        endpoints = self._get_chain_endpoints(chain)
        
        # Create concurrent queries
        query_tasks = []
        tokens_per_query = 100
        num_queries = (target_count + tokens_per_query - 1) // tokens_per_query
        
        for i in range(num_queries):
            skip = i * tokens_per_query
            for endpoint in endpoints:
                task = self._query_subgraph_with_retry(endpoint, skip, tokens_per_query)
                query_tasks.append(task)
        
        # Execute queries concurrently
        query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
        
        # Combine and deduplicate results
        unique_tokens = {}
        for result in query_results:
            if isinstance(result, list):
                for token in result:
                    token_id = token.get('id', '')
                    if token_id and token_id not in unique_tokens:
                        unique_tokens[token_id] = token
        
        return list(unique_tokens.values())[:target_count]
    
    async def _query_subgraph_with_retry(self, endpoint: str, skip: int, 
                                       first: int, max_retries: int = 3) -> List[Dict]:
        """Query subgraph with exponential backoff retry"""
        
        query = """
        query($skip: Int!, $first: Int!) {
          pools(first: $first, skip: $skip, 
                where: {volumeUSD_gt: 500000, liquidity_gt: 100000}) {
            id
            token0 { symbol decimals }
            token1 { symbol decimals }
            volumeUSD
            liquidity
            tick
            sqrtPriceX96
            swaps(first: 50, orderBy: timestamp, orderDirection: desc) {
              amount0
              amount1
              timestamp
              amountUSD
            }
          }
        }
        """
        
        variables = {"skip": skip, "first": first}
        
        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    async with self.session.post(
                        endpoint,
                        json={"query": query, "variables": variables},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get('data', {}).get('pools', [])
                        else:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status
                            )
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Subgraph query failed after {max_retries} attempts: {e}")
                    return []
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return []
    
    def _get_chain_endpoints(self, chain: str) -> List[str]:
        """Get multiple endpoints for redundancy"""
        endpoints = {
            'arbitrum': [
                'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-arbitrum',
                'https://api.thegraph.com/subgraphs/name/camelotlabs/camelot-amm'
            ],
            'polygon': [
                'https://api.thegraph.com/subgraphs/name/pancakeswap/exchange-v3-polygon',
                'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-polygon'
            ],
            'optimism': [
                'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-optimism'
            ]
        }
        
        return endpoints.get(chain, [])
    
    def _create_optimized_batches(self, tokens: List[Dict], 
                                batch_size: int = 256) -> List[TokenBatch]:
        """Create optimized batches for vectorized processing"""
        
        batches = []
        
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i + batch_size]
            
            # Extract features and metadata
            addresses = [token.get('id', '') for token in batch_tokens]
            features = []
            metadata = []
            
            for token in batch_tokens:
                # Extract numerical features for vectorization
                feature_vector = self._extract_feature_vector(token)
                features.append(feature_vector)
                
                # Store metadata separately
                meta = {
                    'symbol': f"{token.get('token0', {}).get('symbol', 'UNK')}/{token.get('token1', {}).get('symbol', 'UNK')}",
                    'liquidity': float(token.get('liquidity', 0)),
                    'volume': float(token.get('volumeUSD', 0)),
                    'swaps': token.get('swaps', [])
                }
                metadata.append(meta)
            
            # Convert to optimized numpy array
            features_array = np.array(features, dtype=np.float32)
            
            batch = TokenBatch(
                addresses=addresses,
                features=features_array,
                metadata=metadata,
                batch_id=f"batch_{i//batch_size}_{int(time.time())}",
                timestamp=time.time()
            )
            
            batches.append(batch)
        
        return batches
    
    def _extract_feature_vector(self, token: Dict) -> List[float]:
        """Extract optimized feature vector from token data"""
        
        # Core liquidity and volume features
        liquidity = float(token.get('liquidity', 0))
        volume = float(token.get('volumeUSD', 0))
        
        # Price features
        sqrt_price = float(token.get('sqrtPriceX96', 0))
        price = (sqrt_price / 2**96) ** 2 if sqrt_price > 0 else 0
        
        # Swap analysis
        swaps = token.get('swaps', [])
        swap_features = self._analyze_swaps(swaps)
        
        # Combine all features
        features = [
            np.log1p(liquidity),  # Log-scaled liquidity
            np.log1p(volume),     # Log-scaled volume
            price,
            swap_features['avg_size'],
            swap_features['frequency'],
            swap_features['direction_bias'],
            swap_features['size_variance'],
            swap_features['time_pattern'],
            swap_features['whale_ratio'],
            swap_features['momentum'],
            swap_features['volatility']
        ]
        
        return features
    
    def _analyze_swaps(self, swaps: List[Dict]) -> Dict[str, float]:
        """Fast swap analysis for feature extraction"""
        
        if not swaps:
            return {
                'avg_size': 0, 'frequency': 0, 'direction_bias': 0,
                'size_variance': 0, 'time_pattern': 0, 'whale_ratio': 0,
                'momentum': 0, 'volatility': 0
            }
        
        # Convert to numpy arrays for vectorized operations
        amounts = np.array([float(swap.get('amountUSD', 0)) for swap in swaps])
        timestamps = np.array([int(swap.get('timestamp', 0)) for swap in swaps])
        
        # Calculate features using vectorized operations
        avg_size = np.mean(amounts) if len(amounts) > 0 else 0
        frequency = len(swaps) / 3600  # Swaps per hour
        
        # Direction bias (positive vs negative amounts)
        direction_bias = np.mean(np.sign(amounts)) if len(amounts) > 0 else 0
        
        # Size variance
        size_variance = np.var(amounts) if len(amounts) > 1 else 0
        
        # Time pattern (regularity of swaps)
        if len(timestamps) > 1:
            time_diffs = np.diff(sorted(timestamps))
            time_pattern = np.std(time_diffs) / (np.mean(time_diffs) + 1)
        else:
            time_pattern = 0
        
        # Whale activity (large trades ratio)
        if len(amounts) > 0:
            large_threshold = np.percentile(amounts, 90)
            whale_ratio = np.sum(amounts > large_threshold) / len(amounts)
        else:
            whale_ratio = 0
        
        # Momentum and volatility
        if len(amounts) > 5:
            momentum = np.corrcoef(range(len(amounts)), amounts)[0, 1]
            volatility = np.std(amounts) / (np.mean(amounts) + 1)
        else:
            momentum = 0
            volatility = 0
        
        return {
            'avg_size': avg_size,
            'frequency': frequency,
            'direction_bias': direction_bias,
            'size_variance': size_variance,
            'time_pattern': time_pattern,
            'whale_ratio': whale_ratio,
            'momentum': momentum if not np.isnan(momentum) else 0,
            'volatility': volatility
        }

class UltraFastPipeline:
    """Complete ultra-fast trading pipeline for 10,000+ tokens/day"""
    
    def __init__(self):
        self.ml_processor = VectorizedMLProcessor('models/momentum_model.tflite')
        self.cache = MemoryMappedCache('data/cache/token_cache.dat')
        self.scanner = None  # Will be initialized in context manager
        
        # Performance metrics
        self.processed_tokens = 0
        self.total_time = 0
        self.start_time = time.time()
        
        logging.info("Ultra-fast pipeline initialized")
    
    async def __aenter__(self):
        self.scanner = AsyncTokenScanner(max_connections=100)
        await self.scanner.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.scanner:
            await self.scanner.__aexit__(exc_type, exc_val, exc_tb)
        self.cache.save()
    
    async def process_tokens_ultra_fast(self, chains: List[str], 
                                      target_count: int = 10000) -> List[Dict]:
        """Process 10,000+ tokens with maximum performance"""
        
        pipeline_start = time.time()
        
        # Phase 1: Ultra-fast token scanning
        logging.info(f"🔍 Scanning {target_count} tokens across {len(chains)} chains...")
        scan_start = time.time()
        
        token_batches = await self.scanner.scan_tokens_ultra_fast(chains, target_count)
        
        scan_time = time.time() - scan_start
        total_tokens = sum(len(batch.addresses) for batch in token_batches)
        
        logging.info(f"✅ Scanned {total_tokens} tokens in {scan_time:.2f}s ({total_tokens/scan_time:.0f} tokens/sec)")
        
        # Phase 2: Vectorized ML processing
        logging.info("🧠 Running vectorized ML inference...")
        ml_start = time.time()
        
        all_predictions = []
        for batch in token_batches:
            # Vectorized prediction for entire batch
            batch_predictions = self.ml_processor.predict_batch(batch.features)
            
            # Combine with metadata
            for i, prediction in enumerate(batch_predictions):
                result = {
                    'address': batch.addresses[i],
                    'prediction': float(prediction[0]) if prediction.ndim > 0 else float(prediction),
                    'metadata': batch.metadata[i],
                    'batch_id': batch.batch_id,
                    'timestamp': batch.timestamp
                }
                all_predictions.append(result)
        
        ml_time = time.time() - ml_start
        logging.info(f"✅ ML inference completed in {ml_time:.2f}s ({len(all_predictions)/ml_time:.0f} predictions/sec)")
        
        # Phase 3: High-value token filtering
        logging.info("🎯 Filtering high-value opportunities...")
        filter_start = time.time()
        
        # Sort by prediction score and apply sophisticated filtering
        high_value_tokens = self._filter_high_value_tokens(all_predictions)
        
        filter_time = time.time() - filter_start
        
        total_time = time.time() - pipeline_start
        
        # Performance summary
        logging.info("📊 ULTRA-FAST PIPELINE PERFORMANCE:")
        logging.info(f"   Total tokens processed: {total_tokens:,}")
        logging.info(f"   High-value opportunities: {len(high_value_tokens)}")
        logging.info(f"   Total pipeline time: {total_time:.2f}s")
        logging.info(f"   Scanning: {scan_time:.2f}s ({total_tokens/scan_time:.0f} tokens/sec)")
        logging.info(f"   ML inference: {ml_time:.2f}s ({len(all_predictions)/ml_time:.0f} predictions/sec)")
        logging.info(f"   Filtering: {filter_time:.2f}s")
        logging.info(f"   Overall throughput: {total_tokens/total_time:.0f} tokens/sec")
        
        # Update performance tracking
        self.processed_tokens += total_tokens
        self.total_time += total_time
        
        return high_value_tokens
    
    def _filter_high_value_tokens(self, predictions: List[Dict]) -> List[Dict]:
        """Advanced filtering for high-value trading opportunities"""
        
        # Sort by prediction score
        sorted_predictions = sorted(predictions, key=lambda x: x['prediction'], reverse=True)
        
        high_value_tokens = []
        
        for token in sorted_predictions:
            prediction = token['prediction']
            metadata = token['metadata']
            
            # Multi-criteria filtering
            if self._passes_quality_filters(prediction, metadata):
                high_value_tokens.append(token)
            
            # Limit to top opportunities
            if len(high_value_tokens) >= 100:
                break
        
        return high_value_tokens
    
    def _passes_quality_filters(self, prediction: float, metadata: Dict) -> bool:
        """Sophisticated quality filtering"""
        
        # Prediction threshold
        if prediction < 0.75:
            return False
        
        # Liquidity requirements
        if metadata['liquidity'] < 250000:  # $250k minimum
            return False
        
        # Volume requirements
        if metadata['volume'] < 1000000:  # $1M daily volume
            return False
        
        # Additional quality checks can be added here
        # - Token age verification
        # - Holder distribution analysis
        # - Contract verification status
        
        return True
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        
        uptime = time.time() - self.start_time
        avg_throughput = self.processed_tokens / max(self.total_time, 1)
        
        return {
            'total_tokens_processed': self.processed_tokens,
            'total_processing_time': self.total_time,
            'system_uptime': uptime,
            'average_throughput': avg_throughput,
            'target_throughput': 10000 / 86400,  # 10k tokens per day
            'efficiency_ratio': avg_throughput / (10000 / 86400),
            'tokens_per_second_peak': avg_throughput,
            'tokens_per_hour_sustained': avg_throughput * 3600
        }
