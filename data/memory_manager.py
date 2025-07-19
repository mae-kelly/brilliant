import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import psutil
import gc
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import weakref
import threading
from dataclasses import dataclass

@dataclass
class MemoryStats:
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float
    cache_size: int
    object_count: int

class LRUCache:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)

    def remove(self, key: str):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_order.remove(key)

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

    def size(self) -> int:
        with self.lock:
            return len(self.cache)

class MemoryManager:
    def __init__(self):
        self.token_cache = LRUCache(max_size=50000)
        self.price_cache = LRUCache(max_size=100000)
        self.signal_cache = LRUCache(max_size=20000)
        self.feature_cache = LRUCache(max_size=30000)
        
        self.memory_threshold = 0.85
        self.cleanup_threshold = 0.90
        self.monitoring = True
        
        self.weak_references = weakref.WeakSet()
        self.object_pools = defaultdict(list)
        
        self.stats = {
            'memory_cleanups': 0,
            'cache_evictions': 0,
            'gc_collections': 0,
            'peak_memory': 0.0
        }

    async def start_monitoring(self):
        asyncio.create_task(self.memory_monitor_loop())
        asyncio.create_task(self.periodic_cleanup())

    async def memory_monitor_loop(self):
        while self.monitoring:
            try:
                memory_stats = self.get_memory_stats()
                
                if memory_stats.memory_percent > self.cleanup_threshold:
                    await self.emergency_cleanup()
                elif memory_stats.memory_percent > self.memory_threshold:
                    await self.moderate_cleanup()
                
                self.stats['peak_memory'] = max(self.stats['peak_memory'], memory_stats.used_memory)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                await asyncio.sleep(30)

    async def periodic_cleanup(self):
        while self.monitoring:
            try:
                await asyncio.sleep(300)
                await self.routine_cleanup()
                
            except Exception as e:
                await asyncio.sleep(60)

    def get_memory_stats(self) -> MemoryStats:
        memory = psutil.virtual_memory()
        
        total_cache_size = (
            self.token_cache.size() +
            self.price_cache.size() +
            self.signal_cache.size() +
            self.feature_cache.size()
        )
        
        return MemoryStats(
            total_memory=memory.total / (1024**3),
            available_memory=memory.available / (1024**3),
            used_memory=memory.used / (1024**3),
            memory_percent=memory.percent,
            cache_size=total_cache_size,
            object_count=len(gc.get_objects())
        )

    async def emergency_cleanup(self):
        self.stats['memory_cleanups'] += 1
        
        self.token_cache.clear()
        self.price_cache.clear()
        self.signal_cache.clear()
        self.feature_cache.clear()
        
        self.clear_object_pools()
        
        gc.collect()
        self.stats['gc_collections'] += 1
        
        print("🚨 Emergency memory cleanup executed")

    async def moderate_cleanup(self):
        self.stats['memory_cleanups'] += 1
        
        self.reduce_cache_sizes(0.5)
        
        self.clear_old_object_pools()
        
        gc.collect()
        self.stats['gc_collections'] += 1
        
        print("⚠️ Moderate memory cleanup executed")

    async def routine_cleanup(self):
        self.reduce_cache_sizes(0.8)
        
        self.clear_old_object_pools()
        
        if gc.isenabled():
            collected = gc.collect()
            if collected > 0:
                self.stats['gc_collections'] += 1

    def reduce_cache_sizes(self, factor: float):
        caches = [self.token_cache, self.price_cache, self.signal_cache, self.feature_cache]
        
        for cache in caches:
            target_size = int(cache.max_size * factor)
            
            with cache.lock:
                while len(cache.cache) > target_size and cache.access_order:
                    oldest = cache.access_order.pop(0)
                    del cache.cache[oldest]
                    self.stats['cache_evictions'] += 1

    def clear_object_pools(self):
        for pool in self.object_pools.values():
            pool.clear()
        self.object_pools.clear()

    def clear_old_object_pools(self):
        for pool_name, pool in list(self.object_pools.items()):
            if len(pool) > 1000:
                pool.clear()
                del self.object_pools[pool_name]

    def cache_token_data(self, key: str, data: Any):
        self.token_cache.put(key, data)

    def get_cached_token_data(self, key: str) -> Any:
        return self.token_cache.get(key)

    def cache_price_data(self, key: str, data: Any):
        self.price_cache.put(key, data)

    def get_cached_price_data(self, key: str) -> Any:
        return self.price_cache.get(key)

    def cache_signal_data(self, key: str, data: Any):
        self.signal_cache.put(key, data)

    def get_cached_signal_data(self, key: str) -> Any:
        return self.signal_cache.get(key)

    def cache_feature_data(self, key: str, data: Any):
        self.feature_cache.put(key, data)

    def get_cached_feature_data(self, key: str) -> Any:
        return self.feature_cache.get(key)

    def register_weak_reference(self, obj: Any):
        self.weak_references.add(obj)

    def get_object_from_pool(self, pool_name: str, factory_func):
        pool = self.object_pools[pool_name]
        
        if pool:
            return pool.pop()
        else:
            return factory_func()

    def return_object_to_pool(self, pool_name: str, obj: Any):
        pool = self.object_pools[pool_name]
        
        if len(pool) < 100:
            if hasattr(obj, 'reset'):
                obj.reset()
            pool.append(obj)

    def get_cache_info(self) -> Dict[str, Any]:
        return {
            'token_cache_size': self.token_cache.size(),
            'price_cache_size': self.price_cache.size(),
            'signal_cache_size': self.signal_cache.size(),
            'feature_cache_size': self.feature_cache.size(),
            'weak_references': len(self.weak_references),
            'object_pools': {name: len(pool) for name, pool in self.object_pools.items()},
            'stats': self.stats.copy()
        }

    async def shutdown(self):
        self.monitoring = False
        
        self.token_cache.clear()
        self.price_cache.clear()
        self.signal_cache.clear()
        self.feature_cache.clear()
        
        self.clear_object_pools()
        
        gc.collect()

memory_manager = MemoryManager()
