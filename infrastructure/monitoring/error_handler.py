import logging
import traceback
import asyncio
from functools import wraps
from typing import Optional, Dict, Any
import json
import time

class TradingSystemError(Exception):
    """Base exception for trading system"""
    pass

class InsufficientFundsError(TradingSystemError):
    """Raised when wallet has insufficient funds"""
    pass

class NetworkError(TradingSystemError):
    """Raised when network connection fails"""
    pass

class ModelInferenceError(TradingSystemError):
    """Raised when ML model inference fails"""
    pass

class SafetyCheckError(TradingSystemError):
    """Raised when safety checks fail"""
    pass

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except (NetworkError, ConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        logging.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    continue
                except Exception as e:
                    # Don't retry other types of exceptions
                    raise e
            
            raise last_exception
        return wrapper
    return decorator

def log_performance(func):
    """Decorator to log function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            logging.info(json.dumps({
                'event': 'function_performance',
                'function': func.__name__,
                'execution_time': execution_time,
                'success': True
            }))
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(json.dumps({
                'event': 'function_performance',
                'function': func.__name__,
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }))
            raise
    return wrapper

class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time < self.timeout:
                    raise TradingSystemError(f"Circuit breaker OPEN for {func.__name__}")
                else:
                    self.state = 'HALF_OPEN'
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - reset failure count
                self.failure_count = 0
                self.state = 'CLOSED'
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logging.error(f"Circuit breaker OPENED for {func.__name__}")
                
                raise e
        return wrapper

def safe_execute(default_return=None):
    """Decorator to safely execute functions with fallback"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                logging.error(json.dumps({
                    'event': 'safe_execute_fallback',
                    'function': func.__name__,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }))
                return default_return
        return wrapper
    return decorator

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
    
    def register_check(self, name: str, check_func, interval: int = 60):
        """Register a health check function"""
        self.checks[name] = {'func': check_func, 'interval': interval}
        self.last_check_time[name] = 0
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        current_time = time.time()
        
        for name, check_info in self.checks.items():
            if current_time - self.last_check_time[name] >= check_info['interval']:
                try:
                    if asyncio.iscoroutinefunction(check_info['func']):
                        result = await check_info['func']()
                    else:
                        result = check_info['func']()
                    
                    results[name] = {'status': 'healthy', 'result': result}
                    self.last_check_time[name] = current_time
                    
                except Exception as e:
                    results[name] = {'status': 'unhealthy', 'error': str(e)}
                    logging.error(f"Health check {name} failed: {e}")
        
        return results
