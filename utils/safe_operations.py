
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
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
# Dynamic configuration import


import logging
import time
import functools
import requests
from typing import Any, Callable, Optional, Dict
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SafeFileOperations:
    @staticmethod
    def load_json(filepath: str, default: dict = None) -> dict:
        try:
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                return default or {}
            
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            return default or {}
    
    @staticmethod
    def save_json(filepath: str, data: dict) -> bool:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {filepath}: {e}")
            return False

class SafeNetworkOperations:
    @staticmethod
    def safe_request(url: str, method: str = 'GET', **kwargs) -> Optional[dict]:
        try:
            kwargs.setdefault('timeout', 10)
            kwargs.setdefault('headers', {})
            
            if method.upper() == 'GET':
                response = requests.get(url, **kwargs, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, **kwargs, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from {url}")
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
        
        return None

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator

def safe_execute(func: Callable, default_return: Any = None) -> Any:
    try:
        return func()
    except Exception as e:
        logger.error(f"Error in {func.__name__ if hasattr(func, '__name__') else 'function'}: {e}")
        return default_return

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'
    
    def call(self, func: Callable, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = 'half-open'
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise

circuit_breaker = CircuitBreaker()
file_ops = SafeFileOperations()
net_ops = SafeNetworkOperations()
