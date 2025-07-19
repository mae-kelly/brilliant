
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

from safe_operations import logger, retry_on_failure, safe_execute, file_ops, net_ops

import logging
import functools
import traceback
from typing import Any, Callable, Optional
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TradingError(Exception):
    pass

class NetworkError(TradingError):
    pass

class ValidationError(TradingError):
    pass

class SecurityError(TradingError):
    pass

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (NetworkError, ConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            raise last_exception
        return wrapper
    return decorator

def safe_execute(func: Callable, default_return: Any = None, log_errors: bool = True) -> Any:
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Error in {func.__name__ if hasattr(func, '__name__') else 'anonymous'}: {e}")
            logger.debug(traceback.format_exc())
        return default_return

def validate_input(data: Any, validator: Callable[[Any], bool], error_msg: str = "Invalid input"):
    if not validator(data):
        raise ValidationError(error_msg)
    return data

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
                raise TradingError("Circuit breaker is open")
        
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

trading_circuit_breaker = CircuitBreaker()
