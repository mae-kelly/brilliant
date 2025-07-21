import logging
import logging.handlers
import json
import os
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'chain'):
            log_entry['chain'] = record.chain
        if hasattr(record, 'token'):
            log_entry['token'] = record.token
        if hasattr(record, 'trade_id'):
            log_entry['trade_id'] = record.trade_id
        
        return json.dumps(log_entry)

def setup_logging(log_level='INFO', log_dir='logs'):
    """Setup comprehensive logging configuration"""
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    json_formatter = JSONFormatter()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    all_logs_handler = logging.handlers.RotatingFileHandler(
        f'{log_dir}/trading_system.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5
    )
    all_logs_handler.setFormatter(json_formatter)
    root_logger.addHandler(all_logs_handler)
    
    # Separate handler for errors
    error_handler = logging.handlers.RotatingFileHandler(
        f'{log_dir}/errors.log',
        maxBytes=50*1024*1024,  # 50MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    root_logger.addHandler(error_handler)
    
    # Trading-specific logger
    trading_logger = logging.getLogger('trading')
    trading_handler = logging.handlers.RotatingFileHandler(
        f'{log_dir}/trades.log',
        maxBytes=50*1024*1024,
        backupCount=5
    )
    trading_handler.setFormatter(json_formatter)
    trading_logger.addHandler(trading_handler)
    
    # Performance logger
    perf_logger = logging.getLogger('performance')
    perf_handler = logging.handlers.RotatingFileHandler(
        f'{log_dir}/performance.log',
        maxBytes=25*1024*1024,
        backupCount=3
    )
    perf_handler.setFormatter(json_formatter)
    perf_logger.addHandler(perf_handler)
    
    return root_logger

# Custom log functions for specific events
def log_trade_event(event_type, chain, token, **kwargs):
    """Log trading-specific events"""
    logger = logging.getLogger('trading')
    logger.info('', extra={
        'event_type': event_type,
        'chain': chain,
        'token': token,
        **kwargs
    })

def log_performance_metric(metric_name, value, **kwargs):
    """Log performance metrics"""
    logger = logging.getLogger('performance')
    logger.info('', extra={
        'metric_name': metric_name,
        'value': value,
        **kwargs
    })
