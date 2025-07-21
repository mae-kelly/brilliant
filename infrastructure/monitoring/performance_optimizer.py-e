import psutil
import GPUtil
import numpy as np
import logging
import json
import time
import threading
from typing import Dict, List
import yaml
import subprocess
import os

class SystemOptimizer:
    """Optimizes system performance for maximum trading efficiency"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.gpus = GPUtil.getGPUs()
        
        logging.info(f"System: {self.cpu_count} CPUs, {self.memory_gb:.1f}GB RAM, {len(self.gpus)} GPUs")
    
    def optimize_system_performance(self):
        """Apply comprehensive system optimizations"""
        
        logging.info("🚀 Applying system optimizations...")
        
        # CPU optimizations
        self._optimize_cpu_performance()
        
        # Memory optimizations
        self._optimize_memory_usage()
        
        # GPU optimizations
        self._optimize_gpu_performance()
        
        # Network optimizations
        self._optimize_network_settings()
        
        # Python runtime optimizations
        self._optimize_python_runtime()
        
        logging.info("✅ System optimizations applied")
    
    def _optimize_cpu_performance(self):
        """Optimize CPU performance and affinity"""
        try:
            # Set CPU affinity for current process
            process = psutil.Process()
            
            # Use all available cores
            cpu_cores = list(range(self.cpu_count))
            process.cpu_affinity(cpu_cores)
            
            # Set high priority
            if os.name == 'posix':  # Linux/Mac
                os.nice(-10)  # Higher priority
            
            logging.info(f"✅ CPU optimization: {self.cpu_count} cores, high priority")
            
        except Exception as e:
            logging.warning(f"CPU optimization failed: {e}")
    
    def _optimize_memory_usage(self):
        """Optimize memory usage and allocation"""
        try:
            # Configure numpy for optimal memory usage
            os.environ['OPENBLAS_NUM_THREADS'] = str(self.cpu_count)
            os.environ['MKL_NUM_THREADS'] = str(self.cpu_count)
            
            # Optimize garbage collection
            import gc
            gc.set_threshold(700, 10, 10)  # More aggressive GC
            
            logging.info("✅ Memory optimization: threads configured, GC optimized")
            
        except Exception as e:
            logging.warning(f"Memory optimization failed: {e}")
    
    def _optimize_gpu_performance(self):
        """Optimize GPU performance for ML inference"""
        try:
            if self.gpus:
                # Set GPU memory growth for TensorFlow
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                os.environ['TF_GPU_MEMORY_LIMIT'] = '4096'  # 4GB limit
                
                # CUDA optimizations
                os.environ['CUDA_CACHE_DISABLE'] = '0'
                os.environ['CUDA_CACHE_MAXSIZE'] = '2147483647'
                
                logging.info(f"✅ GPU optimization: {len(self.gpus)} GPUs configured")
            else:
                logging.info("No GPUs detected, using CPU optimization")
                
        except Exception as e:
            logging.warning(f"GPU optimization failed: {e}")
    
    def _optimize_network_settings(self):
        """Optimize network settings for high-throughput trading"""
        try:
            # Set environment variables for aiohttp optimization
            os.environ['AIOHTTP_NO_EXTENSIONS'] = '1'  # Disable C extensions if problematic
            
            # TCP optimizations (Linux)
            if os.name == 'posix':
                try:
                    # These require root privileges
                    subprocess.run(['sysctl', '-w', 'net.core.rmem_max=16777216'], 
                                 capture_output=True)
                    subprocess.run(['sysctl', '-w', 'net.core.wmem_max=16777216'], 
                                 capture_output=True)
                except:
                    pass  # Ignore if we don't have permissions
            
            logging.info("✅ Network optimization: TCP buffers configured")
            
        except Exception as e:
            logging.warning(f"Network optimization failed: {e}")
    
    def _optimize_python_runtime(self):
        """Optimize Python runtime for maximum performance"""
        try:
            # Disable bytecode generation (faster startup)
            os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
            
            # Optimize import system
            os.environ['PYTHONOPTIMIZE'] = '2'
            
            # Use faster JSON library if available
            try:
                import ujson
                import json
                json.loads = ujson.loads
                json.dumps = ujson.dumps
            except ImportError:
                pass
            
            logging.info("✅ Python runtime optimization: bytecode disabled, imports optimized")
            
        except Exception as e:
            logging.warning(f"Python optimization failed: {e}")

class PerformanceMonitor:
    """Real-time performance monitoring and alerting"""
    
    def __init__(self, alert_thresholds: Dict = None):
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 90,
            'memory_percent': 85,
            'gpu_memory_percent': 90,
            'disk_percent': 95,
            'inference_latency_ms': 200
        }
        
        self.monitoring = False
        self.monitor_thread = None
        self.performance_history = []
        
    def start_monitoring(self, interval: float = 5.0):
        """Start real-time performance monitoring"""
        
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logging.info(f"Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logging.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self._check_alerts(metrics)
                self._store_metrics(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> Dict:
        """Collect comprehensive system metrics"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        network = psutil.net_io_counters()
        
        # GPU metrics
        gpu_metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature,
                    'load': gpu.load * 100
                })
        except:
            pass
        
        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'timestamp': time.time(),
            'cpu': {
                'percent': cpu_percent,
                'frequency_mhz': cpu_freq.current if cpu_freq else 0,
                'cores': psutil.cpu_count()
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent,
                'available_gb': memory.available / (1024**3)
            },
            'swap': {
                'total_gb': swap.total / (1024**3),
                'used_gb': swap.used / (1024**3),
                'percent': swap.percent
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'percent': (disk.used / disk.total) * 100
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            'gpu': gpu_metrics,
            'process': {
                'memory_rss_mb': process_memory.rss / (1024**2),
                'memory_vms_mb': process_memory.vms / (1024**2),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
        }
    
    def _check_alerts(self, metrics: Dict):
        """Check for performance alerts"""
        
        alerts = []
        
        # CPU alert
        if metrics['cpu']['percent'] > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'value': metrics['cpu']['percent'],
                'threshold': self.alert_thresholds['cpu_percent'],
                'severity': 'warning'
            })
        
        # Memory alert
        if metrics['memory']['percent'] > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'value': metrics['memory']['percent'],
                'threshold': self.alert_thresholds['memory_percent'],
                'severity': 'warning'
            })
        
        # GPU alerts
        for gpu in metrics['gpu']:
            if gpu['memory_percent'] > self.alert_thresholds['gpu_memory_percent']:
                alerts.append({
                    'type': 'gpu_memory_high',
                    'gpu_id': gpu['id'],
                    'value': gpu['memory_percent'],
                    'threshold': self.alert_thresholds['gpu_memory_percent'],
                    'severity': 'warning'
                })
        
        # Disk alert
        if metrics['disk']['percent'] > self.alert_thresholds['disk_percent']:
            alerts.append({
                'type': 'disk_full',
                'value': metrics['disk']['percent'],
                'threshold': self.alert_thresholds['disk_percent'],
                'severity': 'critical'
            })
        
        # Log alerts
        for alert in alerts:
            logging.warning(json.dumps({
                'event': 'performance_alert',
                'alert': alert,
                'timestamp': metrics['timestamp']
            }))
    
    def _store_metrics(self, metrics: Dict):
        """Store metrics in history"""
        
        self.performance_history.append(metrics)
        
        # Keep only recent history (last hour)
        max_history = 720  # 1 hour at 5s intervals
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
    
    def get_performance_summary(self, duration_minutes: int = 10) -> Dict:
        """Get performance summary for specified duration"""
        
        if not self.performance_history:
            return {}
        
        # Filter recent history
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [m for m in self.performance_history if m['timestamp'] > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [m['cpu']['percent'] for m in recent_metrics]
        memory_values = [m['memory']['percent'] for m in recent_metrics]
        
        summary = {
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory': {
                'avg': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            }
        }
        
        # GPU statistics if available
        if recent_metrics[0]['gpu']:
            gpu_memory_values = [m['gpu'][0]['memory_percent'] for m in recent_metrics if m['gpu']]
            if gpu_memory_values:
                summary['gpu'] = {
                    'memory_avg': np.mean(gpu_memory_values),
                    'memory_max': np.max(gpu_memory_values),
                    'memory_min': np.min(gpu_memory_values)
                }
        
        return summary

def optimize_settings_for_performance():
    """Optimize settings.yaml for maximum performance"""
    
    logging.info("🔧 Optimizing settings for performance...")
    
    with open('settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)
    
    # Performance optimizations
    performance_settings = {
        'scanning': {
            'max_tokens_per_scan': 15000,  # Increased from 10000
            'concurrent_requests': 100,    # Increased for A100 GPU
            'batch_size': 256,            # Optimized for GPU memory
            'scan_interval_seconds': 15,   # Faster scanning
            'connection_pool_size': 200
        },
        'ml': {
            'batch_inference_size': 512,   # Large batch for GPU
            'feature_cache_size': 50000,   # Increased cache
            'inference_timeout': 0.05,     # 50ms timeout
            'model_update_frequency': 1800, # 30 min updates
            'use_mixed_precision': True,
            'enable_tensorrt': True
        },
        'performance': {
            'max_concurrent_requests': 200,
            'cache_ttl_seconds': 15,       # Faster cache expiry
            'prediction_timeout': 3,       # Reduced timeout
            'trade_execution_timeout': 15,
            'health_check_interval': 30,
            'enable_fast_math': True,
            'optimize_memory': True
        },
        'network_config': {}
    }
    
    # Update chain-specific settings for high performance
    for chain in ['arbitrum', 'polygon', 'optimism']:
        performance_settings['network_config'][chain] = {
            'chain_id': settings['network_config'][chain]['chain_id'],
            'gas_multiplier': settings['network_config'][chain]['gas_multiplier'],
            'confirmation_blocks': 1,  # Faster confirmations
            'priority_fee': settings['network_config'][chain]['priority_fee'],
            'rpc_timeout': 15,         # Faster timeout
            'max_retries': 2,          # Fewer retries for speed
            'connection_pool_size': 50,
            'request_rate_limit': 30   # Higher rate limit
        }
    
    # Merge with existing settings
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(settings, performance_settings)
    
    # Save optimized settings
    with open('settings.yaml', 'w') as f:
        yaml.dump(settings, f, default_flow_style=False, indent=2)
    
    logging.info("✅ Settings optimized for maximum performance")
    
    return settings
