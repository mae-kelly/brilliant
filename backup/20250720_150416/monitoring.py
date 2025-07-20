
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


import os
import time
import psutil
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json
import requests
from prometheus_client import Counter, Histogram, Gauge, start_http_server

trades_counter = Counter('trading_bot_trades_total', 'Total trades executed', ['result'])
trade_duration = Histogram('trading_bot_trade_duration_seconds', 'Trade execution time')
profit_gauge = Gauge('trading_bot_profit_usd', 'Current profit in USD')
position_gauge = Gauge('trading_bot_position_size_usd', 'Current position size')
gas_price_gauge = Gauge('trading_bot_gas_price_gwei', 'Current gas price')

@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    successful_trades: int = 0
    total_profit_usd: float = 0.0
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

@dataclass
class SystemMetrics:
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    uptime_hours: float = 0.0

class ProductionMonitoring:
    def __init__(self):
        self.start_time = time.time()
        self.trade_history = deque(maxlen=1000)
        self.profit_history = deque(maxlen=100)
        self.system_alerts = []
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.monitoring_thread = None
        self.running = False
        
        start_http_server(8090)
    
    def start_monitoring(self):
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitoring_loop(self):
        while self.running:
            try:
                self._update_system_metrics()
                self._check_alerts()
                time.sleep(30)
            except Exception as e:
                print(f"Monitoring error: {e}")
    
    def record_trade(self, token: str, amount_usd: float, profit_usd: float, duration: float, success: bool):
        trade_record = {
            'timestamp': time.time(),
            'token': token,
            'amount_usd': amount_usd,
            'profit_usd': profit_usd,
            'duration': duration,
            'success': success
        }
        
        self.trade_history.append(trade_record)
        self.profit_history.append(profit_usd)
        
        trades_counter.labels(result='success' if success else 'failure').inc()
        trade_duration.observe(duration)
        profit_gauge.set(sum(self.profit_history))
        position_gauge.set(amount_usd)
        
        if not success or profit_usd < -10:
            self._send_alert(f"Trade alert: {token} - Profit: ${profit_usd:.2f}")
    
    def _update_system_metrics(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        if cpu_percent > 80:
            self._send_alert(f"High CPU usage: {cpu_percent}%")
        if memory_percent > 85:
            self._send_alert(f"High memory usage: {memory_percent}%")
        if disk_percent > 90:
            self._send_alert(f"High disk usage: {disk_percent}%")
    
    def _check_alerts(self):
        if len(self.profit_history) >= 10:
            recent_losses = sum(1 for p in list(self.profit_history)[-10:] if p < 0)
            if recent_losses >= 7:
                self._send_alert("⚠️ HIGH LOSS RATE: 7+ losses in last 10 trades")
        
        if len(self.profit_history) >= 50:
            total_profit = sum(self.profit_history)
            if total_profit < -100:
                self._send_alert(f"🚨 LARGE LOSS: Total profit: ${total_profit:.2f}")
    
    def _send_alert(self, message: str):
        if message in [alert['message'] for alert in self.system_alerts[-5:]]:
            return
        
        alert = {
            'timestamp': time.time(),
            'message': message
        }
        self.system_alerts.append(alert)
        
        if self.webhook_url:
            try:
                payload = {
                    'content': f"🤖 Trading Bot Alert: {message}",
                    'username': 'DeFi Sniper Bot'
                }
                requests.post(self.webhook_url, json=payload, timeout=10)
            except:
                pass
        
        print(f"ALERT: {message}")
    
    def get_performance_report(self) -> PerformanceMetrics:
        if not self.trade_history:
            return PerformanceMetrics()
        
        total_trades = len(self.trade_history)
        successful_trades = sum(1 for t in self.trade_history if t['success'])
        total_profit = sum(t['profit_usd'] for t in self.trade_history)
        
        win_rate = successful_trades / total_trades if total_trades > 0 else 0
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        profits = [t['profit_usd'] for t in self.trade_history]
        running_max = 0
        max_drawdown = 0
        running_total = 0
        
        for profit in profits:
            running_total += profit
            if running_total > running_max:
                running_max = running_total
            drawdown = (running_max - running_total) / running_max if running_max > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return PerformanceMetrics(
            total_trades=total_trades,
            successful_trades=successful_trades,
            total_profit_usd=total_profit,
            win_rate=win_rate,
            avg_profit_per_trade=avg_profit,
            max_drawdown=max_drawdown,
            sharpe_ratio=self._calculate_sharpe_ratio(profits)
        )
    
    def _calculate_sharpe_ratio(self, profits: List[float]) -> float:
        if len(profits) < 2:
            return 0.0
        
        import numpy as np
        returns = np.array(profits)
        excess_return = np.mean(returns)
        volatility = np.std(returns)
        
        return excess_return / volatility if volatility > 0 else 0.0

monitor = ProductionMonitoring()
