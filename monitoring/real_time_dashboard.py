import asyncio
import websockets
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import uvicorn
import psutil
import os

@dataclass
class DashboardMetrics:
    timestamp: float
    tokens_scanned: int
    signals_generated: int
    trades_executed: int
    portfolio_value: float
    total_pnl: float
    win_rate: float
    avg_execution_time: float
    active_positions: int
    cpu_usage: float
    memory_usage: float
    network_io: float

@dataclass
class TradingSignal:
    token_address: str
    symbol: str
    chain: str
    price: float
    momentum_score: float
    confidence: float
    signal_type: str
    timestamp: float

@dataclass
class AlertData:
    alert_id: str
    severity: str
    message: str
    timestamp: float
    category: str
    details: Dict[str, Any]

class MetricsCollector:
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.trading_signals = deque(maxlen=500)
        self.alerts = deque(maxlen=100)
        self.performance_data = defaultdict(list)
        
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
    def add_metrics(self, metrics: DashboardMetrics):
        self.metrics_history.append(metrics)
        
        self.performance_data['timestamp'].append(metrics.timestamp)
        self.performance_data['portfolio_value'].append(metrics.portfolio_value)
        self.performance_data['total_pnl'].append(metrics.total_pnl)
        self.performance_data['tokens_scanned'].append(metrics.tokens_scanned)
        self.performance_data['cpu_usage'].append(metrics.cpu_usage)
        self.performance_data['memory_usage'].append(metrics.memory_usage)
        
        if len(self.performance_data['timestamp']) > 1000:
            for key in self.performance_data:
                self.performance_data[key] = self.performance_data[key][-1000:]
    
    def add_trading_signal(self, signal: TradingSignal):
        self.trading_signals.append(signal)
        
        if signal.confidence > 0.8:
            alert = AlertData(
                alert_id=f"signal_{int(time.time())}",
                severity="info",
                message=f"High confidence signal: {signal.symbol} ({signal.confidence:.2f})",
                timestamp=signal.timestamp,
                category="trading_signal",
                details=asdict(signal)
            )
            self.add_alert(alert)
    
    def add_alert(self, alert: AlertData):
        self.alerts.append(alert)
    
    def get_latest_metrics(self) -> Optional[DashboardMetrics]:
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_performance_chart_data(self, hours: int = 24) -> Dict:
        if not self.performance_data['timestamp']:
            return self.get_empty_chart_data()
        
        cutoff_time = time.time() - (hours * 3600)
        
        filtered_data = {}
        for key, values in self.performance_data.items():
            filtered_values = []
            for i, timestamp in enumerate(self.performance_data['timestamp']):
                if timestamp >= cutoff_time:
                    filtered_values.append(values[i])
            filtered_data[key] = filtered_values
        
        if not filtered_data['timestamp']:
            return self.get_empty_chart_data()
        
        timestamps = [datetime.fromtimestamp(ts) for ts in filtered_data['timestamp']]
        
        portfolio_chart = go.Scatter(
            x=timestamps,
            y=filtered_data['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff88', width=2)
        )
        
        pnl_chart = go.Scatter(
            x=timestamps,
            y=filtered_data['total_pnl'],
            mode='lines',
            name='Total P&L',
            line=dict(color='#ff6b6b', width=2),
            yaxis='y2'
        )
        
        tokens_chart = go.Scatter(
            x=timestamps,
            y=filtered_data['tokens_scanned'],
            mode='lines',
            name='Tokens Scanned',
            line=dict(color='#4ecdc4', width=2)
        )
        
        cpu_chart = go.Scatter(
            x=timestamps,
            y=filtered_data['cpu_usage'],
            mode='lines',
            name='CPU Usage (%)',
            line=dict(color='#ffe66d', width=2)
        )
        
        memory_chart = go.Scatter(
            x=timestamps,
            y=filtered_data['memory_usage'],
            mode='lines',
            name='Memory Usage (%)',
            line=dict(color='#ff6b9d', width=2)
        )
        
        return {
            'portfolio': json.loads(plotly.utils.PlotlyJSONEncoder().encode([portfolio_chart])),
            'pnl': json.loads(plotly.utils.PlotlyJSONEncoder().encode([pnl_chart])),
            'tokens': json.loads(plotly.utils.PlotlyJSONEncoder().encode([tokens_chart])),
            'system': json.loads(plotly.utils.PlotlyJSONEncoder().encode([cpu_chart, memory_chart]))
        }
    
    def get_empty_chart_data(self) -> Dict:
        empty_trace = go.Scatter(x=[], y=[], mode='lines')
        empty_chart = json.loads(plotly.utils.PlotlyJSONEncoder().encode([empty_trace]))
        
        return {
            'portfolio': empty_chart,
            'pnl': empty_chart,
            'tokens': empty_chart,
            'system': empty_chart
        }
    
    def get_trading_signals_data(self, limit: int = 50) -> List[Dict]:
        recent_signals = list(self.trading_signals)[-limit:]
        return [
            {
                'symbol': signal.symbol,
                'chain': signal.chain,
                'price': f"${signal.price:.6f}",
                'momentum': f"{signal.momentum_score:.3f}",
                'confidence': f"{signal.confidence:.2f}",
                'type': signal.signal_type,
                'time': datetime.fromtimestamp(signal.timestamp).strftime('%H:%M:%S')
            }
            for signal in recent_signals
        ]
    
    def get_alerts_data(self, limit: int = 20) -> List[Dict]:
        recent_alerts = list(self.alerts)[-limit:]
        return [
            {
                'id': alert.alert_id,
                'severity': alert.severity,
                'message': alert.message,
                'category': alert.category,
                'time': datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S')
            }
            for alert in recent_alerts
        ]

class SystemMonitor:
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        self.baseline_metrics = {
            'tokens_scanned': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'portfolio_value': 10.0,
            'total_pnl': 0.0,
            'active_positions': 0
        }
        
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'execution_time': 60.0,
            'error_rate': 0.1,
            'drawdown': 0.15
        }
    
    async def start_monitoring(self):
        asyncio.create_task(self.collect_system_metrics())
        asyncio.create_task(self.check_anomalies())
        asyncio.create_task(self.simulate_trading_activity())
        
        self.logger.info("System monitoring started")
    
    async def collect_system_metrics(self):
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                metrics = DashboardMetrics(
                    timestamp=time.time(),
                    tokens_scanned=self.baseline_metrics['tokens_scanned'],
                    signals_generated=self.baseline_metrics['signals_generated'],
                    trades_executed=self.baseline_metrics['trades_executed'],
                    portfolio_value=self.baseline_metrics['portfolio_value'],
                    total_pnl=self.baseline_metrics['total_pnl'],
                    win_rate=0.65,
                    avg_execution_time=25.0,
                    active_positions=self.baseline_metrics['active_positions'],
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    network_io=0.0
                )
                
                self.metrics_collector.add_metrics(metrics)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(10)
    
    async def check_anomalies(self):
        while True:
            try:
                latest_metrics = self.metrics_collector.get_latest_metrics()
                
                if latest_metrics:
                    if latest_metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
                        alert = AlertData(
                            alert_id=f"cpu_alert_{int(time.time())}",
                            severity="warning",
                            message=f"High CPU usage: {latest_metrics.cpu_usage:.1f}%",
                            timestamp=time.time(),
                            category="system",
                            details={'cpu_usage': latest_metrics.cpu_usage}
                        )
                        self.metrics_collector.add_alert(alert)
                    
                    if latest_metrics.memory_usage > self.alert_thresholds['memory_usage']:
                        alert = AlertData(
                            alert_id=f"memory_alert_{int(time.time())}",
                            severity="warning",
                            message=f"High memory usage: {latest_metrics.memory_usage:.1f}%",
                            timestamp=time.time(),
                            category="system",
                            details={'memory_usage': latest_metrics.memory_usage}
                        )
                        self.metrics_collector.add_alert(alert)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error checking anomalies: {e}")
                await asyncio.sleep(60)
    
    async def simulate_trading_activity(self):
        tokens = [
            {'symbol': 'WETH', 'address': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'},
            {'symbol': 'USDC', 'address': '0xA0b86a33E6441545C1F45DAB67F5d1C52bcfC8f4'},
            {'symbol': 'PEPE', 'address': '0x6982508145454Ce325dDbE47a25d4ec3d2311933'},
            {'symbol': 'SHIB', 'address': '0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE'},
            {'symbol': 'LINK', 'address': '0x514910771AF9Ca656af840dff83E8264EcF986CA'}
        ]
        
        while True:
            try:
                self.baseline_metrics['tokens_scanned'] += np.random.randint(50, 200)
                
                if np.random.random() > 0.7:
                    self.baseline_metrics['signals_generated'] += 1
                    
                    token = np.random.choice(tokens)
                    signal = TradingSignal(
                        token_address=token['address'],
                        symbol=token['symbol'],
                        chain='ethereum',
                        price=np.random.uniform(0.001, 10.0),
                        momentum_score=np.random.uniform(0.6, 0.95),
                        confidence=np.random.uniform(0.7, 0.98),
                        signal_type='momentum_breakout',
                        timestamp=time.time()
                    )
                    
                    self.metrics_collector.add_trading_signal(signal)
                
                if np.random.random() > 0.85:
                    self.baseline_metrics['trades_executed'] += 1
                    
                    profit_loss = np.random.uniform(-0.5, 2.0)
                    self.baseline_metrics['total_pnl'] += profit_loss
                    self.baseline_metrics['portfolio_value'] += profit_loss
                    
                    if np.random.random() > 0.5:
                        self.baseline_metrics['active_positions'] += 1
                    elif self.baseline_metrics['active_positions'] > 0:
                        self.baseline_metrics['active_positions'] -= 1
                
                await asyncio.sleep(np.random.uniform(2, 8))
                
            except Exception as e:
                self.logger.error(f"Error simulating trading activity: {e}")
                await asyncio.sleep(10)

class WebSocketManager:
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.connected_clients = set()
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connected_clients.add(websocket)
        self.logger.info(f"Client connected. Total: {len(self.connected_clients)}")
    
    def disconnect(self, websocket: WebSocket):
        self.connected_clients.discard(websocket)
        self.logger.info(f"Client disconnected. Total: {len(self.connected_clients)}")
    
    async def broadcast_data(self):
        while True:
            try:
                if self.connected_clients:
                    latest_metrics = self.metrics_collector.get_latest_metrics()
                    
                    if latest_metrics:
                        data = {
                            'type': 'metrics_update',
                            'data': {
                                'metrics': asdict(latest_metrics),
                                'charts': self.metrics_collector.get_performance_chart_data(),
                                'signals': self.metrics_collector.get_trading_signals_data(20),
                                'alerts': self.metrics_collector.get_alerts_data(10)
                            }
                        }
                        
                        dead_connections = set()
                        
                        for client in self.connected_clients:
                            try:
                                await client.send_text(json.dumps(data))
                            except Exception:
                                dead_connections.add(client)
                        
                        for dead_client in dead_connections:
                            self.disconnect(dead_client)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error broadcasting data: {e}")
                await asyncio.sleep(5)

app = FastAPI(title="Renaissance Trading Dashboard")
templates = Jinja2Templates(directory="templates")

metrics_collector = MetricsCollector()
system_monitor = SystemMonitor(metrics_collector)
websocket_manager = WebSocketManager(metrics_collector)

@app.on_event("startup")
async def startup_event():
    await system_monitor.start_monitoring()
    asyncio.create_task(websocket_manager.broadcast_data())

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

@app.get("/api/metrics")
async def get_metrics():
    latest_metrics = metrics_collector.get_latest_metrics()
    if latest_metrics:
        return asdict(latest_metrics)
    return {"error": "No metrics available"}

@app.get("/api/performance")
async def get_performance(hours: int = 24):
    return metrics_collector.get_performance_chart_data(hours)

@app.get("/api/signals")
async def get_signals(limit: int = 50):
    return metrics_collector.get_trading_signals_data(limit)

@app.get("/api/alerts")
async def get_alerts(limit: int = 20):
    return metrics_collector.get_alerts_data(limit)

@app.get("/api/system_status")
async def get_system_status():
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    return {
        "status": "operational",
        "uptime": time.time() - metrics_collector.start_time,
        "cpu_usage": cpu_usage,
        "memory_usage": memory.percent,
        "connected_clients": len(websocket_manager.connected_clients),
        "total_alerts": len(metrics_collector.alerts),
        "total_signals": len(metrics_collector.trading_signals)
    }

dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Renaissance Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            overflow-x: hidden;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            grid-template-rows: auto auto auto auto;
            gap: 20px;
            padding: 20px;
            min-height: 100vh;
        }
        
        .header {
            grid-column: 1 / -1;
            text-align: center;
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .chart-container {
            grid-column: span 2;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .full-width {
            grid-column: 1 / -1;
        }
        
        .signals-table, .alerts-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .signals-table th, .signals-table td,
        .alerts-table th, .alerts-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .signals-table th, .alerts-table th {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
        }
        
        .status-online { background: #00ff88; }
        .status-warning { background: #ffaa00; }
        .status-error { background: #ff4444; }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🚀 Renaissance Trading System Dashboard</h1>
            <p>Real-time monitoring of autonomous DeFi momentum trading</p>
            <span class="status-indicator status-online pulse"></span>
            <span id="connection-status">Connected</span>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Portfolio Value</div>
            <div class="metric-value" id="portfolio-value">$10.00</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Total P&L</div>
            <div class="metric-value" id="total-pnl">$0.00</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Tokens Scanned</div>
            <div class="metric-value" id="tokens-scanned">0</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Active Trades</div>
            <div class="metric-value" id="active-trades">0</div>
        </div>
        
        <div class="chart-container">
            <h3>Portfolio Performance</h3>
            <div id="portfolio-chart"></div>
        </div>
        
        <div class="chart-container">
            <h3>System Resources</h3>
            <div id="system-chart"></div>
        </div>
        
        <div class="chart-container full-width">
            <h3>Recent Trading Signals</h3>
            <table class="signals-table" id="signals-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Chain</th>
                        <th>Price</th>
                        <th>Momentum</th>
                        <th>Confidence</th>
                        <th>Type</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
        
        <div class="chart-container full-width">
            <h3>System Alerts</h3>
            <table class="alerts-table" id="alerts-table">
                <thead>
                    <tr>
                        <th>Severity</th>
                        <th>Message</th>
                        <th>Category</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <script>
        let ws;
        let reconnectInterval = 5000;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                document.getElementById('connection-status').textContent = 'Connected';
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'metrics_update') {
                    updateDashboard(data.data);
                }
            };
            
            ws.onclose = function() {
                document.getElementById('connection-status').textContent = 'Reconnecting...';
                console.log('WebSocket disconnected, attempting to reconnect...');
                setTimeout(connectWebSocket, reconnectInterval);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateDashboard(data) {
            const metrics = data.metrics;
            
            document.getElementById('portfolio-value').textContent = `$${metrics.portfolio_value.toFixed(2)}`;
            document.getElementById('total-pnl').textContent = `$${metrics.total_pnl.toFixed(2)}`;
            document.getElementById('tokens-scanned').textContent = metrics.tokens_scanned.toLocaleString();
            document.getElementById('active-trades').textContent = metrics.active_positions;
            
            if (data.charts && data.charts.portfolio) {
                updateChart('portfolio-chart', data.charts.portfolio);
            }
            
            if (data.charts && data.charts.system) {
                updateChart('system-chart', data.charts.system);
            }
            
            if (data.signals) {
                updateSignalsTable(data.signals);
            }
            
            if (data.alerts) {
                updateAlertsTable(data.alerts);
            }
        }
        
        function updateChart(elementId, chartData) {
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'white' },
                margin: { t: 20, r: 20, b: 40, l: 60 },
                height: 300,
                xaxis: { color: 'white', gridcolor: 'rgba(255,255,255,0.2)' },
                yaxis: { color: 'white', gridcolor: 'rgba(255,255,255,0.2)' }
            };
            
            Plotly.newPlot(elementId, chartData, layout, {responsive: true});
        }
        
        function updateSignalsTable(signals) {
            const tbody = document.querySelector('#signals-table tbody');
            tbody.innerHTML = '';
            
            signals.forEach(signal => {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${signal.symbol}</td>
                    <td>${signal.chain}</td>
                    <td>${signal.price}</td>
                    <td>${signal.momentum}</td>
                    <td>${signal.confidence}</td>
                    <td>${signal.type}</td>
                    <td>${signal.time}</td>
                `;
            });
        }
        
        function updateAlertsTable(alerts) {
            const tbody = document.querySelector('#alerts-table tbody');
            tbody.innerHTML = '';
            
            alerts.forEach(alert => {
                const row = tbody.insertRow();
                const severityClass = alert.severity === 'warning' ? 'status-warning' : 
                                   alert.severity === 'error' ? 'status-error' : 'status-online';
                
                row.innerHTML = `
                    <td><span class="status-indicator ${severityClass}"></span>${alert.severity}