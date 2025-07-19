#!/usr/bin/env python3
"""
PERFORMANCE TRACKING FOR DYNAMIC OPTIMIZATION
Integrates with all trading modules to track performance
"""

import time
import threading
from typing import Dict, List
from collections import deque
import numpy as np
from dataclasses import dataclass

@dataclass
class TradeRecord:
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    size: float
    roi: float
    token: str

class PerformanceTracker:
    def __init__(self):
        self.trades = deque(maxlen=1000)
        self.lock = threading.Lock()
        
    def record_trade(self, entry_time: float, exit_time: float,
                    entry_price: float, exit_price: float,
                    size: float, token: str):
        """Record a completed trade"""
        roi = (exit_price - entry_price) / entry_price
        
        trade = TradeRecord(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            roi=roi,
            token=token
        )
        
        with self.lock:
            self.trades.append(trade)
            
        # Update optimizer every 10 trades
        if len(self.trades) % 10 == 0:
            self.update_optimizer()
    
    def update_optimizer(self):
        """Update parameter optimizer with recent performance"""
        if len(self.trades) < 10:
            return
            
        recent_trades = list(self.trades)[-50:]  # Last 50 trades
        
        # Calculate metrics
        rois = [t.roi for t in recent_trades]
        roi_mean = np.mean(rois)
        roi_std = np.std(rois)
        
        win_rate = sum(1 for roi in rois if roi > 0) / len(rois)
        sharpe_ratio = roi_mean / (roi_std + 1e-8) * np.sqrt(365 * 24)  # Annualized
        
        # Calculate max drawdown
        cumulative = np.cumprod([1 + roi for roi in rois])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdown)
        
        # Update optimizer
        from config.dynamic_parameters import update_performance
        update_performance(roi_mean, win_rate, sharpe_ratio, max_drawdown, len(recent_trades))

# Global tracker
performance_tracker = PerformanceTracker()
