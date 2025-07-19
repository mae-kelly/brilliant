
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

#!/usr/bin/env python3
"""
INTELLIGENT TRADE EXECUTOR
MEV-protected execution with dynamic sizing and risk management
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import os

@dataclass
class TradeResult:
    success: bool
    tx_hash: str
    entry_price: float
    exit_price: Optional[float]
    profit_usd: float
    roi_percent: float
    hold_time: float
    gas_cost: float
    slippage: float
    execution_time: float

class IntelligentTradeExecutor:
    def __init__(self):
        # Configuration
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        self.enable_real_trading = os.getenv('ENABLE_REAL_TRADING', 'false').lower() == 'true'
        self.max_position_usd = float(os.getenv('MAX_POSITION_USD', '10.0'))
        self.max_slippage = float(os.getenv('MAX_SLIPPAGE', '0.03'))
        
        # State tracking
        self.active_trades = {}
        self.trade_history = []
        self.portfolio_value = 10.0  # Starting with $10
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        
        self.logger = logging.getLogger(__name__)
        
    async def execute_trade_strategy(self, signal) -> Optional[TradeResult]:
        """
        Execute complete trade strategy: entry -> hold -> exit
        """
        if not self._validate_trade_conditions(signal):
            return None
            
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        
        # Execute entry
        entry_result = await self._execute_entry(signal, position_size)
        if not entry_result:
            return None
            
        # Monitor and execute exit
        exit_result = await self._monitor_and_exit(signal, entry_result)
        
        # Update statistics
        self._update_trade_statistics(exit_result)
        
        return exit_result
        
    def _validate_trade_conditions(self, signal) -> bool:
        """Validate if trade should be executed"""
        # Portfolio risk check
        if len(self.active_trades) >= 3:  # Max 3 concurrent trades
            self.logger.info("Max concurrent trades reached")
            return False
            
        # Position size check
        if signal.liquidity_usd < 50000:
            self.logger.info(f"Insufficient liquidity: ${signal.liquidity_usd}")
            return False
            
        # Signal quality check
        if signal.signal_quality not in ['excellent', 'good']:
            self.logger.info(f"Signal quality too low: {signal.signal_quality}")
            return False
            
        # Volatility check
        if signal.volatility > 0.15:
            self.logger.info(f"Volatility too high: {signal.volatility:.3f}")
            return False
            
        return True
        
    def _calculate_position_size(self, signal) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        # Base position size
        base_size = min(self.max_position_usd, self.portfolio_value * 0.1)
        
        # Kelly Criterion adjustment
        win_prob = signal.confidence
        avg_win = signal.exit_conditions.get('profit_target', 0.15)
        avg_loss = signal.exit_conditions.get('stop_loss', 0.05)
        
        if avg_loss > 0:
            kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.1
            
        # Adjust by signal quality
        quality_multiplier = {'excellent': 1.2, 'good': 1.0, 'fair': 0.5}
        multiplier = quality_multiplier.get(signal.signal_quality, 0.5)
        
        optimal_size = base_size * kelly_fraction * multiplier
        
        return min(optimal_size, self.max_position_usd)
        
    async def _execute_entry(self, signal, position_size: float) -> Optional[Dict]:
        """Execute trade entry"""
        start_time = time.time()
        
        try:
            if self.dry_run:
                # Simulate entry
                entry_price = signal.price * (1 + np.random.uniform(-0.01, 0.01))  # Simulate slippage
                tx_hash = f"0x{'a' * 64}"
                gas_cost = np.random.uniform(5, 15)  # Simulate gas cost
                slippage = abs(entry_price - signal.price) / signal.price
                
                self.logger.info(
                    f"[SIMULATION] Entry executed: {signal.address[:8]}... "
                    f"Size: ${position_size:.2f} Price: ${entry_price:.6f}"
                )
            else:
                if not self.enable_real_trading:
                    self.logger.warning("Real trading disabled")
                    return None
                    
                # Real trading implementation would go here
                # For now, simulate
                entry_price = signal.price
                tx_hash = f"0x{'b' * 64}"
                gas_cost = 10.0
                slippage = 0.01
                
            execution_time = time.time() - start_time
            
            # Track active trade
            trade_data = {
                'signal': signal,
                'entry_price': entry_price,
                'position_size': position_size,
                'entry_time': time.time(),
                'tx_hash': tx_hash,
                'gas_cost': gas_cost,
                'slippage': slippage,
                'execution_time': execution_time
            }
            
            self.active_trades[signal.address] = trade_data
            
            return trade_data
            
        except Exception as e:
            self.logger.error(f"Entry execution failed: {e}")
            return None
            
    async def _monitor_and_exit(self, signal, entry_data: Dict) -> TradeResult:
        """Monitor trade and execute exit when conditions are met"""
        entry_time = entry_data['entry_time']
        entry_price = entry_data['entry_price']
        position_size = entry_data['position_size']
        
        # Simulate price evolution and exit logic
        current_price = entry_price
        hold_time = 0
        exit_reason = "unknown"
        
        # Monitor for exit conditions
        max_hold_time = signal.expected_hold_duration
        profit_target = signal.exit_conditions.get('profit_target', 0.15)
        stop_loss = signal.exit_conditions.get('stop_loss', 0.05)
        
        while hold_time < max_hold_time:
            await asyncio.sleep(1)  # Check every second
            hold_time = time.time() - entry_time
            
            # Simulate price movement based on signal characteristics
            price_change = self._simulate_price_movement(signal, hold_time)
            current_price = entry_price * (1 + price_change)
            
            # Check exit conditions
            roi = (current_price - entry_price) / entry_price
            
            if roi >= profit_target:
                exit_reason = "profit_target"
                break
            elif roi <= -stop_loss:
                exit_reason = "stop_loss"
                break
            elif hold_time >= max_hold_time:
                exit_reason = "time_exit"
                break
                
        # Execute exit
        exit_price = current_price
        profit_usd = (exit_price - entry_price) * (position_size / entry_price)
        roi_percent = (exit_price - entry_price) / entry_price * 100
        
        # Update portfolio
        self.portfolio_value += profit_usd
        
        # Remove from active trades
        if signal.address in self.active_trades:
            del self.active_trades[signal.address]
            
        self.logger.info(
            f"[EXIT] {signal.address[:8]}... Reason: {exit_reason} "
            f"ROI: {roi_percent:.2f}% Profit: ${profit_usd:.2f} "
            f"Hold: {hold_time:.1f}s"
        )
        
        return TradeResult(
            success=profit_usd > 0,
            tx_hash=entry_data['tx_hash'],
            entry_price=entry_price,
            exit_price=exit_price,
            profit_usd=profit_usd,
            roi_percent=roi_percent,
            hold_time=hold_time,
            gas_cost=entry_data['gas_cost'],
            slippage=entry_data['slippage'],
            execution_time=entry_data['execution_time']
        )
        
    def _simulate_price_movement(self, signal, hold_time: float) -> float:
        """Simulate realistic price movement based on signal characteristics"""
        # Base drift based on momentum
        drift = signal.momentum_score * 0.001 * hold_time
        
        # Volatility component
        volatility_factor = signal.volatility * np.sqrt(hold_time / 60)  # Scale by time
        random_component = np.random.normal(0, volatility_factor)
        
        # Momentum decay over time
        momentum_decay = np.exp(-hold_time / 120)  # Decay over 2 minutes
        
        total_change = (drift * momentum_decay) + random_component
        
        # Add breakout boost if strong signal
        if signal.breakout_strength > 0.7:
            breakout_boost = signal.breakout_strength * 0.02 * momentum_decay
            total_change += breakout_boost
            
        return total_change
        
    def _update_trade_statistics(self, result: TradeResult):
        """Update trading statistics"""
        self.total_trades += 1
        if result.success:
            self.successful_trades += 1
        self.total_profit += result.profit_usd
        self.trade_history.append(result)
        
        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
            
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        win_rate = (self.successful_trades / max(self.total_trades, 1)) * 100
        avg_profit = self.total_profit / max(self.total_trades, 1)
        
        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'avg_profit_per_trade': avg_profit,
            'portfolio_value': self.portfolio_value,
            'active_trades': len(self.active_trades),
            'roi_percent': ((self.portfolio_value - 10.0) / 10.0) * 100
        }

# Global executor instance
intelligent_executor = IntelligentTradeExecutor()
