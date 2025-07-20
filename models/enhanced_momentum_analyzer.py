from config.dynamic_settings import dynamic_settings

# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

#!/usr/bin/env python3
"""
ENHANCED MOMENTUM ANALYZER
Advanced signal processing for Renaissance-level performance
"""

import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time
import logging

@dataclass
class EnhancedSignal:
    address: str
    chain: str
    dex: str
    
    # Core metrics
    price: float
    momentum_score: float
    confidence: float
    
    # Advanced metrics
    velocity: float
    acceleration: float
    volume_profile: float
    liquidity_stability: float
    breakout_strength: float
    
    # Risk metrics
    volatility: float
    max_drawdown_risk: float
    honeypot_risk: float
    
    # Timing
    optimal_entry_time: float
    expected_hold_duration: int
    exit_conditions: Dict[str, float]
    
    # Metadata
    detected_at: float
    signal_quality: str  # 'excellent', 'good', 'fair'

class EnhancedMomentumAnalyzer:
    def __init__(self):
        self.price_history = {}
        self.volume_history = {}
        self.analysis_cache = {}
        
        # Advanced parameters
        self.momentum_windows = [5, 10, 20, 50]  # Multiple timeframes
        self.volatility_threshold = get_dynamic_config()["volatility_threshold"]
        self.min_liquidity = dynamic_settings.get_trading_params()["liquidity_threshold"]
        
        self.logger = logging.getLogger(__name__)
        
    async def analyze_enhanced_momentum(self, token_data: Dict, 
                                      price_history: List[float], 
                                      volume_history: List[float]) -> Optional[EnhancedSignal]:
        """
        Enhanced momentum analysis with multiple timeframes and risk assessment
        """
        try:
            if len(price_history) < 10:
                return None
                
            address = token_data['address']
            chain = token_data['chain']
            
            # Convert to numpy arrays for analysis
            prices = np.array(price_history)
            volumes = np.array(volume_history) if volume_history else np.ones_like(prices)
            
            # Multi-timeframe momentum analysis
            momentum_scores = {}
            for window in self.momentum_windows:
                if len(prices) >= window:
                    momentum_scores[window] = self.calculate_windowed_momentum(prices, volumes, window)
            
            if not momentum_scores:
                return None
                
            # Advanced metrics calculation
            velocity = self.calculate_velocity(prices)
            acceleration = self.calculate_acceleration(prices)
            volume_profile = self.calculate_volume_profile(volumes)
            liquidity_stability = self.calculate_liquidity_stability(token_data)
            breakout_strength = self.calculate_breakout_strength(prices, volumes)
            
            # Risk assessment
            volatility = self.calculate_volatility(prices)
            max_drawdown_risk = self.calculate_max_drawdown_risk(prices)
            honeypot_risk = await self.assess_honeypot_risk(token_data)
            
            # Overall momentum score (weighted by timeframe)
            weights = {5: 0.4, 10: 0.3, 20: 0.2, 50: 0.1}
            overall_momentum = sum(
                momentum_scores.get(window, 0) * weight 
                for window, weight in weights.items()
            )
            
            # Confidence calculation
            confidence = self.calculate_enhanced_confidence(
                momentum_scores, volatility, liquidity_stability, len(prices)
            )
            
            # Signal quality assessment
            signal_quality = self.assess_signal_quality(
                overall_momentum, confidence, volatility, breakout_strength
            )
            
            # Exit conditions
            exit_conditions = self.calculate_exit_conditions(
                overall_momentum, volatility, breakout_strength
            )
            
            # Only generate signal if passes enhanced criteria
            if (overall_momentum > 0.6 and 
                confidence > 0.7 and 
                breakout_strength > 0.5 and
                volatility < self.volatility_threshold and
                honeypot_risk < 0.3):
                
                return EnhancedSignal(
                    address=address,
                    chain=chain,
                    dex=token_data.get('dex', 'unknown'),
                    price=float(prices[-1]),
                    momentum_score=float(overall_momentum),
                    confidence=float(confidence),
                    velocity=float(velocity),
                    acceleration=float(acceleration),
                    volume_profile=float(volume_profile),
                    liquidity_stability=float(liquidity_stability),
                    breakout_strength=float(breakout_strength),
                    volatility=float(volatility),
                    max_drawdown_risk=float(max_drawdown_risk),
                    honeypot_risk=float(honeypot_risk),
                    optimal_entry_time=time.time(),
                    expected_hold_duration=self.calculate_optimal_hold_time(overall_momentum, volatility),
                    exit_conditions=exit_conditions,
                    detected_at=time.time(),
                    signal_quality=signal_quality
                )
                
        except Exception as e:
            self.logger.error(f"Enhanced momentum analysis error: {e}")
            
        return None
        
    def calculate_windowed_momentum(self, prices: np.ndarray, volumes: np.ndarray, window: int) -> float:
        """Calculate momentum for specific window"""
        if len(prices) < window:
            return 0.0
            
        # Price momentum
        price_change = (prices[-1] - prices[-window]) / (prices[-window] + 1e-10)
        price_momentum = np.tanh(price_change * 5)
        
        # Volume momentum
        vol_recent = np.mean(volumes[-window//2:])
        vol_historical = np.mean(volumes[-window:-window//2])
        volume_momentum = min(vol_recent / (vol_historical + 1e-6), 2.0) - 1.0
        
        # Combine
        return (price_momentum * 0.7 + volume_momentum * 0.3)
        
    def calculate_velocity(self, prices: np.ndarray) -> float:
        """Calculate price velocity (rate of change)"""
        if len(prices) < 3:
            return 0.0
        return float(np.mean(np.diff(prices[-5:])))
        
    def calculate_acceleration(self, prices: np.ndarray) -> float:
        """Calculate price acceleration (rate of velocity change)"""
        if len(prices) < 5:
            return 0.0
        velocity = np.diff(prices)
        acceleration = np.diff(velocity)
        return float(np.mean(acceleration[-3:]))
        
    def calculate_volume_profile(self, volumes: np.ndarray) -> float:
        """Calculate volume profile strength"""
        if len(volumes) < 5:
            return 0.5
            
        recent_volume = np.mean(volumes[-3:])
        historical_volume = np.mean(volumes[:-3])
        
        return min(recent_volume / (historical_volume + 1e-6), 3.0) / 3.0
        
    def calculate_liquidity_stability(self, token_data: Dict) -> float:
        """Calculate liquidity stability score"""
        liquidity = token_data.get('liquidity_usd', 0)
        
        if liquidity > 500000:
            return 0.9
        elif liquidity > 100000:
            return 0.7
        elif liquidity > 50000:
            return 0.5
        else:
            return 0.2
            
    def calculate_breakout_strength(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate breakout strength using technical analysis"""
        if len(prices) < 20:
            return 0.0
            
        # Calculate support/resistance levels
        recent_high = np.max(prices[-10:])
        historical_high = np.max(prices[-20:-10])
        
        # Breakout strength
        if recent_high > historical_high:
            breakout_magnitude = (recent_high - historical_high) / (historical_high + 1e-10)
            
            # Volume confirmation
            volume_confirmation = self.calculate_volume_profile(volumes)
            
            return min(breakout_magnitude * volume_confirmation, 1.0)
        
        return 0.0
        
    def calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate price volatility"""
        if len(prices) < 5:
            return 0.0
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        return float(np.std(returns))
        
    def calculate_max_drawdown_risk(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown risk"""
        if len(prices) < 5:
            return 0.5
            
        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)
        drawdown = (running_max - prices) / (running_max + 1e-10)
        max_drawdown = np.max(drawdown)
        
        return float(min(max_drawdown, 1.0))
        
    async def assess_honeypot_risk(self, token_data: Dict) -> float:
        """Assess honeypot risk (placeholder for real implementation)"""
        # This would integrate with honeypot detection services
        # For now, return low risk for tokens with good liquidity
        liquidity = token_data.get('liquidity_usd', 0)
        if liquidity > 100000:
            return 0.1
        else:
            return 0.5
            
    def calculate_enhanced_confidence(self, momentum_scores: Dict, volatility: float, 
                                   liquidity_stability: float, data_points: int) -> float:
        """Calculate enhanced confidence score"""
        factors = []
        
        # Data quality factor
        if data_points >= 50:
            factors.append(0.9)
        elif data_points >= 20:
            factors.append(0.7)
        else:
            factors.append(0.5)
            
        # Momentum consistency across timeframes
        if len(momentum_scores) >= 3:
            momentum_values = list(momentum_scores.values())
            momentum_consistency = 1.0 - np.std(momentum_values)
            factors.append(max(momentum_consistency, 0.0))
        else:
            factors.append(0.5)
            
        # Volatility factor (lower volatility = higher confidence)
        volatility_factor = max(0.0, 1.0 - volatility * 5)
        factors.append(volatility_factor)
        
        # Liquidity factor
        factors.append(liquidity_stability)
        
        return float(np.mean(factors))
        
    def assess_signal_quality(self, momentum: float, confidence: float, 
                            volatility: float, breakout_strength: float) -> str:
        """Assess overall signal quality"""
        score = (momentum * 0.3 + confidence * 0.3 + 
                breakout_strength * 0.3 + (1 - volatility) * 0.1)
        
        if score > 0.8:
            return 'excellent'
        elif score > 0.6:
            return 'good'
        else:
            return 'fair'
            
    def calculate_exit_conditions(self, momentum: float, volatility: float, 
                                breakout_strength: float) -> Dict[str, float]:
        """Calculate dynamic exit conditions"""
        base_profit_target = 0.15  # 15% profit target
        base_stop_loss = 0.05      # 5% stop loss
        
        # Adjust based on momentum and volatility
        profit_multiplier = 1.0 + (momentum - 0.5) * 0.5
        volatility_adjustment = min(volatility * 2, 0.5)
        
        return {
            'profit_target': base_profit_target * profit_multiplier,
            'stop_loss': base_stop_loss + volatility_adjustment,
            'momentum_exit': momentum * 0.7,  # Exit if momentum drops below this
            'time_exit': 300,  # 5 minutes max hold time
            'breakout_failure': breakout_strength * 0.5
        }
        
    def calculate_optimal_hold_time(self, momentum: float, volatility: float) -> int:
        """Calculate optimal hold time in seconds"""
        base_time = 180  # 3 minutes
        
        # Higher momentum = longer hold
        momentum_adjustment = (momentum - 0.5) * 120
        
        # Higher volatility = shorter hold
        volatility_adjustment = -volatility * 60
        
        optimal_time = base_time + momentum_adjustment + volatility_adjustment
        
        return int(np.clip(optimal_time, 60, 600))  # 1-10 minutes

# Global analyzer instance
enhanced_analyzer = EnhancedMomentumAnalyzer()
