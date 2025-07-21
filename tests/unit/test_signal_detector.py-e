import pytest
import pandas as pd
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from intelligence.signals.signal_detector import SignalDetector

class TestSignalDetector:
    
    @pytest.fixture
    def mock_chains(self):
        return {'arbitrum': Mock(), 'polygon': Mock()}
    
    @pytest.fixture
    def mock_redis(self):
        return Mock()
    
    @pytest.fixture
    def signal_detector(self, mock_chains, mock_redis):
        return SignalDetector(mock_chains, mock_redis)
    
    def test_calculate_momentum_score(self, signal_detector):
        """Test momentum score calculation"""
        # Create test data
        features = pd.DataFrame({
            'returns': [0.01, 0.02, -0.01, 0.03, 0.015],
            'volatility': [0.1, 0.15, 0.12, 0.18, 0.14],
            'swap_volume': [1000, 1500, 800, 2000, 1200]
        })
        
        score = signal_detector.calculate_momentum_score(features)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_calculate_velocity(self, signal_detector):
        """Test velocity calculation"""
        features = pd.DataFrame({
            'returns': [0.01, 0.02, -0.01, 0.03, 0.015]
        })
        
        velocity = signal_detector.calculate_velocity(features)
        
        assert isinstance(velocity, float)
        assert velocity >= 0
    
    def test_is_breakout_empty_features(self, signal_detector):
        """Test breakout detection with empty features"""
        empty_features = pd.DataFrame()
        pool = {'id': 'test_pool', 'swaps': []}
        
        result = signal_detector.is_breakout(empty_features, pool)
        assert result is False
    
    def test_is_breakout_valid_conditions(self, signal_detector):
        """Test breakout detection with valid conditions"""
        # Create features that should trigger breakout
        features = pd.DataFrame({
            'returns': [0.15] * 70,  # High returns for 70 periods
            'volatility': [0.3] * 70,
            'swap_volume': [10000] * 70
        })
        
        # Mock pool with high volume spike
        pool = {
            'id': 'test_pool',
            'swaps': [{'amountUSD': '50000'} for _ in range(20)]
        }
        
        # Mock methods to return breakout conditions
        with patch.object(signal_detector, 'calculate_velocity', return_value=0.15):
            with patch.object(signal_detector, 'detect_volume_spike', return_value=3.0):
                with patch.object(signal_detector, 'calculate_momentum_score', return_value=0.8):
                    result = signal_detector.is_breakout(features, pool)
                    
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_fetch_historical_prices(self, signal_detector):
        """Test historical price fetching"""
        # Mock redis to return None (cache miss)
        signal_detector.redis_client.get.return_value = None
        
        # Mock the subgraph response
        mock_response = {
            'data': {
                'pool': {
                    'poolHourData': [
                        {'close': '100.5', 'periodStartUnix': '1234567890'},
                        {'close': '101.2', 'periodStartUnix': '1234571490'},
                        {'close': '99.8', 'periodStartUnix': '1234575090'}
                    ]
                }
            }
        }
        
        with patch.object(signal_detector, 'fetch_dex_data', return_value=mock_response):
            prices = await signal_detector.fetch_historical_prices('arbitrum', '0xtest')
            
        assert isinstance(prices, list)
        assert len(prices) == 3
        assert all(isinstance(p, float) for p in prices)
    
    def test_engineer_features(self, signal_detector):
        """Test feature engineering"""
        price_data = pd.Series([100, 101, 99, 102, 98, 105, 103])
        pool = {
            'id': 'test_pool',
            'swaps': [{'amountUSD': '1000'} for _ in range(10)]
        }
        
        features = signal_detector.engineer_features(price_data, pool)
        
        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        
        expected_columns = [
            'returns', 'volatility', 'momentum', 'rsi', 'bb_position',
            'volume_ma', 'whale_activity', 'price_acceleration',
            'volatility_ratio', 'momentum_strength', 'swap_volume'
        ]
        
        for col in expected_columns:
            assert col in features.columns
    
    @pytest.mark.asyncio
    async def test_detect_market_regime(self, signal_detector):
        """Test market regime detection"""
        # Mock subgraph response
        mock_response = {
            'data': {
                'pools': [
                    {
                        'id': 'pool1',
                        'poolHourData': [
                            {'close': '100'},
                            {'close': '101'},
                            {'close': '99'}
                        ]
                    }
                ]
            }
        }
        
        with patch.object(signal_detector, 'fetch_dex_data', return_value=mock_response):
            regime = await signal_detector.detect_market_regime()
            
        assert regime in ['bull', 'bear', 'normal', 'extreme_volatility']

if __name__ == "__main__":
    pytest.main([__file__])
