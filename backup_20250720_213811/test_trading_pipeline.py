import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backup_20250720_213811.test_signal_detector import backup_20250720_213811.signal_detector
from core.models.inference_model import backup_20250720_213811.inference_model
from core.execution.trade_executor import backup_20250720_213811.trade_executor
from core.execution.risk_manager import backup_20250720_213811.risk_manager
from security.validators.safety_checks import backup_20250720_213811.safety_checks

class TestTradingPipeline:
    
    @pytest.fixture
    def mock_chains(self):
        mock_w3 = Mock()
        mock_w3.is_connected.return_value = True
        mock_w3.eth.block_number = 12345678
        mock_w3.eth.get_balance.return_value = 1000000000000000000  # 1 ETH
        return {
            'arbitrum': mock_w3,
            'polygon': mock_w3
        }
    
    @pytest.fixture
    def components(self, mock_chains):
        redis_mock = Mock()
        
        signal_detector = SignalDetector(mock_chains, redis_mock)
        model = MomentumEnsemble()
        trade_executor = TradeExecutor(mock_chains)
        risk_manager = RiskManager()
        safety_checker = SafetyChecker(mock_chains)
        
        return {
            'signal_detector': signal_detector,
            'model': model,
            'trade_executor': trade_executor,
            'risk_manager': risk_manager,
            'safety_checker': safety_checker
        }
    
    @pytest.mark.asyncio
    async def test_full_pipeline_safe_token(self, components):
        """Test complete pipeline with safe token"""
        
        # Mock token data
        token_data = {
            'address': '0x1234567890123456789012345678901234567890',
            'symbol': 'TEST',
            'data': pd.DataFrame({
                'returns': np.random.normal(0.1, 0.02, 50),
                'volatility': np.random.uniform(0.2, 0.4, 50),
                'momentum': np.random.normal(0.05, 0.01, 50),
                'rsi': np.random.uniform(40, 80, 50),
                'bb_position': np.random.uniform(0.2, 0.8, 50),
                'volume_ma': np.random.uniform(5000, 15000, 50),
                'whale_activity': np.random.uniform(0, 0.1, 50),
                'price_acceleration': np.random.normal(0, 0.001, 50),
                'volatility_ratio': np.random.uniform(0.8, 1.2, 50),
                'momentum_strength': np.random.uniform(0.05, 0.15, 50),
                'swap_volume': np.random.uniform(5000, 15000, 50)
            }),
            'velocity': 0.15,
            'volume_spike': 3.0,
            'momentum_score': 0.8
        }
        
        # Mock safety checks to pass
        with patch.object(components['safety_checker'], 'check_token', return_value=True):
            
            # Mock model prediction
            with patch.object(components['model'], 'predict', return_value=0.85):
                
                # Mock risk checks to pass
                with patch.object(components['risk_manager'], 'calculate_position_size', return_value=0.001):
                    with patch.object(components['risk_manager'], 'check_portfolio_exposure', return_value=True):
                        with patch.object(components['trade_executor'], 'check_wallet_balance', return_value=True):
                            with patch.object(components['risk_manager'], 'check_gas_budget', return_value=True):
                                
                                # Mock trade execution
                                mock_tx_hash = b'\\x12\\x34\\x56\\x78' * 8  # 32 bytes
                                with patch.object(components['trade_executor'], 'execute_trade', return_value=mock_tx_hash):
                                    
                                    # Run pipeline
                                    result = await self.run_pipeline_step(components, 'arbitrum', token_data)
                                    
                                    # Assertions
                                    assert result['success'] is True
                                    assert result['trade_executed'] is True
                                    assert 'tx_hash' in result
                                    assert result['prediction'] == 0.85
    
    @pytest.mark.asyncio
    async def test_full_pipeline_unsafe_token(self, components):
        """Test complete pipeline with unsafe token"""
        
        token_data = {
            'address': '0xdangerous1234567890123456789012345678901234567890',
            'symbol': 'SCAM',
            'data': pd.DataFrame({
                'returns': [0.1] * 50,
                'volatility': [0.3] * 50,
                'momentum': [0.05] * 50,
                'rsi': [50] * 50,
                'bb_position': [0.5] * 50,
                'volume_ma': [10000] * 50,
                'whale_activity': [0.05] * 50,
                'price_acceleration': [0] * 50,
                'volatility_ratio': [1.0] * 50,
                'momentum_strength': [0.1] * 50,
                'swap_volume': [10000] * 50
            })
        }
        
        # Mock safety checks to fail
        with patch.object(components['safety_checker'], 'check_token', return_value=False):
            
            result = await self.run_pipeline_step(components, 'arbitrum', token_data)
            
            # Should not execute trade
            assert result['success'] is False
            assert result['reason'] == 'safety_failed'
    
    async def run_pipeline_step(self, components, chain, token_data):
        """Simulate single pipeline step"""
        try:
            # Safety checks
            is_safe = await components['safety_checker'].check_token(chain, token_data['address'])
            
            if not is_safe:
                return {'success': False, 'reason': 'safety_failed'}
            
            # Model prediction
            features_df = token_data['data']
            prediction = components['model'].predict(features_df)
            
            # Trading decision
            should_trade = (
                prediction > 0.75 and
                token_data.get('velocity', 0) >= 0.13 and
                token_data.get('volume_spike', 1) >= 2.5
            )
            
            if should_trade:
                # Risk management
                position_size = components['risk_manager'].calculate_position_size(features_df, chain)
                
                portfolio_check = components['risk_manager'].check_portfolio_exposure(chain, position_size)
                balance_check = await components['trade_executor'].check_wallet_balance(chain, position_size)
                gas_check = components['risk_manager'].check_gas_budget(chain, 0.001)
                
                if portfolio_check and balance_check and gas_check:
                    # Execute trade
                    tx_hash = await components['trade_executor'].execute_trade(
                        chain, token_data['address'], prediction, position_size
                    )
                    
                    if tx_hash:
                        return {
                            'success': True,
                            'trade_executed': True,
                            'tx_hash': tx_hash.hex(),
                            'prediction': prediction
                        }
                    else:
                        return {'success': False, 'reason': 'execution_failed'}
                else:
                    return {'success': False, 'reason': 'risk_limits'}
            else:
                return {'success': True, 'trade_executed': False, 'prediction': prediction}
                
        except Exception as e:
            return {'success': False, 'reason': 'processing_error', 'error': str(e)}

if __name__ == "__main__":
    pytest.main([__file__])
