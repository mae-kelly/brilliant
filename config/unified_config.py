import os
import json
import yaml
import time
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

@dataclass
class TradingConfig:
    confidence_threshold: float = 0.75
    momentum_threshold: float = 0.65
    volatility_threshold: float = 0.10
    liquidity_threshold: float = 50000
    min_liquidity_threshold: float = 10000
    max_risk_score: float = 0.4
    honeypot_risk_threshold: float = 0.3
    max_slippage: float = 0.03
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.12
    max_hold_time: float = 300
    min_hold_time: float = 30
    min_price_change: float = 9
    max_price_change: float = 15
    price_momentum_decay: float = 0.95
    max_position_size: float = 10.0
    starting_capital: float = 10.0
    kelly_multiplier: float = 0.25
    max_correlation: float = 0.6
    order_flow_threshold: float = 0.3
    microstructure_noise_limit: float = 0.1
    jump_intensity_threshold: float = 0.2
    sentiment_threshold: float = 0.6
    social_momentum_weight: float = 0.2
    whale_threshold: float = 100000
    max_gas_price: float = 50
    mev_protection_threshold: float = 0.01
    flashbots_threshold: float = 1.0
    regime_change_threshold: float = 0.7
    volatility_regime_threshold: float = 0.15
    trend_regime_threshold: float = 0.05
    sharpe_target: float = 2.0
    max_drawdown_limit: float = 0.15
    win_rate_target: float = 0.6
    roi_target: float = 0.15

@dataclass
class ChainConfig:
    rpc_url: str
    chain_id: int
    weth_address: str
    usdc_address: str
    gas_multiplier: float = 1.1
    block_time: float = 2.0
    max_gas_limit: int = 500000

@dataclass
class SecurityConfig:
    enable_real_trading: bool = False
    private_key: Optional[str] = None
    alchemy_api_key: Optional[str] = None
    etherscan_api_key: Optional[str] = None
    wallet_address: Optional[str] = None
    max_daily_trades: int = 1000
    emergency_stop_enabled: bool = True
    dry_run_mode: bool = True

class UnifiedConfig:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.trading = TradingConfig()
        self.security = SecurityConfig()
        self.chains = self._initialize_chains()
        self._load_config()
        self._load_environment()
        self._validate_config()
        
        self.logger = logging.getLogger(__name__)

    def _initialize_chains(self) -> Dict[str, ChainConfig]:
        return {
            'ethereum': ChainConfig(
                rpc_url=f"https://eth-mainnet.g.alchemy.com/v2/{self._get_env('ALCHEMY_API_KEY', 'demo')}",
                chain_id=1,
                weth_address='0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
                usdc_address='0xA0b86a33E6441545C1F45DAB67F5d1C52bcfC8f4',
                gas_multiplier=1.2,
                block_time=12.0,
                max_gas_limit=500000
            ),
            'arbitrum': ChainConfig(
                rpc_url=f"https://arb-mainnet.g.alchemy.com/v2/{self._get_env('ALCHEMY_API_KEY', 'demo')}",
                chain_id=42161,
                weth_address='0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
                usdc_address='0xaf88d065e77c8cC2239327C5EDb3A432268e5831',
                gas_multiplier=1.1,
                block_time=0.25,
                max_gas_limit=10000000
            ),
            'polygon': ChainConfig(
                rpc_url=f"https://polygon-mainnet.g.alchemy.com/v2/{self._get_env('ALCHEMY_API_KEY', 'demo')}",
                chain_id=137,
                weth_address='0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
                usdc_address='0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
                gas_multiplier=1.3,
                block_time=2.0,
                max_gas_limit=2000000
            ),
            'optimism': ChainConfig(
                rpc_url=f"https://opt-mainnet.g.alchemy.com/v2/{self._get_env('ALCHEMY_API_KEY', 'demo')}",
                chain_id=10,
                weth_address='0x4200000000000000000000000000000000000006',
                usdc_address='0x7F5c764cBc14f9669B88837ca1490cCa17c31607',
                gas_multiplier=1.1,
                block_time=2.0,
                max_gas_limit=1000000
            )
        }

    def _get_env(self, key: str, default: Any = None) -> Any:
        return os.getenv(key, default)

    def _load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                if 'trading' in config_data:
                    self._update_dataclass(self.trading, config_data['trading'])
                
                if 'security' in config_data:
                    self._update_dataclass(self.security, config_data['security'])
                    
            except Exception as e:
                self.logger.warning(f"Failed to load config file {self.config_path}: {e}")

    def _load_environment(self):
        env_mappings = {
            'ENABLE_REAL_TRADING': ('security', 'enable_real_trading', lambda x: x.lower() == 'true'),
            'PRIVATE_KEY': ('security', 'private_key', str),
            'ALCHEMY_API_KEY': ('security', 'alchemy_api_key', str),
            'ETHERSCAN_API_KEY': ('security', 'etherscan_api_key', str),
            'WALLET_ADDRESS': ('security', 'wallet_address', str),
            'STARTING_CAPITAL': ('trading', 'starting_capital', float),
            'MAX_POSITION_SIZE': ('trading', 'max_position_size', float),
            'CONFIDENCE_THRESHOLD': ('trading', 'confidence_threshold', float),
            'MOMENTUM_THRESHOLD': ('trading', 'momentum_threshold', float),
            'VOLATILITY_THRESHOLD': ('trading', 'volatility_threshold', float),
            'LIQUIDITY_THRESHOLD': ('trading', 'liquidity_threshold', float),
            'STOP_LOSS_THRESHOLD': ('trading', 'stop_loss_threshold', float),
            'TAKE_PROFIT_THRESHOLD': ('trading', 'take_profit_threshold', float),
            'MAX_HOLD_TIME': ('trading', 'max_hold_time', float),
            'DRY_RUN': ('security', 'dry_run_mode', lambda x: x.lower() == 'true')
        }
        
        for env_key, (section, attr, converter) in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    if section == 'trading':
                        setattr(self.trading, attr, converted_value)
                    elif section == 'security':
                        setattr(self.security, attr, converted_value)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid environment variable {env_key}={env_value}: {e}")

    def _update_dataclass(self, dataclass_instance, data_dict):
        for key, value in data_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)

    def _validate_config(self):
        if self.security.enable_real_trading:
            if not self.security.private_key or len(self.security.private_key) != 66:
                raise ValueError("Valid private key required for real trading")
            
            if not self.security.alchemy_api_key or self.security.alchemy_api_key == 'demo':
                raise ValueError("Valid Alchemy API key required for real trading")
        
        if self.trading.confidence_threshold < 0.5 or self.trading.confidence_threshold > 1.0:
            raise ValueError("Confidence threshold must be between 0.5 and 1.0")
        
        if self.trading.momentum_threshold < 0.5 or self.trading.momentum_threshold > 1.0:
            raise ValueError("Momentum threshold must be between 0.5 and 1.0")
        
        if self.trading.max_position_size > self.trading.starting_capital:
            raise ValueError("Max position size cannot exceed starting capital")

    def get_trading_params(self) -> Dict[str, Any]:
        return asdict(self.trading)

    def get_security_params(self) -> Dict[str, Any]:
        return asdict(self.security)

    def get_chain_config(self, chain: str) -> Optional[ChainConfig]:
        return self.chains.get(chain)

    def update_trading_param(self, param: str, value: Any):
        if hasattr(self.trading, param):
            setattr(self.trading, param, value)
            self.save_config()

    def update_performance_based_params(self, roi: float, win_rate: float, sharpe: float, drawdown: float, trades: int):
        if trades < 10:
            return
        
        config_updated = False
        
        if roi < self.trading.roi_target:
            new_confidence = min(0.95, self.trading.confidence_threshold * 1.02)
            if abs(new_confidence - self.trading.confidence_threshold) > 0.001:
                self.trading.confidence_threshold = new_confidence
                config_updated = True
        elif roi > self.trading.roi_target * 1.5:
            new_confidence = max(0.60, self.trading.confidence_threshold * 0.98)
            if abs(new_confidence - self.trading.confidence_threshold) > 0.001:
                self.trading.confidence_threshold = new_confidence
                config_updated = True
        
        if win_rate < self.trading.win_rate_target:
            new_momentum = min(0.90, self.trading.momentum_threshold * 1.05)
            if abs(new_momentum - self.trading.momentum_threshold) > 0.001:
                self.trading.momentum_threshold = new_momentum
                config_updated = True
        elif win_rate > self.trading.win_rate_target * 1.2:
            new_momentum = max(0.50, self.trading.momentum_threshold * 0.97)
            if abs(new_momentum - self.trading.momentum_threshold) > 0.001:
                self.trading.momentum_threshold = new_momentum
                config_updated = True
        
        if drawdown > self.trading.max_drawdown_limit:
            new_stop_loss = max(0.02, self.trading.stop_loss_threshold * 0.8)
            if abs(new_stop_loss - self.trading.stop_loss_threshold) > 0.001:
                self.trading.stop_loss_threshold = new_stop_loss
                config_updated = True
            
            new_risk_score = max(0.2, self.trading.max_risk_score * 0.9)
            if abs(new_risk_score - self.trading.max_risk_score) > 0.001:
                self.trading.max_risk_score = new_risk_score
                config_updated = True
        
        if sharpe < self.trading.sharpe_target:
            new_volatility = min(0.20, self.trading.volatility_threshold * 1.1)
            if abs(new_volatility - self.trading.volatility_threshold) > 0.001:
                self.trading.volatility_threshold = new_volatility
                config_updated = True
        
        if config_updated:
            self.save_config()
            self.logger.info("Configuration updated based on performance metrics")

    def get_position_size(self, portfolio_value: float, confidence: float) -> float:
        kelly_fraction = self.calculate_kelly_criterion(confidence)
        base_size = portfolio_value * kelly_fraction * self.trading.kelly_multiplier
        return min(base_size, self.trading.max_position_size)

    def calculate_kelly_criterion(self, confidence: float) -> float:
        win_prob = confidence
        avg_win = self.trading.take_profit_threshold
        avg_loss = self.trading.stop_loss_threshold
        
        kelly = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        return max(0, min(kelly, 0.25))

    def should_exit_position(self, entry_momentum: float, current_momentum: float, hold_time: float) -> bool:
        momentum_decay = (entry_momentum - current_momentum) / entry_momentum
        
        if momentum_decay >= (1 - self.trading.price_momentum_decay):
            return True
        
        if hold_time >= self.trading.max_hold_time:
            return True
        
        return False

    def save_config(self):
        try:
            config_data = {
                'trading': asdict(self.trading),
                'security': {k: v for k, v in asdict(self.security).items() if k != 'private_key'},
                'last_updated': time.time()
            }
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def get_regime_adjusted_params(self, market_regime: str, confidence: float = 0.5) -> Dict[str, Any]:
        params = self.get_trading_params()
        
        if market_regime == 'high_volatility' and confidence > 0.7:
            params['confidence_threshold'] = min(0.95, params['confidence_threshold'] * 1.15)
            params['stop_loss_threshold'] = min(0.12, params['stop_loss_threshold'] * 1.3)
            params['max_hold_time'] = max(60, params['max_hold_time'] * 0.7)
            
        elif market_regime == 'low_volatility' and confidence > 0.7:
            params['confidence_threshold'] = max(0.60, params['confidence_threshold'] * 0.9)
            params['take_profit_threshold'] = min(0.25, params['take_profit_threshold'] * 1.2)
            params['max_hold_time'] = min(600, params['max_hold_time'] * 1.3)
            
        elif market_regime in ['bull_trend', 'bear_trend'] and confidence > 0.7:
            params['momentum_threshold'] = max(0.5, params['momentum_threshold'] * 0.95)
            params['min_price_change'] = max(7, params['min_price_change'] * 0.9)
        
        return params

    def is_trading_enabled(self) -> bool:
        return self.security.enable_real_trading and not self.security.dry_run_mode

    def get_api_config(self) -> Dict[str, str]:
        return {
            'alchemy_api_key': self.security.alchemy_api_key or 'demo',
            'etherscan_api_key': self.security.etherscan_api_key or 'demo'
        }

    def __repr__(self) -> str:
        return f"UnifiedConfig(trading_enabled={self.is_trading_enabled()}, chains={list(self.chains.keys())})"

global_config = UnifiedConfig()

def get_dynamic_config() -> Dict[str, Any]:
    return global_config.get_trading_params()

def update_performance(roi: float, win_rate: float, sharpe: float, drawdown: float, trades: int):
    global_config.update_performance_based_params(roi, win_rate, sharpe, drawdown, trades)

def get_chain_config(chain: str) -> Optional[ChainConfig]:
    return global_config.get_chain_config(chain)

def is_trading_enabled() -> bool:
    return global_config.is_trading_enabled()

def get_position_size(portfolio_value: float, confidence: float) -> float:
    return global_config.get_position_size(portfolio_value, confidence)