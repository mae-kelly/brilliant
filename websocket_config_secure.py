#!/usr/bin/env python3
"""
SECURE WebSocket Scanner Configuration Management
All addresses loaded from environment variables
"""

import os
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ChainConfig:
    name: str
    rpc_endpoints: List[str]
    websocket_endpoints: List[str]
    dex_factories: Dict[str, str]
    gas_token: str
    chain_id: int

@dataclass
class ScannerConfig:
    chains: Dict[str, ChainConfig]
    performance_targets: Dict[str, int]
    risk_thresholds: Dict[str, float]
    worker_counts: Dict[str, int]

class SecureWebSocketConfigManager:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self) -> ScannerConfig:
        """Load scanner configuration from environment variables"""
        
        chains = {
            'ethereum': ChainConfig(
                name='ethereum',
                rpc_endpoints=[
                    f"https://eth-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                    f"https://mainnet.infura.io/v3/{os.getenv('INFURA_API_KEY', 'demo')}",
                ],
                websocket_endpoints=[
                    "wss://ethereum-rpc.publicnode.com",
                    "wss://eth.llamarpc.com",
                ],
                dex_factories={
                    'uniswap_v2': os.getenv('UNISWAP_V2_FACTORY', '0x0000000000000000000000000000000000000000'),
                    'uniswap_v3': os.getenv('UNISWAP_V3_FACTORY', '0x0000000000000000000000000000000000000000'),
                    'sushiswap': os.getenv('SUSHISWAP_FACTORY', '0x0000000000000000000000000000000000000000'),
                },
                gas_token='ETH',
                chain_id=1
            ),
            'arbitrum': ChainConfig(
                name='arbitrum',
                rpc_endpoints=[
                    f"https://arb-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                    "https://arb1.arbitrum.io/rpc",
                ],
                websocket_endpoints=[
                    "wss://arbitrum-one.publicnode.com",
                    "wss://arbitrum.llamarpc.com",
                ],
                dex_factories={
                    'uniswap_v3': os.getenv('ARBITRUM_UNISWAP_V3', '0x0000000000000000000000000000000000000000'),
                    'camelot': os.getenv('ARBITRUM_CAMELOT', '0x0000000000000000000000000000000000000000'),
                    'sushiswap': os.getenv('ARBITRUM_SUSHISWAP', '0x0000000000000000000000000000000000000000'),
                },
                gas_token='ETH',
                chain_id=42161
            ),
            'polygon': ChainConfig(
                name='polygon',
                rpc_endpoints=[
                    f"https://polygon-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY', 'demo')}",
                    "https://polygon-rpc.com",
                ],
                websocket_endpoints=[
                    "wss://polygon-bor-rpc.publicnode.com",
                    "wss://polygon.llamarpc.com",
                ],
                dex_factories={
                    'quickswap': os.getenv('POLYGON_QUICKSWAP', '0x0000000000000000000000000000000000000000'),
                    'sushiswap': os.getenv('POLYGON_SUSHISWAP', '0x0000000000000000000000000000000000000000'),
                    'uniswap_v3': os.getenv('POLYGON_UNISWAP_V3', '0x0000000000000000000000000000000000000000'),
                },
                gas_token='MATIC',
                chain_id=137
            )
        }
        
        return ScannerConfig(
            chains=chains,
            performance_targets={
                'tokens_per_day': 10000,
                'events_per_second': 1000,
                'signals_per_hour': 100,
                'latency_ms': 100
            },
            risk_thresholds={
                'momentum_score_min': 0.7,
                'confidence_min': 0.75,
                'honeypot_risk_max': 0.3,
                'rug_risk_max': 0.2,
                'liquidity_min_usd': 10000
            },
            worker_counts={
                'websocket_connections_per_chain': 2,
                'momentum_analyzers': 5,
                'transaction_processors': 10,
                'signal_validators': 5
            }
        )

config_manager = SecureWebSocketConfigManager()
