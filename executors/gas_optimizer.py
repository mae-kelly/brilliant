import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import time
import numpy as np
from web3 import Web3
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import requests

@dataclass
class GasMetrics:
    base_fee: int
    priority_fee: int
    network_congestion: float
    pending_tx_count: int
    block_utilization: float

class GasOptimizer:
    def __init__(self):
        self.gas_history = deque(maxlen=100)
        self.congestion_metrics = {}
        self.gas_stations = {
            'ethereum': 'https://ethgasstation.info/api/ethgasAPI.json',
            'polygon': 'https://gasstation-mainnet.matic.network/v2',
            'arbitrum': 'https://arbiscan.io/api?module=gastracker&action=gasoracle'
        }

    async def calculate_optimal_gas_price(self, urgency_score: float, mev_risk: float, chain: str, w3: Web3) -> Dict[str, int]:
        try:
            current_metrics = await self.get_gas_metrics(w3, chain)
            historical_data = await self.get_historical_gas_data(chain)
            
            base_fee = current_metrics.base_fee
            network_load = current_metrics.network_congestion
            
            urgency_multiplier = 1.0 + (urgency_score * 0.5)
            mev_protection_multiplier = 1.0 + (mev_risk * 0.3)
            congestion_multiplier = 1.0 + (network_load * 0.4)
            
            target_priority_fee = int(
                current_metrics.priority_fee * 
                urgency_multiplier * 
                mev_protection_multiplier * 
                congestion_multiplier
            )
            
            if chain == 'ethereum':
                return {
                    'maxFeePerGas': base_fee + target_priority_fee,
                    'maxPriorityFeePerGas': target_priority_fee
                }
            else:
                return {
                    'gasPrice': base_fee + target_priority_fee
                }
                
        except Exception as e:
            return await self.get_fallback_gas_price(chain, w3)

    async def get_gas_metrics(self, w3: Web3, chain: str) -> GasMetrics:
        try:
            latest_block = w3.eth.get_block('latest', full_transactions=True)
            base_fee = latest_block.get('baseFeePerGas', 0)
            
            priority_fee = w3.eth.max_priority_fee if hasattr(w3.eth, 'max_priority_fee') else 2000000000
            
            pending_tx_count = len(latest_block['transactions'])
            block_gas_used = latest_block['gasUsed']
            block_gas_limit = latest_block['gasLimit']
            
            block_utilization = block_gas_used / block_gas_limit
            network_congestion = min(block_utilization * 2, 1.0)
            
            return GasMetrics(
                base_fee=base_fee,
                priority_fee=priority_fee,
                network_congestion=network_congestion,
                pending_tx_count=pending_tx_count,
                block_utilization=block_utilization
            )
            
        except Exception as e:
            return GasMetrics(20000000000, 2000000000, 0.5, 100, 0.5)

    async def get_historical_gas_data(self, chain: str) -> List[int]:
        try:
            if chain in self.gas_stations:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.gas_stations[chain]) as response:
                        data = await response.json()
                        
                        if chain == 'ethereum':
                            return [
                                int(data.get('fast', 50)) * 100000000,
                                int(data.get('standard', 40)) * 100000000,
                                int(data.get('safeLow', 30)) * 100000000
                            ]
                        elif chain == 'polygon':
                            return [
                                int(data.get('fast', {}).get('maxFee', 50)) * 1000000000,
                                int(data.get('standard', {}).get('maxFee', 40)) * 1000000000,
                                int(data.get('safeLow', {}).get('maxFee', 30)) * 1000000000
                            ]
                            
        except Exception as e:
            pass
        
        return [50000000000, 40000000000, 30000000000]

    async def get_fallback_gas_price(self, chain: str, w3: Web3) -> Dict[str, int]:
        try:
            gas_price = w3.eth.gas_price
            
            fallback_prices = {
                'ethereum': int(gas_price * 1.2),
                'arbitrum': int(gas_price * 1.1),
                'optimism': int(gas_price * 1.1),
                'polygon': int(gas_price * 1.3)
            }
            
            price = fallback_prices.get(chain, int(gas_price * 1.2))
            
            if chain == 'ethereum':
                return {
                    'maxFeePerGas': price,
                    'maxPriorityFeePerGas': int(price * 0.1)
                }
            else:
                return {'gasPrice': price}
                
        except Exception as e:
            default_prices = {
                'ethereum': {'maxFeePerGas': 50000000000, 'maxPriorityFeePerGas': 2000000000},
                'arbitrum': {'gasPrice': 1000000000},
                'optimism': {'gasPrice': 1000000000},
                'polygon': {'gasPrice': 30000000000}
            }
            
            return default_prices.get(chain, {'gasPrice': 20000000000})

    async def monitor_gas_trends(self, chain: str, w3: Web3):
        while True:
            try:
                metrics = await self.get_gas_metrics(w3, chain)
                self.gas_history.append({
                    'timestamp': time.time(),
                    'base_fee': metrics.base_fee,
                    'priority_fee': metrics.priority_fee,
                    'congestion': metrics.network_congestion,
                    'chain': chain
                })
                
                await asyncio.sleep(30)
                
            except Exception as e:
                await asyncio.sleep(60)

    def predict_gas_trend(self, chain: str) -> float:
        if len(self.gas_history) < 10:
            return 0.0
        
        recent_fees = [entry['base_fee'] for entry in list(self.gas_history)[-10:] if entry['chain'] == chain]
        
        if len(recent_fees) < 5:
            return 0.0
        
        trend = np.polyfit(range(len(recent_fees)), recent_fees, 1)[0]
        return float(trend)

gas_optimizer = GasOptimizer()
