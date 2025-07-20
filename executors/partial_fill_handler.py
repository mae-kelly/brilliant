
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from optimizer import get_dynamic_config, update_performance
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

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class FillStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PartialFill:
    fill_id: str
    order_id: str
    amount_filled: float
    price_filled: float
    timestamp: float
    tx_hash: str
    gas_used: int

@dataclass
class OrderBook:
    order_id: str
    token_address: str
    chain: str
    side: str
    total_amount: float
    filled_amount: float
    remaining_amount: float
    target_price: float
    fills: List[PartialFill]
    status: FillStatus
    created_at: float
    updated_at: float
    max_slippage: float
    time_limit: Optional[float]

class PartialFillHandler:
    def __init__(self):
        self.active_orders = {}
        self.fill_strategies = {
            'aggressive': {'max_slippage': 0.05, 'time_limit': 60},
            'moderate': {'max_slippage': 0.03, 'time_limit': 180},
            'patient': {'max_slippage': 0.01, 'time_limit': 600}
        }
        
        self.execution_engine = None
        self.liquidity_monitor = None

    async def initialize(self, execution_engine, liquidity_monitor):
        self.execution_engine = execution_engine
        self.liquidity_monitor = liquidity_monitor
        
        asyncio.create_task(self.order_processing_loop())
        asyncio.create_task(self.fill_monitoring_loop())

    async def submit_order(self, token_address: str, chain: str, side: str,
                          total_amount: float, target_price: float,
                          strategy: str = 'moderate') -> str:
        
        order_id = f"order_{int(time.time() * 1000)}"
        
        strategy_params = self.fill_strategies.get(strategy, self.fill_strategies['moderate'])
        
        order = OrderBook(
            order_id=order_id,
            token_address=token_address,
            chain=chain,
            side=side,
            total_amount=total_amount,
            filled_amount=0.0,
            remaining_amount=total_amount,
            target_price=target_price,
            fills=[],
            status=FillStatus.PENDING,
            created_at=time.time(),
            updated_at=time.time(),
            max_slippage=strategy_params['max_slippage'],
            time_limit=strategy_params.get('time_limit')
        )
        
        self.active_orders[order_id] = order
        return order_id

    async def order_processing_loop(self):
        while True:
            try:
                for order in list(self.active_orders.values()):
                    if order.status == FillStatus.PENDING or order.status == FillStatus.PARTIAL:
                        await self.process_order(order)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                await asyncio.sleep(10)

    async def process_order(self, order: OrderBook):
        if order.remaining_amount <= 0:
            order.status = FillStatus.COMPLETE
            return
        
        if order.time_limit and (time.time() - order.created_at) > order.time_limit:
            order.status = FillStatus.CANCELLED
            return
        
        current_price = await self.get_current_price(order.token_address, order.chain)
        if not current_price:
            return
        
        price_deviation = abs(current_price - order.target_price) / order.target_price
        
        if price_deviation > order.max_slippage:
            return
        
        available_liquidity = await self.get_available_liquidity(order.token_address, order.chain)
        
        fill_amount = min(
            order.remaining_amount,
            available_liquidity * 0.1,
            self.calculate_optimal_fill_size(order, current_price)
        )
        
        if fill_amount > 0:
            await self.execute_partial_fill(order, fill_amount, current_price)

    async def get_current_price(self, token_address: str, chain: str) -> Optional[float]:
        return np.random.uniform(0.001, 10.0)

    async def get_available_liquidity(self, token_address: str, chain: str) -> float:
        return np.random.uniform(10000, 1000000)

    def calculate_optimal_fill_size(self, order: OrderBook, current_price: float) -> float:
        market_conditions = self.assess_market_conditions(order.token_address, order.chain)
        
        base_fill_ratio = 0.2
        
        if market_conditions['volatility'] > 0.15:
            base_fill_ratio *= 0.7
        elif market_conditions['volatility'] < 0.05:
            base_fill_ratio *= 1.3
        
        if market_conditions['liquidity_score'] > 0.8:
            base_fill_ratio *= 1.2
        elif market_conditions['liquidity_score'] < 0.3:
            base_fill_ratio *= 0.6
        
        optimal_fill = order.remaining_amount * base_fill_ratio
        
        return max(order.remaining_amount * 0.05, min(optimal_fill, order.remaining_amount * 0.5))

    def assess_market_conditions(self, token_address: str, chain: str) -> Dict[str, float]:
        return {
            'volatility': np.random.uniform(0.02, 0.25),
            'liquidity_score': np.random.uniform(0.2, 1.0),
            'trend_strength': np.random.uniform(-1.0, 1.0),
            'volume_spike': np.random.uniform(0.5, 3.0)
        }

    async def execute_partial_fill(self, order: OrderBook, fill_amount: float, fill_price: float):
        try:
            if order.side == 'buy':
                result = await self.execution_engine.execute_buy_trade(
                    order.token_address, order.chain, fill_amount
                )
            else:
                result = await self.execution_engine.execute_sell_trade(
                    order.token_address, order.chain, int(fill_amount * 1000000)
                )
            
            if result and result.get('success', False):
                fill = PartialFill(
                    fill_id=f"fill_{int(time.time() * 1000)}",
                    order_id=order.order_id,
                    amount_filled=fill_amount,
                    price_filled=fill_price,
                    timestamp=time.time(),
                    tx_hash=result.get('tx_hash', ''),
                    gas_used=result.get('gas_used', 0)
                )
                
                order.fills.append(fill)
                order.filled_amount += fill_amount
                order.remaining_amount -= fill_amount
                order.updated_at = time.time()
                
                if order.remaining_amount <= 0.01:
                    order.status = FillStatus.COMPLETE
                else:
                    order.status = FillStatus.PARTIAL
                
                await self.notify_fill_update(order, fill)
            
        except Exception as e:
            pass

    async def notify_fill_update(self, order: OrderBook, fill: PartialFill):
        fill_percentage = (order.filled_amount / order.total_amount) * 100
        
        print(f"📊 Partial Fill: {order.order_id} - {fill_percentage:.1f}% complete")
        print(f"   Amount: {fill.amount_filled:.6f} at price {fill.price_filled:.6f}")
        print(f"   Remaining: {order.remaining_amount:.6f}")

    async def fill_monitoring_loop(self):
        while True:
            try:
                await self.monitor_stale_orders()
                await self.optimize_pending_orders()
                await asyncio.sleep(30)
                
            except Exception as e:
                await asyncio.sleep(60)

    async def monitor_stale_orders(self):
        current_time = time.time()
        
        for order in list(self.active_orders.values()):
            if order.status in [FillStatus.COMPLETE, FillStatus.FAILED, FillStatus.CANCELLED]:
                continue
            
            time_elapsed = current_time - order.updated_at
            
            if time_elapsed > 300:
                if order.filled_amount > 0:
                    await self.attempt_completion(order)
                else:
                    order.status = FillStatus.FAILED

    async def attempt_completion(self, order: OrderBook):
        if order.remaining_amount <= order.total_amount * 0.05:
            order.status = FillStatus.COMPLETE
            return
        
        current_price = await self.get_current_price(order.token_address, order.chain)
        if not current_price:
            return
        
        relaxed_slippage = order.max_slippage * 1.5
        price_deviation = abs(current_price - order.target_price) / order.target_price
        
        if price_deviation <= relaxed_slippage:
            await self.execute_partial_fill(order, order.remaining_amount, current_price)

    async def optimize_pending_orders(self):
        for order in self.active_orders.values():
            if order.status == FillStatus.PARTIAL:
                await self.adjust_order_strategy(order)

    async def adjust_order_strategy(self, order: OrderBook):
        time_elapsed = time.time() - order.created_at
        fill_ratio = order.filled_amount / order.total_amount
        
        if time_elapsed > 120 and fill_ratio < 0.3:
            order.max_slippage = min(order.max_slippage * 1.2, 0.08)
        
        elif fill_ratio > 0.8:
            order.max_slippage = min(order.max_slippage * 1.5, 0.10)

    async def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        
        if order.status in [FillStatus.COMPLETE, FillStatus.FAILED]:
            return False
        
        order.status = FillStatus.CANCELLED
        order.updated_at = time.time()
        
        return True

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        if order_id not in self.active_orders:
            return None
        
        order = self.active_orders[order_id]
        
        avg_fill_price = 0.0
        if order.fills:
            total_value = sum(fill.amount_filled * fill.price_filled for fill in order.fills)
            avg_fill_price = total_value / order.filled_amount
        
        return {
            'order_id': order.order_id,
            'token_address': order.token_address,
            'chain': order.chain,
            'side': order.side,
            'total_amount': order.total_amount,
            'filled_amount': order.filled_amount,
            'remaining_amount': order.remaining_amount,
            'fill_percentage': (order.filled_amount / order.total_amount) * 100,
            'avg_fill_price': avg_fill_price,
            'target_price': order.target_price,
            'status': order.status.value,
            'created_at': order.created_at,
            'updated_at': order.updated_at,
            'total_fills': len(order.fills),
            'time_elapsed': time.time() - order.created_at
        }

    def get_active_orders_summary(self) -> Dict:
        active_count = sum(1 for order in self.active_orders.values() 
                          if order.status in [FillStatus.PENDING, FillStatus.PARTIAL])
        
        total_volume = sum(order.total_amount for order in self.active_orders.values())
        filled_volume = sum(order.filled_amount for order in self.active_orders.values())
        
        return {
            'active_orders': active_count,
            'total_orders': len(self.active_orders),
            'total_volume': total_volume,
            'filled_volume': filled_volume,
            'fill_rate': (filled_volume / total_volume) * 100 if total_volume > 0 else 0
        }

partial_fill_handler = PartialFillHandler()
