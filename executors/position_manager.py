import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

class PositionStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class Order:
    order_id: str
    token_address: str
    chain: str
    order_type: OrderType
    side: str
    amount: float
    price: Optional[float]
    status: str
    created_at: float
    filled_at: Optional[float] = None
    filled_amount: float = 0.0
    filled_price: Optional[float] = None
    tx_hash: Optional[str] = None

@dataclass
class Position:
    position_id: str
    token_address: str
    chain: str
    strategy: str
    entry_price: float
    current_price: float
    size: float
    unrealized_pnl: float
    realized_pnl: float
    status: PositionStatus
    entry_time: float
    orders: List[Order] = field(default_factory=list)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_time: Optional[float] = None
    risk_params: Dict[str, float] = field(default_factory=dict)

class PositionManager:
    def __init__(self):
        self.positions = {}
        self.pending_orders = {}
        self.position_limits = {
            'max_positions': 10,
            'max_position_size_usd': 10.0,
            'max_total_exposure_usd': 50.0,
            'max_concentration_ratio': 0.3
        }
        
        self.execution_engine = None
        self.price_feed = None
        
        self.stats = {
            'total_positions': 0,
            'active_positions': 0,
            'profitable_positions': 0,
            'total_pnl': 0.0,
            'avg_hold_time': 0.0
        }

    async def initialize(self, execution_engine, price_feed):
        self.execution_engine = execution_engine
        self.price_feed = price_feed
        
        asyncio.create_task(self.position_monitor_loop())
        asyncio.create_task(self.order_execution_loop())
        asyncio.create_task(self.risk_monitor_loop())

    async def open_position(self, token_address: str, chain: str, strategy: str,
                           size: float, entry_signal: Dict) -> Optional[str]:
        
        if not await self.validate_position_limits(token_address, size):
            return None
        
        position_id = str(uuid.uuid4())
        
        current_price = await self.get_current_price(token_address, chain)
        if not current_price:
            return None
        
        stop_loss, take_profit = self.calculate_risk_levels(
            current_price, entry_signal, strategy
        )
        
        max_hold_time = entry_signal.get('max_hold_time', 300)
        
        position = Position(
            position_id=position_id,
            token_address=token_address,
            chain=chain,
            strategy=strategy,
            entry_price=current_price,
            current_price=current_price,
            size=size,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            status=PositionStatus.PENDING,
            entry_time=time.time(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_time=max_hold_time,
            risk_params=entry_signal.get('risk_params', {})
        )
        
        entry_order = await self.create_market_order(
            position, 'buy', size, current_price
        )
        
        if entry_order:
            position.orders.append(entry_order)
            self.positions[position_id] = position
            self.pending_orders[entry_order.order_id] = entry_order
            
            return position_id
        
        return None

    async def close_position(self, position_id: str, reason: str = "manual") -> bool:
        if position_id not in self.positions:
            return False
        
        position = self.positions[position_id]
        
        if position.status not in [PositionStatus.OPEN, PositionStatus.PENDING]:
            return False
        
        position.status = PositionStatus.CLOSING
        
        current_price = await self.get_current_price(position.token_address, position.chain)
        if not current_price:
            return False
        
        exit_order = await self.create_market_order(
            position, 'sell', position.size, current_price
        )
        
        if exit_order:
            position.orders.append(exit_order)
            self.pending_orders[exit_order.order_id] = exit_order
            return True
        
        return False

    async def create_market_order(self, position: Position, side: str, 
                                 amount: float, price: float) -> Optional[Order]:
        order_id = str(uuid.uuid4())
        
        order = Order(
            order_id=order_id,
            token_address=position.token_address,
            chain=position.chain,
            order_type=OrderType.MARKET,
            side=side,
            amount=amount,
            price=price,
            status="pending",
            created_at=time.time()
        )
        
        return order

    async def create_stop_loss_order(self, position: Position) -> Optional[Order]:
        if not position.stop_loss:
            return None
        
        order_id = str(uuid.uuid4())
        
        order = Order(
            order_id=order_id,
            token_address=position.token_address,
            chain=position.chain,
            order_type=OrderType.STOP_LOSS,
            side='sell',
            amount=position.size,
            price=position.stop_loss,
            status="pending",
            created_at=time.time()
        )
        
        return order

    async def create_take_profit_order(self, position: Position) -> Optional[Order]:
        if not position.take_profit:
            return None
        
        order_id = str(uuid.uuid4())
        
        order = Order(
            order_id=order_id,
            token_address=position.token_address,
            chain=position.chain,
            order_type=OrderType.TAKE_PROFIT,
            side='sell',
            amount=position.size,
            price=position.take_profit,
            status="pending",
            created_at=time.time()
        )
        
        return order

    async def validate_position_limits(self, token_address: str, size: float) -> bool:
        active_positions = [p for p in self.positions.values() 
                          if p.status in [PositionStatus.OPEN, PositionStatus.PENDING]]
        
        if len(active_positions) >= self.position_limits['max_positions']:
            return False
        
        position_value = size * await self.get_current_price(token_address, "ethereum")
        if position_value > self.position_limits['max_position_size_usd']:
            return False
        
        total_exposure = sum(p.size * p.current_price for p in active_positions)
        if total_exposure + position_value > self.position_limits['max_total_exposure_usd']:
            return False
        
        token_exposure = sum(p.size * p.current_price for p in active_positions 
                           if p.token_address == token_address)
        concentration_ratio = (token_exposure + position_value) / (total_exposure + position_value)
        
        if concentration_ratio > self.position_limits['max_concentration_ratio']:
            return False
        
        return True

    def calculate_risk_levels(self, entry_price: float, entry_signal: Dict, 
                            strategy: str) -> Tuple[Optional[float], Optional[float]]:
        
        risk_config = get_dynamic_config()
        
        stop_loss_pct = risk_config.get('stop_loss_threshold', 0.05)
        take_profit_pct = risk_config.get('take_profit_threshold', 0.12)
        
        volatility = entry_signal.get('volatility', 0.1)
        confidence = entry_signal.get('confidence', 0.5)
        
        adjusted_stop_loss = stop_loss_pct * (1 + volatility)
        adjusted_take_profit = take_profit_pct * confidence
        
        stop_loss = entry_price * (1 - adjusted_stop_loss)
        take_profit = entry_price * (1 + adjusted_take_profit)
        
        return stop_loss, take_profit

    async def get_current_price(self, token_address: str, chain: str) -> Optional[float]:
        if self.price_feed:
            return await self.price_feed.get_price(token_address, chain)
        
        return np.random.uniform(0.001, 10.0)

    async def position_monitor_loop(self):
        while True:
            try:
                await self.update_positions()
                await self.check_exit_conditions()
                await asyncio.sleep(5)
                
            except Exception as e:
                await asyncio.sleep(10)

    async def update_positions(self):
        for position in self.positions.values():
            if position.status == PositionStatus.OPEN:
                current_price = await self.get_current_price(
                    position.token_address, position.chain
                )
                
                if current_price:
                    position.current_price = current_price
                    position.unrealized_pnl = (current_price - position.entry_price) * position.size

    async def check_exit_conditions(self):
        current_time = time.time()
        
        for position in list(self.positions.values()):
            if position.status != PositionStatus.OPEN:
                continue
            
            should_exit = False
            exit_reason = ""
            
            if position.stop_loss and position.current_price <= position.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
            
            elif position.take_profit and position.current_price >= position.take_profit:
                should_exit = True
                exit_reason = "take_profit"
            
            elif position.max_hold_time and (current_time - position.entry_time) >= position.max_hold_time:
                should_exit = True
                exit_reason = "time_exit"
            
            elif await self.check_momentum_decay(position):
                should_exit = True
                exit_reason = "momentum_decay"
            
            if should_exit:
                await self.close_position(position.position_id, exit_reason)

    async def check_momentum_decay(self, position: Position) -> bool:
        momentum_threshold = position.risk_params.get('momentum_exit_threshold', 0.7)
        
        if 'initial_momentum' in position.risk_params:
            current_momentum = await self.get_current_momentum(
                position.token_address, position.chain
            )
            
            if current_momentum < position.risk_params['initial_momentum'] * momentum_threshold:
                return True
        
        return False

    async def get_current_momentum(self, token_address: str, chain: str) -> float:
        return np.random.uniform(0.3, 0.9)

    async def order_execution_loop(self):
        while True:
            try:
                await self.process_pending_orders()
                await asyncio.sleep(1)
                
            except Exception as e:
                await asyncio.sleep(5)

    async def process_pending_orders(self):
        for order_id, order in list(self.pending_orders.items()):
            if order.status == "pending":
                result = await self.execute_order(order)
                
                if result:
                    await self.handle_order_fill(order, result)
                    del self.pending_orders[order_id]

    async def execute_order(self, order: Order) -> Optional[Dict]:
        if not self.execution_engine:
            await asyncio.sleep(0.5)
            return {
                'success': True,
                'tx_hash': f"0x{'a' * 64}",
                'filled_price': order.price,
                'filled_amount': order.amount,
                'gas_used': 250000
            }
        
        if order.side == 'buy':
            return await self.execution_engine.execute_buy_trade(
                order.token_address, order.chain, order.amount
            )
        else:
            return await self.execution_engine.execute_sell_trade(
                order.token_address, order.chain, int(order.amount * 1000000)
            )

    async def handle_order_fill(self, order: Order, result: Dict):
        order.status = "filled" if result['success'] else "failed"
        order.filled_at = time.time()
        order.tx_hash = result.get('tx_hash')
        order.filled_price = result.get('filled_price', order.price)
        order.filled_amount = result.get('filled_amount', order.amount)
        
        position = next((p for p in self.positions.values() 
                        if any(o.order_id == order.order_id for o in p.orders)), None)
        
        if position:
            await self.update_position_on_fill(position, order, result)

    async def update_position_on_fill(self, position: Position, order: Order, result: Dict):
        if order.side == 'buy' and order.status == "filled":
            position.status = PositionStatus.OPEN
            position.entry_price = order.filled_price
            
            if position.stop_loss and position.take_profit:
                stop_order = await self.create_stop_loss_order(position)
                profit_order = await self.create_take_profit_order(position)
                
                if stop_order:
                    position.orders.append(stop_order)
                    self.pending_orders[stop_order.order_id] = stop_order
                
                if profit_order:
                    position.orders.append(profit_order)
                    self.pending_orders[profit_order.order_id] = profit_order
        
        elif order.side == 'sell' and order.status == "filled":
            position.status = PositionStatus.CLOSED
            
            exit_price = order.filled_price
            realized_pnl = (exit_price - position.entry_price) * position.size
            position.realized_pnl = realized_pnl
            
            self.stats['total_pnl'] += realized_pnl
            self.stats['total_positions'] += 1
            
            if realized_pnl > 0:
                self.stats['profitable_positions'] += 1
            
            hold_time = time.time() - position.entry_time
            self.stats['avg_hold_time'] = (
                (self.stats['avg_hold_time'] * (self.stats['total_positions'] - 1) + hold_time) /
                self.stats['total_positions']
            )

    async def risk_monitor_loop(self):
        while True:
            try:
                await self.monitor_portfolio_risk()
                await asyncio.sleep(30)
                
            except Exception as e:
                await asyncio.sleep(60)

    async def monitor_portfolio_risk(self):
        active_positions = [p for p in self.positions.values() 
                          if p.status == PositionStatus.OPEN]
        
        if not active_positions:
            return
        
        total_exposure = sum(p.size * p.current_price for p in active_positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in active_positions)
        
        portfolio_drawdown = total_unrealized_pnl / total_exposure if total_exposure > 0 else 0
        
        if portfolio_drawdown < -0.15:
            await self.emergency_close_all_positions("portfolio_drawdown")
        
        for position in active_positions:
            position_risk = abs(position.unrealized_pnl) / (position.size * position.entry_price)
            
            if position_risk > 0.20:
                await self.close_position(position.position_id, "high_risk")

    async def emergency_close_all_positions(self, reason: str):
        active_positions = [p for p in self.positions.values() 
                          if p.status == PositionStatus.OPEN]
        
        for position in active_positions:
            await self.close_position(position.position_id, f"emergency_{reason}")

    def get_position_summary(self) -> Dict:
        active_positions = [p for p in self.positions.values() 
                          if p.status == PositionStatus.OPEN]
        
        total_unrealized_pnl = sum(p.unrealized_pnl for p in active_positions)
        total_exposure = sum(p.size * p.current_price for p in active_positions)
        
        return {
            'active_positions': len(active_positions),
            'total_positions': len(self.positions),
            'total_exposure_usd': total_exposure,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': self.stats['total_pnl'],
            'win_rate': (self.stats['profitable_positions'] / max(self.stats['total_positions'], 1)) * 100,
            'avg_hold_time': self.stats['avg_hold_time']
        }

    def get_position_details(self, position_id: str) -> Optional[Dict]:
        if position_id not in self.positions:
            return None
        
        position = self.positions[position_id]
        
        return {
            'position_id': position.position_id,
            'token_address': position.token_address,
            'chain': position.chain,
            'strategy': position.strategy,
            'status': position.status.value,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'size': position.size,
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'entry_time': position.entry_time,
            'hold_time': time.time() - position.entry_time,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'orders': [
                {
                    'order_id': order.order_id,
                    'type': order.order_type.value,
                    'side': order.side,
                    'status': order.status,
                    'amount': order.amount,
                    'price': order.price,
                    'filled_amount': order.filled_amount,
                    'filled_price': order.filled_price
                } for order in position.orders
            ]
        }

position_manager = PositionManager()
