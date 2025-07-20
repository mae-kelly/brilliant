import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import pickle
import os

@dataclass
class ExecutionState:
    market_volatility: float
    bid_ask_spread: float
    order_book_imbalance: float
    recent_volume: float
    price_momentum: float
    time_of_day: float
    gas_price: float
    mev_risk: float
    liquidity_depth: float

@dataclass
class ExecutionAction:
    timing_delay: float
    slice_size: float
    execution_strategy: int
    gas_price_multiplier: float
    slippage_tolerance: float

@dataclass
class ExecutionResult:
    realized_slippage: float
    execution_time: float
    price_impact: float
    gas_cost: float
    fill_ratio: float
    mev_detected: bool

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.value_head = nn.Linear(action_dim, 1)
        self.advantage_head = nn.Linear(action_dim, action_dim)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        features = self.network(x)
        
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class MultiArmedBandit:
    def __init__(self, n_arms: int = 5):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.alpha = 0.1
        self.epsilon = 0.1
        
        self.dex_mapping = {
            0: 'uniswap_v2',
            1: 'uniswap_v3',
            2: 'sushiswap',
            3: 'curve',
            4: 'balancer'
        }
    
    def select_dex(self, context: Dict) -> str:
        if random.random() < self.epsilon:
            arm = random.randint(0, self.n_arms - 1)
        else:
            arm = np.argmax(self.values)
        
        return self.dex_mapping[arm]
    
    def update(self, dex_name: str, reward: float):
        arm = next(k for k, v in self.dex_mapping.items() if v == dex_name)
        
        self.counts[arm] += 1
        self.values[arm] += self.alpha * (reward - self.values[arm])
        
        self.epsilon = max(0.01, self.epsilon * 0.995)

class RLExecutionAgent:
    def __init__(self, state_dim: int = 9, action_dim: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        self.replay_buffer = ReplayBuffer()
        self.dex_bandit = MultiArmedBandit()
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.batch_size = 64
        self.target_update_freq = 1000
        
        self.steps = 0
        self.logger = logging.getLogger(__name__)
        
    def encode_state(self, execution_state: ExecutionState) -> torch.Tensor:
        state_vector = np.array([
            execution_state.market_volatility,
            execution_state.bid_ask_spread,
            execution_state.order_book_imbalance,
            execution_state.recent_volume,
            execution_state.price_momentum,
            execution_state.time_of_day,
            execution_state.gas_price,
            execution_state.mev_risk,
            execution_state.liquidity_depth
        ])
        
        state_vector = np.clip(state_vector, -5, 5)
        
        return torch.FloatTensor(state_vector).to(self.device)
    
    def decode_action(self, action_idx: int, state: ExecutionState) -> ExecutionAction:
        action_space = {
            0: ExecutionAction(0.0, 1.0, 0, 1.0, 0.01),
            1: ExecutionAction(5.0, 0.5, 1, 1.1, 0.02),
            2: ExecutionAction(15.0, 0.3, 2, 1.2, 0.03),
            3: ExecutionAction(30.0, 0.2, 3, 1.3, 0.04),
            4: ExecutionAction(60.0, 0.1, 4, 1.5, 0.05)
        }
        
        base_action = action_space[action_idx]
        
        volatility_adjustment = state.market_volatility * 0.1
        gas_adjustment = (state.gas_price - 20) / 20 * 0.1
        
        return ExecutionAction(
            timing_delay=base_action.timing_delay + volatility_adjustment * 10,
            slice_size=max(0.05, base_action.slice_size - volatility_adjustment),
            execution_strategy=base_action.execution_strategy,
            gas_price_multiplier=base_action.gas_price_multiplier + gas_adjustment,
            slippage_tolerance=base_action.slippage_tolerance + volatility_adjustment * 0.01
        )
    
    def select_action(self, state: ExecutionState) -> Tuple[ExecutionAction, str]:
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            state_tensor = self.encode_state(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        execution_action = self.decode_action(action_idx, state)
        
        dex_context = {
            'volatility': state.market_volatility,
            'spread': state.bid_ask_spread,
            'volume': state.recent_volume
        }
        selected_dex = self.dex_bandit.select_dex(dex_context)
        
        return execution_action, selected_dex
    
    def calculate_reward(self, execution_result: ExecutionResult, target_execution: Dict) -> float:
        slippage_penalty = -abs(execution_result.realized_slippage - target_execution.get('target_slippage', 0.01)) * 100
        
        time_penalty = -max(0, execution_result.execution_time - target_execution.get('target_time', 30)) * 0.1
        
        gas_penalty = -execution_result.gas_cost / 1000
        
        fill_bonus = execution_result.fill_ratio * 10
        
        mev_penalty = -20 if execution_result.mev_detected else 0
        
        impact_penalty = -execution_result.price_impact * 50
        
        reward = slippage_penalty + time_penalty + gas_penalty + fill_bonus + mev_penalty + impact_penalty
        
        return np.clip(reward, -100, 50)
    
    def store_experience(self, state: ExecutionState, action_idx: int, reward: float, 
                        next_state: ExecutionState, done: bool):
        state_tensor = self.encode_state(state).cpu()
        next_state_tensor = self.encode_state(next_state).cpu()
        
        experience = Experience(state_tensor, action_idx, reward, next_state_tensor, done)
        self.replay_buffer.push(experience)
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = torch.stack([exp.state for exp in batch]).to(self.device)
        actions = torch.tensor([exp.action for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float).to(self.device)
        next_states = torch.stack([exp.next_state for exp in batch]).to(self.device)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.bool).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_dex_performance(self, dex_name: str, execution_result: ExecutionResult):
        reward = -execution_result.realized_slippage * 100 + execution_result.fill_ratio * 10
        self.dex_bandit.update(dex_name, reward)
    
    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'dex_bandit': {
                'counts': self.dex_bandit.counts,
                'values': self.dex_bandit.values,
                'epsilon': self.dex_bandit.epsilon
            }
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            
            bandit_data = checkpoint.get('dex_bandit', {})
            self.dex_bandit.counts = bandit_data.get('counts', np.zeros(5))
            self.dex_bandit.values = bandit_data.get('values', np.zeros(5))
            self.dex_bandit.epsilon = bandit_data.get('epsilon', 0.1)
            
            self.logger.info(f"Model loaded from {filepath}")

class AdaptiveSlippageManager:
    def __init__(self):
        self.base_tolerance = 0.01
        self.volatility_multiplier = 0.5
        self.volume_factor = 0.3
        self.liquidity_factor = 0.4
        
        self.recent_slippages = deque(maxlen=100)
        self.market_regime_history = deque(maxlen=50)
        
    def calculate_dynamic_slippage(self, market_state: ExecutionState, 
                                 historical_performance: Dict) -> float:
        
        volatility_adjustment = market_state.market_volatility * self.volatility_multiplier
        
        volume_adjustment = max(-0.005, min(0.01, 
            (market_state.recent_volume - 1000000) / 10000000 * self.volume_factor))
        
        liquidity_adjustment = max(-0.01, min(0.005,
            (50000 - market_state.liquidity_depth) / 100000 * self.liquidity_factor))
        
        spread_adjustment = market_state.bid_ask_spread * 0.5
        
        recent_avg_slippage = np.mean(self.recent_slippages) if self.recent_slippages else 0
        performance_adjustment = recent_avg_slippage * 0.3
        
        dynamic_tolerance = (
            self.base_tolerance +
            volatility_adjustment +
            volume_adjustment +
            liquidity_adjustment +
            spread_adjustment +
            performance_adjustment
        )
        
        return np.clip(dynamic_tolerance, 0.005, 0.1)
    
    def update_slippage_history(self, realized_slippage: float, market_regime: str):
        self.recent_slippages.append(realized_slippage)
        self.market_regime_history.append(market_regime)
        
        regime_performance = self.analyze_regime_performance()
        self.adjust_parameters(regime_performance)
    
    def analyze_regime_performance(self) -> Dict[str, float]:
        if len(self.market_regime_history) < 10:
            return {}
        
        regime_slippages = {}
        for i, regime in enumerate(self.market_regime_history):
            if regime not in regime_slippages:
                regime_slippages[regime] = []
            regime_slippages[regime].append(self.recent_slippages[i])
        
        regime_performance = {}
        for regime, slippages in regime_slippages.items():
            regime_performance[regime] = np.mean(slippages)
        
        return regime_performance
    
    def adjust_parameters(self, regime_performance: Dict[str, float]):
        if not regime_performance:
            return
        
        overall_performance = np.mean(list(regime_performance.values()))
        
        if overall_performance > self.base_tolerance * 1.5:
            self.volatility_multiplier *= 1.05
            self.volume_factor *= 1.02
        elif overall_performance < self.base_tolerance * 0.8:
            self.volatility_multiplier *= 0.98
            self.volume_factor *= 0.99
        
        self.volatility_multiplier = np.clip(self.volatility_multiplier, 0.1, 1.0)
        self.volume_factor = np.clip(self.volume_factor, 0.1, 0.5)

class ReinforcementLearningExecutor:
    def __init__(self):
        self.rl_agent = RLExecutionAgent()
        self.slippage_manager = AdaptiveSlippageManager()
        self.execution_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
        self.training_mode = True
        self.model_path = 'models/rl_execution_agent.pth'
        
    async def initialize(self):
        if os.path.exists(self.model_path):
            self.rl_agent.load_model(self.model_path)
            self.training_mode = False
            self.logger.info("Loaded pre-trained RL execution model")
        else:
            self.logger.info("Starting with untrained RL execution model")
    
    async def execute_trade_with_rl(self, token_address: str, chain: str, 
                                  amount: float, side: str, 
                                  market_data: Dict) -> Dict:
        
        execution_state = self.create_execution_state(market_data)
        
        action, selected_dex = self.rl_agent.select_action(execution_state)
        
        dynamic_slippage = self.slippage_manager.calculate_dynamic_slippage(
            execution_state, 
            self.get_historical_performance()
        )
        
        execution_plan = {
            'dex': selected_dex,
            'timing_delay': action.timing_delay,
            'slice_size': action.slice_size,
            'execution_strategy': action.execution_strategy,
            'gas_multiplier': action.gas_price_multiplier,
            'slippage_tolerance': dynamic_slippage,
            'total_amount': amount
        }
        
        if action.timing_delay > 0:
            await asyncio.sleep(action.timing_delay)
        
        execution_result = await self.execute_with_strategy(execution_plan, token_address, chain, side)
        
        if self.training_mode:
            await self.update_learning_systems(execution_state, action, execution_result, selected_dex)
        
        return execution_result
    
    def create_execution_state(self, market_data: Dict) -> ExecutionState:
        current_hour = datetime.now().hour / 24.0
        
        return ExecutionState(
            market_volatility=market_data.get('volatility', 0.1),
            bid_ask_spread=market_data.get('spread', 0.01),
            order_book_imbalance=market_data.get('imbalance', 0.0),
            recent_volume=market_data.get('volume_1h', 100000),
            price_momentum=market_data.get('momentum', 0.0),
            time_of_day=current_hour,
            gas_price=market_data.get('gas_price', 20),
            mev_risk=market_data.get('mev_risk', 0.1),
            liquidity_depth=market_data.get('liquidity', 100000)
        )
    
    async def execute_with_strategy(self, execution_plan: Dict, token_address: str, 
                                  chain: str, side: str) -> Dict:
        
        strategy_map = {
            0: self.execute_immediate,
            1: self.execute_twap,
            2: self.execute_vwap,
            3: self.execute_iceberg,
            4: self.execute_stealth
        }
        
        strategy_func = strategy_map.get(execution_plan['execution_strategy'], self.execute_immediate)
        
        start_time = time.time()
        
        try:
            result = await strategy_func(execution_plan, token_address, chain, side)
            
            execution_time = time.time() - start_time
            
            execution_result = ExecutionResult(
                realized_slippage=result.get('slippage', 0.01),
                execution_time=execution_time,
                price_impact=result.get('price_impact', 0.005),
                gas_cost=result.get('gas_cost', 50),
                fill_ratio=result.get('fill_ratio', 1.0),
                mev_detected=result.get('mev_detected', False)
            )
            
            return {
                'success': result.get('success', True),
                'execution_result': execution_result,
                'tx_hash': result.get('tx_hash', ''),
                'filled_amount': result.get('filled_amount', execution_plan['total_amount'])
            }
            
        except Exception as e:
            self.logger.error(f"Execution strategy failed: {e}")
            return {
                'success': False,
                'execution_result': ExecutionResult(0.1, execution_time, 0.05, 100, 0.0, True),
                'tx_hash': '',
                'filled_amount': 0
            }
    
    async def execute_immediate(self, plan: Dict, token_address: str, chain: str, side: str) -> Dict:
        await asyncio.sleep(0.1)
        
        return {
            'success': True,
            'slippage': np.random.uniform(0.005, 0.02),
            'price_impact': np.random.uniform(0.001, 0.01),
            'gas_cost': np.random.uniform(30, 80),
            'fill_ratio': 1.0,
            'mev_detected': np.random.random() < 0.1,
            'tx_hash': f"0x{'a' * 64}"
        }
    
    async def execute_twap(self, plan: Dict, token_address: str, chain: str, side: str) -> Dict:
        total_amount = plan['total_amount']
        slice_size = plan['slice_size']
        num_slices = int(1 / slice_size)
        
        total_slippage = 0
        total_gas = 0
        filled_amount = 0
        
        for i in range(num_slices):
            slice_amount = total_amount / num_slices
            
            slice_result = await self.execute_immediate(plan, token_address, chain, side)
            
            if slice_result['success']:
                total_slippage += slice_result['slippage']
                total_gas += slice_result['gas_cost']
                filled_amount += slice_amount
            
            await asyncio.sleep(5)
        
        return {
            'success': filled_amount > 0,
            'slippage': total_slippage / num_slices,
            'price_impact': np.random.uniform(0.002, 0.008),
            'gas_cost': total_gas,
            'fill_ratio': filled_amount / total_amount,
            'mev_detected': False,
            'tx_hash': f"0x{'b' * 64}"
        }
    
    async def execute_vwap(self, plan: Dict, token_address: str, chain: str, side: str) -> Dict:
        return await self.execute_twap(plan, token_address, chain, side)
    
    async def execute_iceberg(self, plan: Dict, token_address: str, chain: str, side: str) -> Dict:
        return await self.execute_twap(plan, token_address, chain, side)
    
    async def execute_stealth(self, plan: Dict, token_address: str, chain: str, side: str) -> Dict:
        return await self.execute_twap(plan, token_address, chain, side)
    
    async def update_learning_systems(self, state: ExecutionState, action: ExecutionAction, 
                                    execution_result: ExecutionResult, dex: str):
        
        target_execution = {'target_slippage': 0.01, 'target_time': 30}
        reward = self.rl_agent.calculate_reward(execution_result, target_execution)
        
        next_state = state
        done = True
        
        action_idx = self.get_action_index(action)
        self.rl_agent.store_experience(state, action_idx, reward, next_state, done)
        
        self.rl_agent.update_dex_performance(dex, execution_result)
        
        if len(self.rl_agent.replay_buffer) >= self.rl_agent.batch_size:
            self.rl_agent.train()
        
        self.slippage_manager.update_slippage_history(execution_result.realized_slippage, "normal")
        
        self.execution_history.append({
            'timestamp': time.time(),
            'state': state,
            'action': action,
            'result': execution_result,
            'reward': reward,
            'dex': dex
        })
        
        if len(self.execution_history) % 100 == 0:
            self.rl_agent.save_model(self.model_path)
    
    def get_action_index(self, action: ExecutionAction) -> int:
        if action.timing_delay == 0:
            return 0
        elif action.timing_delay <= 10:
            return 1
        elif action.timing_delay <= 20:
            return 2
        elif action.timing_delay <= 40:
            return 3
        else:
            return 4
    
    def get_historical_performance(self) -> Dict:
        if not self.execution_history:
            return {}
        
        recent_executions = list(self.execution_history)[-50:]
        
        avg_slippage = np.mean([ex['result'].realized_slippage for ex in recent_executions])
        avg_fill_ratio = np.mean([ex['result'].fill_ratio for ex in recent_executions])
        avg_time = np.mean([ex['result'].execution_time for ex in recent_executions])
        
        return {
            'avg_slippage': avg_slippage,
            'avg_fill_ratio': avg_fill_ratio,
            'avg_execution_time': avg_time,
            'total_executions': len(recent_executions)
        }

rl_executor = ReinforcementLearningExecutor()