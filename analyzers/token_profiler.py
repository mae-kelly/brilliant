
# Dynamic configuration import
import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

@dataclass
class TokenProfile:
    address: str
    name: str
    symbol: str
    decimals: int
    total_supply: float
    market_cap_usd: float
    volume_24h: float
    liquidity_usd: float
    price_usd: float
    age_hours: float
    velocity_score: float
    volatility_score: float
    momentum_score: float
    social_score: float
    safety_score: float
    overall_score: float
    risk_category: str
    profiled_at: float

class TokenProfiler:
    def __init__(self):
        self.profiles = {}
        self.trending_tokens = deque(maxlen=100)
        self.price_feeds = defaultdict(lambda: deque(maxlen=200))
        self.volume_feeds = defaultdict(lambda: deque(maxlen=200))
        
        self.risk_categories = {
            'SAFE': (0.8, 1.0),
            'LOW_RISK': (0.6, 0.8),
            'MEDIUM_RISK': (0.4, 0.6),
            'HIGH_RISK': (0.2, 0.4),
            'EXTREME_RISK': (0.0, 0.2)
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def profile_token(self, token_address: str, chain: str = 'ethereum') -> TokenProfile:
        start_time = time.time()
        
        if token_address in self.profiles:
            cached = self.profiles[token_address]
            if time.time() - cached.profiled_at < 300:
                return cached
        
        basic_info = await self.get_token_basic_info(token_address)
        market_data = await self.get_market_data(token_address)
        technical_analysis = await self.analyze_technical_indicators(token_address)
        social_metrics = await self.analyze_social_metrics(token_address)
        safety_analysis = await self.analyze_safety_metrics(token_address)
        
        velocity_score = technical_analysis['velocity']
        volatility_score = technical_analysis['volatility']
        momentum_score = technical_analysis['momentum']
        social_score = social_metrics['score']
        safety_score = safety_analysis['score']
        
        overall_score = self.calculate_overall_score(
            velocity_score, volatility_score, momentum_score, 
            social_score, safety_score, market_data
        )
        
        risk_category = self.determine_risk_category(overall_score)
        
        profile = TokenProfile(
            address=token_address,
            name=basic_info['name'],
            symbol=basic_info['symbol'],
            decimals=basic_info['decimals'],
            total_supply=basic_info['total_supply'],
            market_cap_usd=market_data['market_cap'],
            volume_24h=market_data['volume_24h'],
            liquidity_usd=market_data['liquidity'],
            price_usd=market_data['price'],
            age_hours=basic_info['age_hours'],
            velocity_score=velocity_score,
            volatility_score=volatility_score,
            momentum_score=momentum_score,
            social_score=social_score,
            safety_score=safety_score,
            overall_score=overall_score,
            risk_category=risk_category,
            profiled_at=time.time()
        )
        
        self.profiles[token_address] = profile
        
        if overall_score > 0.7:
            self.trending_tokens.append(profile)
        
        self.logger.info(
            f"📊 Profiled {basic_info['symbol']}: "
            f"Score: {overall_score:.2f} "
            f"Risk: {risk_category} "
            f"Cap: ${market_data['market_cap']:,.0f}"
        )
        
        return profile

    async def get_token_basic_info(self, token_address: str) -> Dict:
        await asyncio.sleep(0.1)
        
        addr_hash = hash(token_address)
        
        symbols = ['MOON', 'ROCKET', 'DOGE', 'PEPE', 'SHIB', 'FLOKI', 'SAFE', 'GEM']
        symbol = symbols[addr_hash % len(symbols)] + str(addr_hash % 1000)
        
        return {
            'name': f"{symbol} Token",
            'symbol': symbol,
            'decimals': 18,
            'total_supply': float(1000000 + (addr_hash % 999000000)),
            'age_hours': float(1 + (addr_hash % 8760))
        }

    async def get_market_data(self, token_address: str) -> Dict:
        await asyncio.sleep(0.05)
        
        addr_hash = hash(token_address + 'market')
        
        price = (addr_hash % 10000) / 10000000
        market_cap = price * (100000 + (addr_hash % 10000000))
        volume_24h = market_cap * ((addr_hash % 50) / 100)
        liquidity = market_cap * ((addr_hash % 20) / 100)
        
        return {
            'price': price,
            'market_cap': market_cap,
            'volume_24h': volume_24h,
            'liquidity': liquidity
        }

    async def analyze_technical_indicators(self, token_address: str) -> Dict:
        await asyncio.sleep(0.1)
        
        price_feed = self.price_feeds[token_address]
        volume_feed = self.volume_feeds[token_address]
        
        current_price = hash(token_address + str(time.time())) % 1000 / 1000000
        current_volume = hash(token_address + 'vol' + str(time.time())) % 10000
        
        price_feed.append(current_price)
        volume_feed.append(current_volume)
        
        if len(price_feed) < 10:
            return {'velocity': 0.5, 'volatility': 0.5, 'momentum': 0.5}
        
        prices = np.array(list(price_feed))
        volumes = np.array(list(volume_feed))
        
        velocity = self.calculate_velocity(prices)
        volatility = self.calculate_volatility(prices)
        momentum = self.calculate_momentum(prices, volumes)
        
        return {
            'velocity': velocity,
            'volatility': volatility,
            'momentum': momentum
        }

    def calculate_velocity(self, prices: np.ndarray) -> float:
        if len(prices) < 2:
            return 0.5
        
        price_changes = np.diff(prices)
        velocity = np.mean(price_changes) / (np.mean(prices) + 1e-10)
        
        return np.clip(0.5 + velocity * 10, 0.0, 1.0)

    def calculate_volatility(self, prices: np.ndarray) -> float:
        if len(prices) < 2:
            return 0.5
        
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        volatility = np.std(returns)
        
        return np.clip(volatility * 20, 0.0, 1.0)

    def calculate_momentum(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        if len(prices) < 5:
            return 0.5
        
        price_momentum = (prices[-1] - prices[-5]) / (prices[-5] + 1e-10)
        volume_momentum = (volumes[-1] - np.mean(volumes[-5:])) / (np.mean(volumes[-5:]) + 1e-10)
        
        combined_momentum = 0.7 * price_momentum + 0.3 * volume_momentum
        
        return np.clip(0.5 + combined_momentum * 2, 0.0, 1.0)

    async def analyze_social_metrics(self, token_address: str) -> Dict:
        await asyncio.sleep(0.05)
        
        social_hash = hash(token_address + 'social') % 100
        
        twitter_mentions = social_hash * 10
        telegram_members = social_hash * 100
        reddit_posts = social_hash * 2
        
        social_score = min((twitter_mentions + telegram_members/10 + reddit_posts*5) / 1000, 1.0)
        
        return {
            'score': social_score,
            'twitter_mentions': twitter_mentions,
            'telegram_members': telegram_members,
            'reddit_posts': reddit_posts
        }

    async def analyze_safety_metrics(self, token_address: str) -> Dict:
        await asyncio.sleep(0.05)
        
        safety_hash = hash(token_address + 'safety') % 100
        
        contract_verified = safety_hash > 30
        ownership_renounced = safety_hash > 60
        liquidity_locked = safety_hash > 40
        no_mint_function = safety_hash > 50
        
        safety_score = sum([contract_verified, ownership_renounced, liquidity_locked, no_mint_function]) / 4
        
        return {
            'score': safety_score,
            'contract_verified': contract_verified,
            'ownership_renounced': ownership_renounced,
            'liquidity_locked': liquidity_locked,
            'no_mint_function': no_mint_function
        }

    def calculate_overall_score(self, velocity: float, volatility: float, momentum: float, 
                              social: float, safety: float, market_data: Dict) -> float:
        
        liquidity_factor = min(market_data['liquidity'] / 50000, 1.0)
        volume_factor = min(market_data['volume_24h'] / 10000, 1.0)
        
        weighted_score = (
            velocity * 0.2 +
            momentum * 0.25 +
            social * 0.15 +
            safety * 0.3 +
            liquidity_factor * 0.05 +
            volume_factor * 0.05
        )
        
        volatility_penalty = max(0, volatility - 0.7) * 0.3
        
        final_score = max(0, weighted_score - volatility_penalty)
        
        return min(final_score, 1.0)

    def determine_risk_category(self, score: float) -> str:
        for category, (min_score, max_score) in self.risk_categories.items():
            if min_score <= score < max_score:
                return category
        return 'EXTREME_RISK'

    def get_trending_tokens(self, limit: int = 20) -> List[TokenProfile]:
        sorted_trending = sorted(
            self.trending_tokens, 
            key=lambda x: x.overall_score, 
            reverse=True
        )
        return sorted_trending[:limit]

    def get_profile_stats(self) -> Dict:
        if not self.profiles:
            return {'total_profiles': 0}
        
        scores = [p.overall_score for p in self.profiles.values()]
        risk_counts = defaultdict(int)
        
        for profile in self.profiles.values():
            risk_counts[profile.risk_category] += 1
        
        return {
            'total_profiles': len(self.profiles),
            'avg_score': np.mean(scores),
            'trending_count': len(self.trending_tokens),
            'risk_distribution': dict(risk_counts)
        }

token_profiler = TokenProfiler()
