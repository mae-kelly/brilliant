import sys
sys.path.append('config')
from dynamic_parameters import get_dynamic_config, update_performance

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class MarketRegime:
    regime_id: int
    regime_name: str
    volatility_level: float
    trend_direction: float
    liquidity_level: float
    confidence: float
    duration: float

class RegimeDetector:
    def __init__(self):
        self.regime_models = {
            'volatility': GaussianMixture(n_components=3, random_state=42),
            'trend': KMeans(n_clusters=3, random_state=42),
            'liquidity': GaussianMixture(n_components=3, random_state=42)
        }
        
        self.scalers = {
            'volatility': StandardScaler(),
            'trend': StandardScaler(),
            'liquidity': StandardScaler()
        }
        
        self.regime_history = deque(maxlen=1000)
        self.current_regime = None
        self.regime_transitions = []
        
        self.regime_names = {
            0: 'Low_Vol_Sideways',
            1: 'Medium_Vol_Trending',
            2: 'High_Vol_Volatile',
            3: 'Bull_Market',
            4: 'Bear_Market',
            5: 'Consolidation',
            6: 'Breakout',
            7: 'Correction',
            8: 'Recovery'
        }

    async def detect_regime(self, market_data: Dict) -> MarketRegime:
        features = self.extract_regime_features(market_data)
        
        if len(self.regime_history) < 50:
            regime = MarketRegime(
                regime_id=0,
                regime_name='Initial',
                volatility_level=0.5,
                trend_direction=0.0,
                liquidity_level=0.5,
                confidence=0.5,
                duration=0.0
            )
        else:
            regime = await self.classify_regime(features)
        
        self.regime_history.append({
            'regime': regime,
            'features': features,
            'timestamp': time.time()
        })
        
        self.current_regime = regime
        return regime

    def extract_regime_features(self, market_data: Dict) -> np.ndarray:
        price_series = market_data.get('prices', [])
        volume_series = market_data.get('volumes', [])
        
        if len(price_series) < 10:
            return np.zeros(9)
        
        prices = np.array(price_series)
        volumes = np.array(volume_series) if volume_series else np.ones_like(prices)
        
        volatility = self.calculate_volatility(prices)
        trend_strength = self.calculate_trend_strength(prices)
        trend_direction = self.calculate_trend_direction(prices)
        liquidity_score = self.calculate_liquidity_score(volumes)
        momentum = self.calculate_momentum(prices)
        mean_reversion = self.calculate_mean_reversion(prices)
        volume_profile = self.calculate_volume_profile(volumes)
        correlation_breakdown = self.calculate_correlation_breakdown(prices)
        regime_persistence = self.calculate_regime_persistence()
        
        return np.array([
            volatility, trend_strength, trend_direction, liquidity_score,
            momentum, mean_reversion, volume_profile, correlation_breakdown,
            regime_persistence
        ])

    def calculate_volatility(self, prices: np.ndarray) -> float:
        if len(prices) < 5:
            return 0.0
        
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        volatility = np.std(returns) * np.sqrt(252)
        return volatility

    def calculate_trend_strength(self, prices: np.ndarray) -> float:
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        r_squared = np.corrcoef(x, prices)[0, 1] ** 2
        trend_strength = abs(slope) * r_squared
        
        return trend_strength

    def calculate_trend_direction(self, prices: np.ndarray) -> float:
        if len(prices) < 5:
            return 0.0
        
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-10:]) if len(prices) >= 10 else short_ma
        
        trend_direction = (short_ma - long_ma) / (long_ma + 1e-10)
        return np.tanh(trend_direction * 10)

    def calculate_liquidity_score(self, volumes: np.ndarray) -> float:
        if len(volumes) < 5:
            return 0.5
        
        recent_volume = np.mean(volumes[-5:])
        historical_volume = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_volume
        
        liquidity_score = recent_volume / (historical_volume + 1e-6)
        return min(liquidity_score / 2.0, 1.0)

    def calculate_momentum(self, prices: np.ndarray) -> float:
        if len(prices) < 10:
            return 0.0
        
        momentum_periods = [3, 5, 10]
        momentum_scores = []
        
        for period in momentum_periods:
            if len(prices) >= period:
                momentum = (prices[-1] - prices[-period]) / (prices[-period] + 1e-10)
                momentum_scores.append(momentum)
        
        return np.mean(momentum_scores) if momentum_scores else 0.0

    def calculate_mean_reversion(self, prices: np.ndarray) -> float:
        if len(prices) < 20:
            return 0.0
        
        mean_price = np.mean(prices)
        current_price = prices[-1]
        
        deviation = (current_price - mean_price) / (mean_price + 1e-10)
        
        hurst_exponent = self.calculate_hurst_exponent(prices)
        mean_reversion_tendency = 1.0 - hurst_exponent
        
        return mean_reversion_tendency * abs(deviation)

    def calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        if len(prices) < 10:
            return 0.5
        
        try:
            lags = range(2, min(10, len(prices) // 2))
            tau = []
            
            for lag in lags:
                pp = np.subtract(prices[lag:], prices[:-lag])
                variance = np.var(pp)
                tau.append(np.sqrt(variance))
            
            if len(tau) < 2:
                return 0.5
            
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = m[0]
            
            return max(0.0, min(1.0, hurst))
            
        except:
            return 0.5

    def calculate_volume_profile(self, volumes: np.ndarray) -> float:
        if len(volumes) < 5:
            return 0.5
        
        volume_changes = np.diff(volumes)
        volume_volatility = np.std(volume_changes)
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
        
        volume_profile = (volume_volatility + abs(volume_trend)) / (np.mean(volumes) + 1e-6)
        return min(volume_profile, 1.0)

    def calculate_correlation_breakdown(self, prices: np.ndarray) -> float:
        if len(prices) < 20:
            return 0.0
        
        mid_point = len(prices) // 2
        first_half = prices[:mid_point]
        second_half = prices[mid_point:]
        
        if len(first_half) < 3 or len(second_half) < 3:
            return 0.0
        
        correlation = np.corrcoef(
            range(len(first_half)), first_half
        )[0, 1] if len(first_half) > 1 else 0.0
        
        recent_correlation = np.corrcoef(
            range(len(second_half)), second_half
        )[0, 1] if len(second_half) > 1 else 0.0
        
        correlation_breakdown = abs(correlation - recent_correlation)
        return correlation_breakdown

    def calculate_regime_persistence(self) -> float:
        if len(self.regime_history) < 10:
            return 0.5
        
        recent_regimes = [entry['regime'].regime_id for entry in list(self.regime_history)[-10:]]
        regime_changes = sum(1 for i in range(1, len(recent_regimes)) if recent_regimes[i] != recent_regimes[i-1])
        
        persistence = 1.0 - (regime_changes / len(recent_regimes))
        return persistence

    async def classify_regime(self, features: np.ndarray) -> MarketRegime:
        if len(self.regime_history) < 100:
            await self.train_regime_models()
        
        regime_scores = {}
        
        try:
            volatility_regime = self.classify_volatility_regime(features)
            trend_regime = self.classify_trend_regime(features)
            liquidity_regime = self.classify_liquidity_regime(features)
            
            composite_regime_id = self.combine_regime_classifications(
                volatility_regime, trend_regime, liquidity_regime
            )
            
            regime_name = self.regime_names.get(composite_regime_id, 'Unknown')
            
            confidence = self.calculate_regime_confidence(features, composite_regime_id)
            duration = self.calculate_regime_duration(composite_regime_id)
            
            return MarketRegime(
                regime_id=composite_regime_id,
                regime_name=regime_name,
                volatility_level=features[0] if len(features) > 0 else 0.5,
                trend_direction=features[2] if len(features) > 2 else 0.0,
                liquidity_level=features[3] if len(features) > 3 else 0.5,
                confidence=confidence,
                duration=duration
            )
            
        except Exception as e:
            return MarketRegime(
                regime_id=0,
                regime_name='Unknown',
                volatility_level=0.5,
                trend_direction=0.0,
                liquidity_level=0.5,
                confidence=0.5,
                duration=0.0
            )

    def classify_volatility_regime(self, features: np.ndarray) -> int:
        volatility = features[0]
        
        if volatility < 0.1:
            return 0  # Low volatility
        elif volatility < 0.25:
            return 1  # Medium volatility
        else:
            return 2  # High volatility

    def classify_trend_regime(self, features: np.ndarray) -> int:
        trend_strength = features[1] if len(features) > 1 else 0.0
        trend_direction = features[2] if len(features) > 2 else 0.0
        
        if trend_strength > 0.5:
            if trend_direction > 0.1:
                return 1  # Strong uptrend
            elif trend_direction < -0.1:
                return 2  # Strong downtrend
            else:
                return 0  # Sideways
        else:
            return 0  # Weak trend (sideways)

    def classify_liquidity_regime(self, features: np.ndarray) -> int:
        liquidity_score = features[3] if len(features) > 3 else 0.5
        
        if liquidity_score > 0.8:
            return 2  # High liquidity
        elif liquidity_score > 0.4:
            return 1  # Medium liquidity
        else:
            return 0  # Low liquidity

    def combine_regime_classifications(self, vol_regime: int, trend_regime: int, liq_regime: int) -> int:
        regime_matrix = {
            (0, 0, 0): 0,  # Low vol, sideways, low liq -> Consolidation
            (0, 0, 1): 0,  # Low vol, sideways, med liq -> Consolidation
            (0, 0, 2): 0,  # Low vol, sideways, high liq -> Consolidation
            (0, 1, 0): 3,  # Low vol, uptrend, low liq -> Bull market
            (0, 1, 1): 3,  # Low vol, uptrend, med liq -> Bull market
            (0, 1, 2): 3,  # Low vol, uptrend, high liq -> Bull market
            (0, 2, 0): 4,  # Low vol, downtrend, low liq -> Bear market
            (0, 2, 1): 4,  # Low vol, downtrend, med liq -> Bear market
            (0, 2, 2): 4,  # Low vol, downtrend, high liq -> Bear market
            (1, 0, 0): 5,  # Med vol, sideways, low liq -> Consolidation
            (1, 0, 1): 5,  # Med vol, sideways, med liq -> Consolidation
            (1, 0, 2): 5,  # Med vol, sideways, high liq -> Consolidation
            (1, 1, 0): 6,  # Med vol, uptrend, low liq -> Breakout
            (1, 1, 1): 6,  # Med vol, uptrend, med liq -> Breakout
            (1, 1, 2): 6,  # Med vol, uptrend, high liq -> Breakout
            (1, 2, 0): 7,  # Med vol, downtrend, low liq -> Correction
            (1, 2, 1): 7,  # Med vol, downtrend, med liq -> Correction
            (1, 2, 2): 7,  # Med vol, downtrend, high liq -> Correction
            (2, 0, 0): 2,  # High vol, sideways, low liq -> High vol volatile
            (2, 0, 1): 2,  # High vol, sideways, med liq -> High vol volatile
            (2, 0, 2): 2,  # High vol, sideways, high liq -> High vol volatile
            (2, 1, 0): 8,  # High vol, uptrend, low liq -> Recovery
            (2, 1, 1): 8,  # High vol, uptrend, med liq -> Recovery
            (2, 1, 2): 8,  # High vol, uptrend, high liq -> Recovery
            (2, 2, 0): 2,  # High vol, downtrend, low liq -> High vol volatile
            (2, 2, 1): 2,  # High vol, downtrend, med liq -> High vol volatile
            (2, 2, 2): 2,  # High vol, downtrend, high liq -> High vol volatile
        }
        
        return regime_matrix.get((vol_regime, trend_regime, liq_regime), 0)

    def calculate_regime_confidence(self, features: np.ndarray, regime_id: int) -> float:
        if len(self.regime_history) < 10:
            return 0.5
        
        recent_regimes = [entry['regime'].regime_id for entry in list(self.regime_history)[-10:]]
        regime_consistency = sum(1 for r in recent_regimes if r == regime_id) / len(recent_regimes)
        
        feature_consistency = self.calculate_feature_consistency(features)
        
        confidence = (regime_consistency + feature_consistency) / 2.0
        return confidence

    def calculate_feature_consistency(self, features: np.ndarray) -> float:
        if len(self.regime_history) < 5:
            return 0.5
        
        recent_features = [entry['features'] for entry in list(self.regime_history)[-5:]]
        
        if not recent_features:
            return 0.5
        
        feature_stds = []
        for i in range(len(features)):
            feature_values = [f[i] for f in recent_features if len(f) > i]
            if feature_values:
                feature_stds.append(np.std(feature_values))
        
        if not feature_stds:
            return 0.5
        
        consistency = 1.0 - np.mean(feature_stds)
        return max(0.0, min(1.0, consistency))

    def calculate_regime_duration(self, regime_id: int) -> float:
        if not self.regime_history:
            return 0.0
        
        duration = 0.0
        for entry in reversed(list(self.regime_history)):
            if entry['regime'].regime_id == regime_id:
                duration += 1.0
            else:
                break
        
        return duration

    async def train_regime_models(self):
        if len(self.regime_history) < 50:
            return
        
        features_list = []
        for entry in self.regime_history:
            features_list.append(entry['features'])
        
        X = np.array(features_list)
        
        try:
            volatility_features = X[:, [0, 4, 5]]  # volatility, momentum, mean_reversion
            self.scalers['volatility'].fit(volatility_features)
            scaled_vol_features = self.scalers['volatility'].transform(volatility_features)
            self.regime_models['volatility'].fit(scaled_vol_features)
            
            trend_features = X[:, [1, 2, 4]]  # trend_strength, trend_direction, momentum
            self.scalers['trend'].fit(trend_features)
            scaled_trend_features = self.scalers['trend'].transform(trend_features)
            self.regime_models['trend'].fit(scaled_trend_features)
            
            liquidity_features = X[:, [3, 6, 7]]  # liquidity_score, volume_profile, correlation_breakdown
            self.scalers['liquidity'].fit(liquidity_features)
            scaled_liq_features = self.scalers['liquidity'].transform(liquidity_features)
            self.regime_models['liquidity'].fit(scaled_liq_features)
            
        except Exception as e:
            pass

    def get_regime_statistics(self) -> Dict:
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for entry in self.regime_history:
            regime_id = entry['regime'].regime_id
            regime_counts[regime_id] = regime_counts.get(regime_id, 0) + 1
        
        total_entries = len(self.regime_history)
        regime_probabilities = {k: v / total_entries for k, v in regime_counts.items()}
        
        return {
            'current_regime': self.current_regime.regime_name if self.current_regime else 'Unknown',
            'regime_probabilities': regime_probabilities,
            'total_observations': total_entries,
            'regime_transitions': len(self.regime_transitions)
        }

regime_detector = RegimeDetector()
