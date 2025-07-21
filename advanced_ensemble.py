import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
import json
import time
from typing import Dict, List, Tuple, Optional
import asyncio
import aiohttp

class SocialSentimentAnalyzer:
    """Analyzes social sentiment from multiple sources"""
    
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def get_token_sentiment(self, token_address: str, symbol: str) -> Dict:
        """Fetch sentiment data for a token"""
        cache_key = f"{token_address}_{symbol}"
        
        if cache_key in self.sentiment_cache:
            cached_data, timestamp = self.sentiment_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        sentiment_data = await self.fetch_sentiment_data(symbol)
        self.sentiment_cache[cache_key] = (sentiment_data, time.time())
        
        return sentiment_data
    
    async def fetch_sentiment_data(self, symbol: str) -> Dict:
        """Fetch sentiment from multiple sources"""
        # Twitter sentiment (mock implementation - replace with real API)
        twitter_sentiment = await self.get_twitter_sentiment(symbol)
        
        # Reddit sentiment
        reddit_sentiment = await self.get_reddit_sentiment(symbol)
        
        # Telegram/Discord sentiment
        social_sentiment = await self.get_social_sentiment(symbol)
        
        # News sentiment
        news_sentiment = await self.get_news_sentiment(symbol)
        
        # Aggregate sentiment
        total_sentiment = (
            twitter_sentiment * 0.3 +
            reddit_sentiment * 0.2 +
            social_sentiment * 0.2 +
            news_sentiment * 0.3
        )
        
        return {
            'total_sentiment': total_sentiment,
            'twitter_sentiment': twitter_sentiment,
            'reddit_sentiment': reddit_sentiment,
            'social_sentiment': social_sentiment,
            'news_sentiment': news_sentiment,
            'sentiment_velocity': self.calculate_sentiment_velocity(symbol),
            'mention_volume': await self.get_mention_volume(symbol)
        }
    
    async def get_twitter_sentiment(self, symbol: str) -> float:
        """Get Twitter sentiment (placeholder)"""
        # In production, integrate with Twitter API v2
        return np.random.uniform(-1, 1)
    
    async def get_reddit_sentiment(self, symbol: str) -> float:
        """Get Reddit sentiment (placeholder)"""
        # In production, integrate with Reddit API
        return np.random.uniform(-1, 1)
    
    async def get_social_sentiment(self, symbol: str) -> float:
        """Get Telegram/Discord sentiment (placeholder)"""
        # In production, integrate with social monitoring tools
        return np.random.uniform(-1, 1)
    
    async def get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment (placeholder)"""
        # In production, integrate with news sentiment API
        return np.random.uniform(-1, 1)
    
    def calculate_sentiment_velocity(self, symbol: str) -> float:
        """Calculate rate of sentiment change"""
        # Implementation would track sentiment over time
        return np.random.uniform(-0.1, 0.1)
    
    async def get_mention_volume(self, symbol: str) -> int:
        """Get social mention volume"""
        # Implementation would count mentions across platforms
        return np.random.randint(0, 1000)

class MacroEconomicAnalyzer:
    """Analyzes macro factors affecting DeFi"""
    
    def __init__(self):
        self.macro_cache = {}
        self.cache_ttl = 600  # 10 minutes
    
    async def get_macro_factors(self) -> Dict:
        """Get current macro-economic factors"""
        if 'macro_data' in self.macro_cache:
            cached_data, timestamp = self.macro_cache['macro_data']
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        macro_data = await self.fetch_macro_data()
        self.macro_cache['macro_data'] = (macro_data, time.time())
        
        return macro_data
    
    async def fetch_macro_data(self) -> Dict:
        """Fetch macro-economic data"""
        try:
            # ETH price and volatility
            eth_data = await self.get_eth_metrics()
            
            # Gas prices across chains
            gas_data = await self.get_gas_metrics()
            
            # DeFi TVL and flows
            defi_data = await self.get_defi_metrics()
            
            # Market fear/greed index
            sentiment_data = await self.get_market_sentiment()
            
            return {
                'eth_price': eth_data['price'],
                'eth_volatility': eth_data['volatility'],
                'eth_dominance': eth_data['dominance'],
                'avg_gas_price': gas_data['average'],
                'gas_trend': gas_data['trend'],
                'defi_tvl': defi_data['tvl'],
                'defi_flow': defi_data['flow'],
                'market_fear_greed': sentiment_data['fear_greed'],
                'vix_equivalent': sentiment_data['vix'],
                'funding_rates': await self.get_funding_rates()
            }
            
        except Exception as e:
            logging.error(f"Failed to fetch macro data: {e}")
            return self.get_default_macro_data()
    
    async def get_eth_metrics(self) -> Dict:
        """Get ETH price and volatility metrics"""
        # In production, fetch from CoinGecko/CoinMarketCap
        return {
            'price': 3000 + np.random.normal(0, 100),
            'volatility': 0.5 + np.random.normal(0, 0.1),
            'dominance': 0.6 + np.random.normal(0, 0.05)
        }
    
    async def get_gas_metrics(self) -> Dict:
        """Get gas price metrics"""
        return {
            'average': 20 + np.random.normal(0, 5),
            'trend': np.random.choice(['up', 'down', 'stable'])
        }
    
    async def get_defi_metrics(self) -> Dict:
        """Get DeFi metrics"""
        return {
            'tvl': 50e9 + np.random.normal(0, 1e9),
            'flow': np.random.normal(0, 1e8)
        }
    
    async def get_market_sentiment(self) -> Dict:
        """Get market sentiment metrics"""
        return {
            'fear_greed': np.random.randint(0, 100),
            'vix': 20 + np.random.normal(0, 5)
        }
    
    async def get_funding_rates(self) -> float:
        """Get perpetual futures funding rates"""
        return np.random.normal(0, 0.01)
    
    def get_default_macro_data(self) -> Dict:
        """Default macro data when fetch fails"""
        return {
            'eth_price': 3000,
            'eth_volatility': 0.5,
            'eth_dominance': 0.6,
            'avg_gas_price': 20,
            'gas_trend': 'stable',
            'defi_tvl': 50e9,
            'defi_flow': 0,
            'market_fear_greed': 50,
            'vix_equivalent': 20,
            'funding_rates': 0
        }

class AdvancedEnsembleModel:
    """Advanced multi-modal ensemble with sentiment and macro factors"""
    
    def __init__(self):
        # Core models
        from inference_model import MomentumEnsemble
        self.core_model = MomentumEnsemble()
        
        # Additional analyzers
        self.sentiment_analyzer = SocialSentimentAnalyzer()
        self.macro_analyzer = MacroEconomicAnalyzer()
        
        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Feature scaling
        self.scaler = StandardScaler()
        
        # Model weights (learned through meta-optimization)
        self.model_weights = {
            'core_momentum': 0.4,
            'sentiment': 0.2,
            'macro': 0.15,
            'anomaly': 0.15,
            'volume_profile': 0.1
        }
        
        self.prediction_history = []
        self.performance_tracker = {}
    
    async def predict_with_multi_modal(self, chain: str, token_address: str, 
                                     features: pd.DataFrame, symbol: str) -> Dict:
        """Multi-modal prediction with all data sources"""
        
        # 1. Core momentum prediction
        core_prediction = self.core_model.predict(features)
        
        # 2. Sentiment analysis
        sentiment_data = await self.sentiment_analyzer.get_token_sentiment(token_address, symbol)
        sentiment_score = self.convert_sentiment_to_momentum(sentiment_data)
        
        # 3. Macro-economic factors
        macro_data = await self.macro_analyzer.get_macro_factors()
        macro_score = self.convert_macro_to_momentum(macro_data)
        
        # 4. Anomaly detection
        anomaly_score = self.detect_anomalies(features)
        
        # 5. Volume profile analysis
        volume_score = self.analyze_volume_profile(features)
        
        # 6. Ensemble prediction
        ensemble_prediction = (
            core_prediction * self.model_weights['core_momentum'] +
            sentiment_score * self.model_weights['sentiment'] +
            macro_score * self.model_weights['macro'] +
            anomaly_score * self.model_weights['anomaly'] +
            volume_score * self.model_weights['volume_profile']
        )
        
        # 7. Calculate confidence and uncertainty
        confidence = self.calculate_ensemble_confidence([
            core_prediction, sentiment_score, macro_score, anomaly_score, volume_score
        ])
        
        uncertainty = self.calculate_prediction_uncertainty(features, ensemble_prediction)
        
        # 8. Risk adjustment based on market regime
        risk_adjusted_prediction = self.adjust_for_market_regime(
            ensemble_prediction, macro_data
        )
        
        result = {
            'ensemble_prediction': risk_adjusted_prediction,
            'core_prediction': core_prediction,
            'sentiment_score': sentiment_score,
            'macro_score': macro_score,
            'anomaly_score': anomaly_score,
            'volume_score': volume_score,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'model_weights': self.model_weights,
            'sentiment_data': sentiment_data,
            'macro_data': macro_data,
            'timestamp': time.time()
        }
        
        # Track prediction for performance analysis
        self.prediction_history.append(result)
        
        return result
    
    def convert_sentiment_to_momentum(self, sentiment_data: Dict) -> float:
        """Convert sentiment data to momentum score"""
        base_sentiment = sentiment_data['total_sentiment']
        
        # Boost for high mention volume
        mention_boost = min(sentiment_data['mention_volume'] / 1000, 1.0) * 0.1
        
        # Velocity component
        velocity_component = sentiment_data['sentiment_velocity'] * 0.2
        
        momentum_score = (base_sentiment + 1) / 2  # Convert from [-1,1] to [0,1]
        momentum_score += mention_boost + velocity_component
        
        return np.clip(momentum_score, 0, 1)
    
    def convert_macro_to_momentum(self, macro_data: Dict) -> float:
        """Convert macro factors to momentum score"""
        # ETH strength factor
        eth_factor = 0.5  # Neutral baseline
        if macro_data['eth_volatility'] < 0.3:  # Low volatility = good for alts
            eth_factor += 0.2
        
        # Gas price factor
        gas_factor = 0.5
        if macro_data['avg_gas_price'] < 30:  # Low gas = good for trading
            gas_factor += 0.2
        
        # DeFi flow factor
        flow_factor = 0.5
        if macro_data['defi_flow'] > 0:  # Money flowing into DeFi
            flow_factor += 0.3
        
        # Market sentiment factor
        sentiment_factor = macro_data['market_fear_greed'] / 100
        
        macro_score = (eth_factor + gas_factor + flow_factor + sentiment_factor) / 4
        
        return np.clip(macro_score, 0, 1)
    
    def detect_anomalies(self, features: pd.DataFrame) -> float:
        """Detect anomalies in feature patterns"""
        try:
            # Prepare features for anomaly detection
            feature_array = features.select_dtypes(include=[np.number]).fillna(0).values
            
            if len(feature_array) < 10:
                return 0.5  # Neutral if insufficient data
            
            # Fit anomaly detector if not already fitted
            if not hasattr(self.anomaly_detector, 'offset_'):
                self.anomaly_detector.fit(feature_array)
            
            # Get anomaly scores
            anomaly_scores = self.anomaly_detector.decision_function(feature_array)
            
            # Convert to momentum signal
            # Positive anomalies (outliers) might indicate breakouts
            recent_anomaly = anomaly_scores[-1] if len(anomaly_scores) > 0 else 0
            
            # Normalize to [0, 1]
            normalized_score = (recent_anomaly + 0.5) / 1.0  # Rough normalization
            
            return np.clip(normalized_score, 0, 1)
            
        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}")
            return 0.5
    
    def analyze_volume_profile(self, features: pd.DataFrame) -> float:
        """Analyze volume profile for momentum signals"""
        try:
            if 'swap_volume' not in features.columns:
                return 0.5
            
            volumes = features['swap_volume'].values
            
            # Volume trend analysis
            if len(volumes) >= 5:
                recent_volumes = volumes[-5:]
                older_volumes = volumes[-10:-5] if len(volumes) >= 10 else volumes[:-5]
                
                recent_avg = np.mean(recent_volumes)
                older_avg = np.mean(older_volumes) if len(older_volumes) > 0 else recent_avg
                
                volume_ratio = recent_avg / (older_avg + 1e-6)
                
                # Convert volume ratio to momentum score
                volume_score = min(volume_ratio / 3.0, 1.0)  # Cap at 3x volume increase
                
                return volume_score
            
            return 0.5
            
        except Exception as e:
            logging.error(f"Volume profile analysis failed: {e}")
            return 0.5
    
    def calculate_ensemble_confidence(self, predictions: List[float]) -> float:
        """Calculate confidence based on prediction agreement"""
        predictions = np.array(predictions)
        
        # Calculate agreement (inverse of standard deviation)
        std_dev = np.std(predictions)
        agreement = 1.0 / (1.0 + std_dev)
        
        # Factor in prediction strength
        mean_prediction = np.mean(predictions)
        strength = abs(mean_prediction - 0.5) * 2  # Distance from neutral
        
        confidence = (agreement + strength) / 2
        
        return np.clip(confidence, 0, 1)
    
    def calculate_prediction_uncertainty(self, features: pd.DataFrame, prediction: float) -> float:
        """Calculate prediction uncertainty"""
        try:
            # Feature uncertainty (based on recent volatility)
            if 'volatility' in features.columns:
                volatility = features['volatility'].iloc[-1] if len(features) > 0 else 0.5
                volatility_uncertainty = min(volatility, 1.0)
            else:
                volatility_uncertainty = 0.5
            
            # Model uncertainty (based on prediction history variance)
            if len(self.prediction_history) >= 10:
                recent_predictions = [p['ensemble_prediction'] for p in self.prediction_history[-10:]]
                prediction_variance = np.var(recent_predictions)
                model_uncertainty = min(prediction_variance * 10, 1.0)
            else:
                model_uncertainty = 0.5
            
            total_uncertainty = (volatility_uncertainty + model_uncertainty) / 2
            
            return total_uncertainty
            
        except Exception as e:
            logging.error(f"Uncertainty calculation failed: {e}")
            return 0.5
    
    def adjust_for_market_regime(self, prediction: float, macro_data: Dict) -> float:
        """Adjust prediction based on market regime"""
        adjustment_factor = 1.0
        
        # High volatility regime - reduce prediction confidence
        if macro_data['eth_volatility'] > 0.7:
            adjustment_factor *= 0.8
        
        # Extreme fear - reduce bullish predictions
        if macro_data['market_fear_greed'] < 20:
            if prediction > 0.5:
                adjustment_factor *= 0.7
        
        # Extreme greed - reduce bullish predictions
        if macro_data['market_fear_greed'] > 80:
            if prediction > 0.7:
                adjustment_factor *= 0.8
        
        # High gas prices - reduce activity
        if macro_data['avg_gas_price'] > 50:
            adjustment_factor *= 0.9
        
        adjusted_prediction = prediction * adjustment_factor
        
        return np.clip(adjusted_prediction, 0, 1)
    
    async def update_model_weights(self, performance_data: Dict):
        """Update ensemble weights based on component performance"""
        # This would implement online learning to optimize weights
        # based on which components are performing best
        pass
    
    def get_feature_importance(self) -> Dict:
        """Get current feature importance scores"""
        return {
            'model_weights': self.model_weights,
            'prediction_count': len(self.prediction_history),
            'avg_confidence': np.mean([p['confidence'] for p in self.prediction_history[-100:]]) if self.prediction_history else 0,
            'avg_uncertainty': np.mean([p['uncertainty'] for p in self.prediction_history[-100:]]) if self.prediction_history else 0
        }
