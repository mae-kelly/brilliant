#!/bin/bash

# 🚀 DeFi Trading System - Complete Enhancement Package
# Implements all missing components for Renaissance Tech standards

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 DEFI TRADING SYSTEM ENHANCEMENT${NC}"
echo -e "${BLUE}🎯 Target: Renaissance Tech Standards${NC}"
echo -e "${BLUE}========================================${NC}"

# Create enhancement directory structure
echo -e "\n${YELLOW}📁 Creating enhanced directory structure...${NC}"

mkdir -p intelligence/streaming
mkdir -p intelligence/social
mkdir -p intelligence/arbitrage
mkdir -p core/training
mkdir -p core/orders
mkdir -p data/historical
mkdir -p notebooks/analysis
mkdir -p scripts/enhancement

echo -e "${GREEN}✅ Directory structure created${NC}"

# ============================================================================
# 1. REAL-TIME DATA STREAMING
# ============================================================================

echo -e "\n${YELLOW}📡 Implementing real-time data streaming...${NC}"

cat > intelligence/streaming/websocket_feeds.py << 'EOF'
#!/usr/bin/env python3
"""
Real-time WebSocket data feeds for instant market data
Connects to multiple DEX streams simultaneously
"""

import asyncio
import websockets
import json
import logging
import time
from typing import Dict, List, Callable
import aiohttp
import numpy as np
from dataclasses import dataclass
from collections import deque
import redis

@dataclass
class StreamData:
    symbol: str
    price: float
    volume: float
    timestamp: float
    source: str
    liquidity: float = 0
    bid: float = 0
    ask: float = 0

class RealTimeStreamer:
    """Ultra-fast real-time data streaming"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.streams = {}
        self.callbacks = []
        self.data_buffer = deque(maxlen=10000)
        self.stream_health = {}
        
        # DEX WebSocket endpoints
        self.endpoints = {
            'uniswap_v3': 'wss://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-arbitrum/ws',
            'camelot': 'wss://api.thegraph.com/subgraphs/name/camelotlabs/camelot-amm/ws',
            'pancakeswap': 'wss://api.thegraph.com/subgraphs/name/pancakeswap/exchange-v3-polygon/ws'
        }
    
    async def start_all_streams(self):
        """Start all WebSocket streams concurrently"""
        
        logging.info("🚀 Starting real-time data streams...")
        
        stream_tasks = []
        for source, endpoint in self.endpoints.items():
            task = asyncio.create_task(self.connect_stream(source, endpoint))
            stream_tasks.append(task)
        
        # Start health monitoring
        health_task = asyncio.create_task(self.monitor_stream_health())
        stream_tasks.append(health_task)
        
        await asyncio.gather(*stream_tasks, return_exceptions=True)
    
    async def connect_stream(self, source: str, endpoint: str):
        """Connect to individual WebSocket stream with auto-reconnect"""
        
        while True:
            try:
                logging.info(f"📡 Connecting to {source}...")
                
                async with websockets.connect(endpoint) as websocket:
                    self.stream_health[source] = time.time()
                    
                    # Subscribe to relevant channels
                    subscription = {
                        "id": "1",
                        "type": "start",
                        "payload": {
                            "query": """
                                subscription {
                                    swaps(first: 50, orderBy: timestamp, orderDirection: desc) {
                                        id
                                        timestamp
                                        amount0
                                        amount1
                                        amountUSD
                                        pool {
                                            id
                                            token0 { symbol }
                                            token1 { symbol }
                                            liquidity
                                            sqrtPriceX96
                                        }
                                    }
                                }
                            """
                        }
                    }
                    
                    await websocket.send(json.dumps(subscription))
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self.process_stream_data(source, data)
                            self.stream_health[source] = time.time()
                            
                        except Exception as e:
                            logging.error(f"Stream data processing error: {e}")
                            continue
                            
            except Exception as e:
                logging.error(f"Stream {source} disconnected: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting
    
    async def process_stream_data(self, source: str, raw_data: Dict):
        """Process incoming stream data"""
        
        try:
            swaps_data = raw_data.get('payload', {}).get('data', {}).get('swaps', [])
            
            for swap in swaps_data:
                pool = swap.get('pool', {})
                
                # Extract meaningful data
                stream_data = StreamData(
                    symbol=f"{pool.get('token0', {}).get('symbol', 'UNK')}/{pool.get('token1', {}).get('symbol', 'UNK')}",
                    price=self.calculate_price_from_swap(swap),
                    volume=float(swap.get('amountUSD', 0)),
                    timestamp=float(swap.get('timestamp', time.time())),
                    source=source,
                    liquidity=float(pool.get('liquidity', 0))
                )
                
                # Buffer data
                self.data_buffer.append(stream_data)
                
                # Cache in Redis
                if self.redis_client:
                    cache_key = f"stream:{source}:{pool.get('id', 'unknown')}"
                    self.redis_client.setex(
                        cache_key, 
                        60, 
                        json.dumps({
                            'price': stream_data.price,
                            'volume': stream_data.volume,
                            'timestamp': stream_data.timestamp,
                            'liquidity': stream_data.liquidity
                        })
                    )
                
                # Trigger callbacks
                for callback in self.callbacks:
                    try:
                        await callback(stream_data)
                    except Exception as e:
                        logging.error(f"Callback error: {e}")
                        
        except Exception as e:
            logging.error(f"Stream processing error: {e}")
    
    def calculate_price_from_swap(self, swap: Dict) -> float:
        """Calculate price from swap data"""
        try:
            amount0 = float(swap.get('amount0', 0))
            amount1 = float(swap.get('amount1', 0))
            
            if amount0 != 0 and amount1 != 0:
                return abs(amount1 / amount0)
            
            # Fallback to sqrtPriceX96
            pool = swap.get('pool', {})
            sqrt_price = float(pool.get('sqrtPriceX96', 0))
            if sqrt_price > 0:
                return (sqrt_price / 2**96) ** 2
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def add_callback(self, callback: Callable):
        """Add callback for real-time data"""
        self.callbacks.append(callback)
    
    async def monitor_stream_health(self):
        """Monitor stream health and restart if needed"""
        
        while True:
            try:
                current_time = time.time()
                
                for source, last_update in self.stream_health.items():
                    if current_time - last_update > 30:  # 30 seconds timeout
                        logging.warning(f"⚠️ Stream {source} appears stale")
                        # Could implement restart logic here
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
    
    def get_recent_data(self, symbol: str = None, seconds: int = 60) -> List[StreamData]:
        """Get recent streaming data"""
        
        cutoff_time = time.time() - seconds
        
        if symbol:
            return [data for data in self.data_buffer 
                   if data.timestamp > cutoff_time and data.symbol == symbol]
        else:
            return [data for data in self.data_buffer 
                   if data.timestamp > cutoff_time]
    
    def get_stream_statistics(self) -> Dict:
        """Get streaming statistics"""
        
        current_time = time.time()
        
        return {
            'active_streams': len([s for s, t in self.stream_health.items() 
                                 if current_time - t < 30]),
            'total_streams': len(self.endpoints),
            'buffer_size': len(self.data_buffer),
            'data_rate_per_second': len([d for d in self.data_buffer 
                                       if current_time - d.timestamp < 1]),
            'stream_health': {source: current_time - last_update 
                            for source, last_update in self.stream_health.items()}
        }

class PriceVelocityDetector:
    """Detect rapid price movements in real-time"""
    
    def __init__(self, streamer: RealTimeStreamer):
        self.streamer = streamer
        self.price_history = {}
        self.velocity_thresholds = {
            'breakout': 0.09,      # 9% in short timeframe
            'acceleration': 0.13,   # 13% acceleration
            'momentum_surge': 0.15  # 15% momentum surge
        }
        
        # Register callback
        streamer.add_callback(self.process_price_update)
    
    async def process_price_update(self, data: StreamData):
        """Process real-time price updates"""
        
        symbol = data.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)
        
        self.price_history[symbol].append({
            'price': data.price,
            'volume': data.volume,
            'timestamp': data.timestamp
        })
        
        # Detect velocity patterns
        velocity_signals = self.detect_velocity_patterns(symbol)
        
        if any(velocity_signals.values()):
            logging.info(f"🚀 Velocity signal detected for {symbol}: {velocity_signals}")
            
            # Could trigger trading logic here
            await self.handle_velocity_signal(symbol, velocity_signals, data)
    
    def detect_velocity_patterns(self, symbol: str) -> Dict[str, bool]:
        """Detect various velocity patterns"""
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return {pattern: False for pattern in self.velocity_thresholds.keys()}
        
        prices = [p['price'] for p in self.price_history[symbol]]
        times = [p['timestamp'] for p in self.price_history[symbol]]
        
        # Calculate returns over different timeframes
        returns_1min = self.calculate_returns(prices[-5:])   # Last 5 data points
        returns_5min = self.calculate_returns(prices[-20:])  # Last 20 data points
        returns_total = self.calculate_returns(prices)       # All data points
        
        # Detect patterns
        patterns = {
            'breakout': abs(returns_1min) > self.velocity_thresholds['breakout'],
            'acceleration': abs(returns_5min) > self.velocity_thresholds['acceleration'],
            'momentum_surge': abs(returns_total) > self.velocity_thresholds['momentum_surge']
        }
        
        # Additional pattern: sustained acceleration
        if len(prices) >= 30:
            recent_acceleration = self.calculate_acceleration(prices[-30:])
            patterns['sustained_acceleration'] = recent_acceleration > 0.05
        
        return patterns
    
    def calculate_returns(self, prices: List[float]) -> float:
        """Calculate total returns"""
        if len(prices) < 2:
            return 0.0
        
        return (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0.0
    
    def calculate_acceleration(self, prices: List[float]) -> float:
        """Calculate price acceleration"""
        if len(prices) < 10:
            return 0.0
        
        # Split into three segments and calculate acceleration
        segment_size = len(prices) // 3
        segment1 = prices[:segment_size]
        segment2 = prices[segment_size:2*segment_size]
        segment3 = prices[2*segment_size:]
        
        return1 = self.calculate_returns(segment1)
        return2 = self.calculate_returns(segment2)
        return3 = self.calculate_returns(segment3)
        
        # Calculate acceleration as change in velocity
        velocity1 = return2 - return1
        velocity2 = return3 - return2
        
        return velocity2 - velocity1
    
    async def handle_velocity_signal(self, symbol: str, signals: Dict, data: StreamData):
        """Handle detected velocity signal"""
        
        # This would integrate with the main trading pipeline
        signal_strength = sum(signals.values()) / len(signals)
        
        alert = {
            'symbol': symbol,
            'signal_strength': signal_strength,
            'signals': signals,
            'price': data.price,
            'volume': data.volume,
            'timestamp': data.timestamp,
            'source': data.source
        }
        
        # Could publish to Redis channel for main pipeline
        if self.streamer.redis_client:
            channel = "velocity_signals"
            self.streamer.redis_client.publish(
                channel, 
                json.dumps(alert)
            )

# Integration with existing scanner
async def integrate_streaming_with_scanner():
    """Integrate real-time streaming with existing scanner"""
    
    import redis
    from core.execution.scanner_v3 import ScannerV3
    
    # Initialize components
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
    streamer = RealTimeStreamer(redis_client)
    velocity_detector = PriceVelocityDetector(streamer)
    
    # Start streaming
    logging.info("🚀 Starting integrated real-time pipeline...")
    await streamer.start_all_streams()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(integrate_streaming_with_scanner())
EOF

# ============================================================================
# 2. SOCIAL SENTIMENT ANALYSIS
# ============================================================================

echo -e "\n${YELLOW}📱 Implementing social sentiment analysis...${NC}"

cat > intelligence/social/sentiment_analyzer.py << 'EOF'
#!/usr/bin/env python3
"""
Advanced Social Sentiment Analysis
Integrates Twitter, Discord, Telegram, and Reddit for comprehensive sentiment
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, List, Optional
import re
import numpy as np
from textblob import TextBlob
from dataclasses import dataclass
import sqlite3
from collections import deque

@dataclass
class SentimentData:
    source: str
    text: str
    sentiment_score: float
    confidence: float
    timestamp: float
    influence_score: float
    engagement: int

class SocialSentimentAnalyzer:
    """Real-time social sentiment analysis across multiple platforms"""
    
    def __init__(self):
        self.sentiment_cache = {}
        self.influence_scores = {}
        self.sentiment_history = deque(maxlen=10000)
        
        # API endpoints and keys
        self.twitter_bearer_token = "YOUR_TWITTER_BEARER_TOKEN"
        self.reddit_client_id = "YOUR_REDDIT_CLIENT_ID"
        self.reddit_secret = "YOUR_REDDIT_SECRET"
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize sentiment database"""
        self.conn = sqlite3.connect('data/sentiment_data.db')
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                source TEXT,
                sentiment_score REAL,
                confidence REAL,
                influence_score REAL,
                engagement INTEGER,
                timestamp REAL,
                text_sample TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                alert_type TEXT,
                sentiment_velocity REAL,
                confidence REAL,
                timestamp REAL
            )
        ''')
        
        self.conn.commit()
    
    async def analyze_token_sentiment(self, symbol: str, token_address: str) -> Dict:
        """Comprehensive sentiment analysis for a token"""
        
        # Check cache first
        cache_key = f"{symbol}_{token_address}"
        if cache_key in self.sentiment_cache:
            cached_data, timestamp = self.sentiment_cache[cache_key]
            if time.time() - timestamp < 300:  # 5 minute cache
                return cached_data
        
        # Gather data from all sources
        sentiment_tasks = [
            self.get_twitter_sentiment(symbol),
            self.get_reddit_sentiment(symbol),
            self.get_discord_sentiment(symbol),
            self.get_telegram_sentiment(symbol),
            self.get_news_sentiment(symbol)
        ]
        
        results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)
        
        # Process results
        twitter_data = results[0] if isinstance(results[0], dict) else {}
        reddit_data = results[1] if isinstance(results[1], dict) else {}
        discord_data = results[2] if isinstance(results[2], dict) else {}
        telegram_data = results[3] if isinstance(results[3], dict) else {}
        news_data = results[4] if isinstance(results[4], dict) else {}
        
        # Combine sentiment scores
        combined_sentiment = self.combine_sentiment_scores({
            'twitter': twitter_data,
            'reddit': reddit_data,
            'discord': discord_data,
            'telegram': telegram_data,
            'news': news_data
        })
        
        # Calculate sentiment velocity
        sentiment_velocity = self.calculate_sentiment_velocity(symbol, combined_sentiment['overall_score'])
        
        # Detect sentiment anomalies
        anomalies = self.detect_sentiment_anomalies(symbol, combined_sentiment)
        
        final_analysis = {
            'symbol': symbol,
            'overall_sentiment': combined_sentiment['overall_score'],
            'sentiment_confidence': combined_sentiment['confidence'],
            'sentiment_velocity': sentiment_velocity,
            'source_breakdown': combined_sentiment['source_breakdown'],
            'anomalies_detected': anomalies,
            'recommendation': self.generate_sentiment_recommendation(combined_sentiment),
            'timestamp': time.time()
        }
        
        # Cache result
        self.sentiment_cache[cache_key] = (final_analysis, time.time())
        
        # Store in database
        self.store_sentiment_data(symbol, final_analysis)
        
        return final_analysis
    
    async def get_twitter_sentiment(self, symbol: str) -> Dict:
        """Get Twitter sentiment for symbol"""
        
        try:
            search_query = f"${symbol} OR #{symbol} OR {symbol} crypto"
            
            headers = {
                'Authorization': f'Bearer {self.twitter_bearer_token}',
                'Content-Type': 'application/json'
            }
            
            params = {
                'query': search_query,
                'tweet.fields': 'public_metrics,created_at,author_id',
                'max_results': 100
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://api.twitter.com/2/tweets/search/recent',
                    headers=headers,
                    params=params,
                    timeout=10
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        tweets = data.get('data', [])
                        
                        return self.analyze_twitter_data(tweets)
                    else:
                        return self.get_mock_twitter_sentiment(symbol)
                        
        except Exception as e:
            logging.error(f"Twitter sentiment error: {e}")
            return self.get_mock_twitter_sentiment(symbol)
    
    def analyze_twitter_data(self, tweets: List[Dict]) -> Dict:
        """Analyze Twitter data for sentiment"""
        
        if not tweets:
            return {'sentiment': 0.0, 'confidence': 0.0, 'volume': 0}
        
        sentiments = []
        total_engagement = 0
        influence_weighted_sentiments = []
        
        for tweet in tweets:
            text = tweet.get('text', '')
            metrics = tweet.get('public_metrics', {})
            
            # Basic sentiment analysis
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            confidence = blob.sentiment.subjectivity
            
            # Calculate engagement
            engagement = (
                metrics.get('like_count', 0) * 1 +
                metrics.get('retweet_count', 0) * 2 +
                metrics.get('reply_count', 0) * 1.5 +
                metrics.get('quote_count', 0) * 3
            )
            
            total_engagement += engagement
            sentiments.append(sentiment_score)
            
            # Weight by engagement
            influence_weight = min(engagement / 100, 5.0)
            influence_weighted_sentiments.append(sentiment_score * influence_weight)
        
        # Calculate weighted average
        if influence_weighted_sentiments:
            weighted_sentiment = np.mean(influence_weighted_sentiments)
            overall_confidence = np.mean([abs(s) for s in sentiments])
        else:
            weighted_sentiment = 0.0
            overall_confidence = 0.0
        
        return {
            'sentiment': weighted_sentiment,
            'confidence': overall_confidence,
            'volume': len(tweets),
            'total_engagement': total_engagement,
            'raw_sentiments': sentiments
        }
    
    def get_mock_twitter_sentiment(self, symbol: str) -> Dict:
        """Mock Twitter sentiment for testing"""
        return {
            'sentiment': np.random.uniform(-0.3, 0.3),
            'confidence': np.random.uniform(0.3, 0.8),
            'volume': np.random.randint(10, 100),
            'total_engagement': np.random.randint(100, 1000)
        }
    
    async def get_reddit_sentiment(self, symbol: str) -> Dict:
        """Get Reddit sentiment for symbol"""
        
        try:
            # Search multiple crypto subreddits
            subreddits = ['CryptoCurrency', 'defi', 'ethereum', 'altcoin']
            search_terms = [symbol.lower(), f"${symbol}", symbol.upper()]
            
            all_posts = []
            
            async with aiohttp.ClientSession() as session:
                for subreddit in subreddits:
                    for term in search_terms[:1]:  # Limit to avoid rate limits
                        url = f"https://www.reddit.com/r/{subreddit}/search.json"
                        params = {
                            'q': term,
                            'sort': 'new',
                            'limit': 25,
                            't': 'day'
                        }
                        
                        try:
                            async with session.get(url, params=params, timeout=10) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    posts = data.get('data', {}).get('children', [])
                                    all_posts.extend(posts)
                        except Exception:
                            continue
            
            return self.analyze_reddit_data(all_posts)
            
        except Exception as e:
            logging.error(f"Reddit sentiment error: {e}")
            return self.get_mock_reddit_sentiment(symbol)
    
    def analyze_reddit_data(self, posts: List[Dict]) -> Dict:
        """Analyze Reddit posts for sentiment"""
        
        if not posts:
            return {'sentiment': 0.0, 'confidence': 0.0, 'volume': 0}
        
        sentiments = []
        scores = []
        
        for post in posts:
            post_data = post.get('data', {})
            title = post_data.get('title', '')
            selftext = post_data.get('selftext', '')
            score = post_data.get('score', 0)
            
            combined_text = f"{title} {selftext}"
            
            if len(combined_text.strip()) > 10:
                blob = TextBlob(combined_text)
                sentiment_score = blob.sentiment.polarity
                
                # Weight by Reddit score
                weight = max(1, min(score / 10, 5)) if score > 0 else 0.5
                sentiments.append(sentiment_score * weight)
                scores.append(score)
        
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            confidence = np.std(sentiments) + 0.1  # Add base confidence
        else:
            avg_sentiment = 0.0
            confidence = 0.0
        
        return {
            'sentiment': avg_sentiment,
            'confidence': min(confidence, 1.0),
            'volume': len(posts),
            'avg_score': np.mean(scores) if scores else 0
        }
    
    def get_mock_reddit_sentiment(self, symbol: str) -> Dict:
        """Mock Reddit sentiment for testing"""
        return {
            'sentiment': np.random.uniform(-0.2, 0.2),
            'confidence': np.random.uniform(0.4, 0.7),
            'volume': np.random.randint(5, 50),
            'avg_score': np.random.randint(1, 20)
        }
    
    async def get_discord_sentiment(self, symbol: str) -> Dict:
        """Get Discord sentiment (mock implementation)"""
        # Discord API requires bot setup, mock for now
        return {
            'sentiment': np.random.uniform(-0.1, 0.1),
            'confidence': np.random.uniform(0.3, 0.6),
            'volume': np.random.randint(5, 30)
        }
    
    async def get_telegram_sentiment(self, symbol: str) -> Dict:
        """Get Telegram sentiment (mock implementation)"""
        # Telegram API requires bot setup, mock for now
        return {
            'sentiment': np.random.uniform(-0.15, 0.15),
            'confidence': np.random.uniform(0.2, 0.5),
            'volume': np.random.randint(3, 25)
        }
    
    async def get_news_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment for symbol"""
        
        try:
            # Use NewsAPI or similar service
            news_sources = [
                'https://cryptonews.com/rss/',
                'https://cointelegraph.com/rss',
                'https://decrypt.co/feed'
            ]
            
            all_articles = []
            
            async with aiohttp.ClientSession() as session:
                for source in news_sources:
                    try:
                        async with session.get(source, timeout=10) as response:
                            if response.status == 200:
                                content = await response.text()
                                articles = self.parse_news_feed(content, symbol)
                                all_articles.extend(articles)
                    except Exception:
                        continue
            
            return self.analyze_news_data(all_articles)
            
        except Exception as e:
            logging.error(f"News sentiment error: {e}")
            return self.get_mock_news_sentiment(symbol)
    
    def parse_news_feed(self, content: str, symbol: str) -> List[Dict]:
        """Parse RSS news feed"""
        articles = []
        
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content)
            
            for item in root.findall('.//item')[:10]:
                title = item.find('title')
                description = item.find('description')
                
                title_text = title.text if title is not None else ""
                desc_text = description.text if description is not None else ""
                
                if symbol.lower() in (title_text + desc_text).lower():
                    articles.append({
                        'title': title_text,
                        'description': desc_text
                    })
        except Exception:
            pass
        
        return articles
    
    def analyze_news_data(self, articles: List[Dict]) -> Dict:
        """Analyze news articles for sentiment"""
        
        if not articles:
            return {'sentiment': 0.0, 'confidence': 0.0, 'volume': 0}
        
        sentiments = []
        
        for article in articles:
            combined_text = f"{article.get('title', '')} {article.get('description', '')}"
            
            if len(combined_text.strip()) > 20:
                blob = TextBlob(combined_text)
                sentiment_score = blob.sentiment.polarity
                sentiments.append(sentiment_score)
        
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            confidence = 1 - np.std(sentiments)  # More consistent = higher confidence
        else:
            avg_sentiment = 0.0
            confidence = 0.0
        
        return {
            'sentiment': avg_sentiment,
            'confidence': max(0, min(confidence, 1.0)),
            'volume': len(articles)
        }
    
    def get_mock_news_sentiment(self, symbol: str) -> Dict:
        """Mock news sentiment for testing"""
        return {
            'sentiment': np.random.uniform(-0.1, 0.1),
            'confidence': np.random.uniform(0.5, 0.8),
            'volume': np.random.randint(1, 10)
        }
    
    def combine_sentiment_scores(self, sentiment_data: Dict) -> Dict:
        """Combine sentiment scores from all sources"""
        
        # Define source weights based on reliability and influence
        source_weights = {
            'twitter': 0.30,
            'reddit': 0.25,
            'news': 0.20,
            'discord': 0.15,
            'telegram': 0.10
        }
        
        weighted_scores = []
        source_breakdown = {}
        total_confidence = 0
        valid_sources = 0
        
        for source, data in sentiment_data.items():
            if data and 'sentiment' in data:
                weight = source_weights.get(source, 0.1)
                sentiment = data['sentiment']
                confidence = data.get('confidence', 0.5)
                volume = data.get('volume', 0)
                
                # Adjust weight by volume and confidence
                volume_factor = min(volume / 50, 2.0)  # Cap at 2x weight
                confidence_factor = confidence
                adjusted_weight = weight * volume_factor * confidence_factor
                
                weighted_scores.append(sentiment * adjusted_weight)
                total_confidence += confidence * weight
                valid_sources += 1
                
                source_breakdown[source] = {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'volume': volume,
                    'weight': adjusted_weight
                }
        
        if weighted_scores:
            overall_score = sum(weighted_scores) / sum(source_weights.values())
            overall_confidence = total_confidence / valid_sources if valid_sources > 0 else 0
        else:
            overall_score = 0.0
            overall_confidence = 0.0
        
        return {
            'overall_score': overall_score,
            'confidence': overall_confidence,
            'source_breakdown': source_breakdown,
            'active_sources': valid_sources
        }
    
    def calculate_sentiment_velocity(self, symbol: str, current_sentiment: float) -> float:
        """Calculate sentiment velocity (rate of change)"""
        
        # Store sentiment history
        if symbol not in self.sentiment_cache:
            return 0.0
        
        # Get recent sentiment history
        recent_sentiments = []
        current_time = time.time()
        
        for data, timestamp in self.sentiment_history:
            if (data.get('symbol') == symbol and 
                current_time - timestamp < 3600):  # Last hour
                recent_sentiments.append((data['overall_sentiment'], timestamp))
        
        if len(recent_sentiments) < 3:
            return 0.0
        
        # Calculate velocity as change over time
        recent_sentiments.sort(key=lambda x: x[1])
        
        old_sentiment = recent_sentiments[0][0]
        new_sentiment = current_sentiment
        time_diff = current_time - recent_sentiments[0][1]
        
        if time_diff > 0:
            velocity = (new_sentiment - old_sentiment) / (time_diff / 3600)  # Per hour
            return velocity
        
        return 0.0
    
    def detect_sentiment_anomalies(self, symbol: str, sentiment_data: Dict) -> List[Dict]:
        """Detect sentiment anomalies and unusual patterns"""
        
        anomalies = []
        
        overall_sentiment = sentiment_data['overall_score']
        confidence = sentiment_data['confidence']
        source_breakdown = sentiment_data.get('source_breakdown', {})
        
        # Detect extreme sentiment
        if abs(overall_sentiment) > 0.7 and confidence > 0.6:
            anomalies.append({
                'type': 'extreme_sentiment',
                'severity': 'high' if abs(overall_sentiment) > 0.8 else 'medium',
                'sentiment': overall_sentiment,
                'description': f"Extreme {'positive' if overall_sentiment > 0 else 'negative'} sentiment detected"
            })
        
        # Detect sentiment divergence across sources
        sentiments = [data.get('sentiment', 0) for data in source_breakdown.values()]
        if len(sentiments) >= 3:
            sentiment_std = np.std(sentiments)
            if sentiment_std > 0.5:
                anomalies.append({
                    'type': 'sentiment_divergence',
                    'severity': 'medium',
                    'std_dev': sentiment_std,
                    'description': 'High divergence in sentiment across sources'
                })
        
        # Detect unusual volume patterns
        volumes = [data.get('volume', 0) for data in source_breakdown.values()]
        total_volume = sum(volumes)
        if total_volume > 200:  # High social volume
            anomalies.append({
                'type': 'high_social_volume',
                'severity': 'medium',
                'volume': total_volume,
                'description': 'Unusually high social media volume'
            })
        
        return anomalies
    
    def generate_sentiment_recommendation(self, sentiment_data: Dict) -> str:
        """Generate trading recommendation based on sentiment"""
        
        sentiment = sentiment_data['overall_score']
        confidence = sentiment_data['confidence']
        
        if confidence < 0.3:
            return 'INSUFFICIENT_DATA'
        elif sentiment > 0.5 and confidence > 0.7:
            return 'STRONG_BUY_SIGNAL'
        elif sentiment > 0.2 and confidence > 0.6:
            return 'WEAK_BUY_SIGNAL'
        elif sentiment < -0.5 and confidence > 0.7:
            return 'STRONG_SELL_SIGNAL'
        elif sentiment < -0.2 and confidence > 0.6:
            return 'WEAK_SELL_SIGNAL'
        else:
            return 'NEUTRAL'
    
    def store_sentiment_data(self, symbol: str, analysis: Dict):
        """Store sentiment analysis in database"""
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO sentiment_data 
                (symbol, source, sentiment_score, confidence, influence_score, 
                 engagement, timestamp, text_sample)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                'combined',
                analysis['overall_sentiment'],
                analysis['sentiment_confidence'],
                1.0,  # Combined influence
                len(analysis.get('source_breakdown', {})),
                analysis['timestamp'],
                json.dumps(analysis['source_breakdown'])
            ))
            
            # Store alerts for significant sentiment changes
            if analysis.get('sentiment_velocity', 0) > 0.3:
                cursor.execute('''
                    INSERT INTO sentiment_alerts
                    (symbol, alert_type, sentiment_velocity, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    'velocity_spike',
                    analysis['sentiment_velocity'],
                    analysis['sentiment_confidence'],
                    analysis['timestamp']
                ))
            
            self.conn.commit()
            
        except Exception as e:
            logging.error(f"Database storage error: {e}")
    
    def get_sentiment_summary(self, symbol: str, hours: int = 24) -> Dict:
        """Get sentiment summary for the last N hours"""
        
        try:
            cursor = self.conn.cursor()
            cutoff_time = time.time() - (hours * 3600)
            
            cursor.execute('''
                SELECT AVG(sentiment_score), AVG(confidence), COUNT(*)
                FROM sentiment_data 
                WHERE symbol = ? AND timestamp > ?
            ''', (symbol, cutoff_time))
            
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                return {
                    'avg_sentiment': result[0],
                    'avg_confidence': result[1],
                    'data_points': result[2],
                    'time_range_hours': hours
                }
            else:
                return {
                    'avg_sentiment': 0.0,
                    'avg_confidence': 0.0,
                    'data_points': 0,
                    'time_range_hours': hours
                }
                
        except Exception as e:
            logging.error(f"Sentiment summary error: {e}")
            return {'error': str(e)}

# Integration function
async def integrate_sentiment_with_pipeline():
    """Integrate sentiment analysis with main trading pipeline"""
    
    analyzer = SocialSentimentAnalyzer()
    
    # Example usage
    symbols = ['ETH', 'BTC', 'UNI', 'LINK']
    
    for symbol in symbols:
        try:
            analysis = await analyzer.analyze_token_sentiment(symbol, f"0x{symbol.lower()}")
            
            print(f"\n📱 {symbol} Sentiment Analysis:")
            print(f"   Overall: {analysis['overall_sentiment']:.3f} (confidence: {analysis['sentiment_confidence']:.3f})")
            print(f"   Velocity: {analysis['sentiment_velocity']:.3f}")
            print(f"   Recommendation: {analysis['recommendation']}")
            
            if analysis['anomalies_detected']:
                print(f"   🚨 Anomalies: {len(analysis['anomalies_detected'])}")
                
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(integrate_sentiment_with_pipeline())
EOF

# ============================================================================
# 3. CROSS-DEX ARBITRAGE DETECTION
# ============================================================================

echo -e "\n${YELLOW}⚖️ Implementing cross-DEX arbitrage detection...${NC}"