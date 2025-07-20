import aiohttp
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"))
try:
    from dynamic_parameters import get_dynamic_config, update_performance
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
import tweepy
import praw
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import json
import re
from textblob import TextBlob
import numpy as np

@dataclass
class SocialSignal:
    platform: str
    mention_count: int
    sentiment_score: float
    volume_spike: float
    influence_score: float
    timestamp: float

class RealSocialSentimentAnalyzer:
    def __init__(self):
        self.twitter_api = self.init_twitter()
        self.reddit_api = self.init_reddit()
        self.telegram_channels = [
            '@uniswap_official', '@arbitrum', '@0xPolygon',
            '@CryptoPumpSignals', '@DefiPulse', '@thedefiedge'
        ]
        
        self.sentiment_cache = {}
        self.influence_weights = {
            'twitter': 0.4,
            'reddit': 0.3,
            'telegram': 0.2,
            'discord': 0.1
        }
        
        self.session = None
        
    def init_twitter(self):
        try:
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            if not bearer_token:
                return None
            
            return tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
        except:
            return None
    
    def init_reddit(self):
        try:
            return praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent='DeFiScanner/1.0'
            )
        except:
            return None
    
    async def initialize(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={'User-Agent': 'DeFiScanner/1.0'}
        )
    
    async def analyze_token_sentiment(self, token_address: str, token_symbol: str) -> SocialSignal:
        twitter_data = await self.get_twitter_sentiment(token_symbol)
        reddit_data = await self.get_reddit_sentiment(token_symbol)
        telegram_data = await self.get_telegram_sentiment(token_symbol)
        
        combined_sentiment = (
            twitter_data['sentiment'] * self.influence_weights['twitter'] +
            reddit_data['sentiment'] * self.influence_weights['reddit'] +
            telegram_data['sentiment'] * self.influence_weights['telegram']
        )
        
        total_mentions = twitter_data['mentions'] + reddit_data['mentions'] + telegram_data['mentions']
        
        volume_spike = self.calculate_volume_spike(token_symbol, total_mentions)
        influence_score = self.calculate_influence_score(twitter_data, reddit_data, telegram_data)
        
        return SocialSignal(
            platform='combined',
            mention_count=total_mentions,
            sentiment_score=combined_sentiment,
            volume_spike=volume_spike,
            influence_score=influence_score,
            timestamp=time.time()
        )
    
    async def get_twitter_sentiment(self, token_symbol: str) -> Dict:
        if not self.twitter_api:
            return {'sentiment': 0.5, 'mentions': 0}
        
        try:
            query = f"#{token_symbol} OR ${token_symbol} -is:retweet lang:en"
            tweets = self.twitter_api.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            )
            
            if not tweets.data:
                return {'sentiment': 0.5, 'mentions': 0}
            
            sentiments = []
            total_engagement = 0
            
            for tweet in tweets.data:
                text_sentiment = TextBlob(tweet.text).sentiment.polarity
                sentiments.append(text_sentiment)
                
                if tweet.public_metrics:
                    engagement = (tweet.public_metrics['like_count'] + 
                                tweet.public_metrics['retweet_count'] + 
                                tweet.public_metrics['reply_count'])
                    total_engagement += engagement
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            normalized_sentiment = (avg_sentiment + 1) / 2
            
            return {
                'sentiment': normalized_sentiment,
                'mentions': len(tweets.data),
                'engagement': total_engagement
            }
            
        except Exception as e:
            return {'sentiment': 0.5, 'mentions': 0}
    
    async def get_reddit_sentiment(self, token_symbol: str) -> Dict:
        if not self.reddit_api:
            return {'sentiment': 0.5, 'mentions': 0}
        
        try:
            subreddits = ['CryptoCurrency', 'defi', 'ethtrader', 'CryptoMoonShots', 'altcoin']
            all_posts = []
            
            for subreddit_name in subreddits:
                subreddit = self.reddit_api.subreddit(subreddit_name)
                posts = subreddit.search(token_symbol, limit=50, time_filter='day')
                
                for post in posts:
                    all_posts.append({
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'comments': post.num_comments
                    })
            
            if not all_posts:
                return {'sentiment': 0.5, 'mentions': 0}
            
            sentiments = []
            total_score = 0
            
            for post in all_posts:
                text = f"{post['title']} {post['text']}"
                sentiment = TextBlob(text).sentiment.polarity
                sentiments.append(sentiment)
                total_score += post['score']
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            normalized_sentiment = (avg_sentiment + 1) / 2
            
            return {
                'sentiment': normalized_sentiment,
                'mentions': len(all_posts),
                'total_score': total_score
            }
            
        except Exception as e:
            return {'sentiment': 0.5, 'mentions': 0}
    
    async def get_telegram_sentiment(self, token_symbol: str) -> Dict:
        try:
            total_mentions = 0
            sentiments = []
            
            for channel in self.telegram_channels:
                url = f"https://t.me/s/{channel.replace('@', '')}"
                
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            pattern = rf'\b{token_symbol}\b'
                            mentions = len(re.findall(pattern, content, re.IGNORECASE))
                            total_mentions += mentions
                            
                            if mentions > 0:
                                sentiment = TextBlob(content).sentiment.polarity
                                sentiments.append(sentiment)
                                
                except:
                    continue
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            normalized_sentiment = (avg_sentiment + 1) / 2
            
            return {
                'sentiment': normalized_sentiment,
                'mentions': total_mentions
            }
            
        except Exception as e:
            return {'sentiment': 0.5, 'mentions': 0}
    
    def calculate_volume_spike(self, token_symbol: str, current_mentions: int) -> float:
        cache_key = f"mentions_{token_symbol}"
        
        if cache_key not in self.sentiment_cache:
            self.sentiment_cache[cache_key] = []
        
        self.sentiment_cache[cache_key].append(current_mentions)
        
        if len(self.sentiment_cache[cache_key]) < 5:
            return 0.0
        
        recent_avg = np.mean(self.sentiment_cache[cache_key][-5:])
        historical_avg = np.mean(self.sentiment_cache[cache_key][:-5]) if len(self.sentiment_cache[cache_key]) > 5 else recent_avg
        
        if historical_avg == 0:
            return 0.0
        
        spike = (recent_avg - historical_avg) / historical_avg
        return min(max(spike, 0), 2.0)
    
    def calculate_influence_score(self, twitter_data: Dict, reddit_data: Dict, telegram_data: Dict) -> float:
        twitter_influence = (twitter_data.get('engagement', 0) / 1000) * 0.4
        reddit_influence = (reddit_data.get('total_score', 0) / 100) * 0.3
        telegram_influence = (telegram_data.get('mentions', 0) / 10) * 0.3
        
        total_influence = twitter_influence + reddit_influence + telegram_influence
        return min(total_influence, 1.0)
    
    async def shutdown(self):
        if self.session:
            await self.session.close()

real_social_sentiment = RealSocialSentimentAnalyzer()