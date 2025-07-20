import asyncio
import aiohttp
import tweepy
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import re
import logging
from textblob import TextBlob
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

@dataclass
class SentimentData:
    platform: str
    token_symbol: str
    mentions_count: int
    positive_sentiment: float
    negative_sentiment: float
    neutral_sentiment: float
    compound_score: float
    volume_24h: int
    influencer_mentions: int
    trending_score: float
    timestamp: int

@dataclass
class InfluencerMention:
    username: str
    followers_count: int
    text: str
    sentiment_score: float
    retweets: int
    likes: int
    timestamp: int

class TwitterSentimentAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        self.auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
        
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            wait_on_rate_limit=True
        )
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        self.crypto_influencers = [
            {'username': 'elonmusk', 'weight': 10.0},
            {'username': 'VitalikButerin', 'weight': 8.0},
            {'username': 'cz_binance', 'weight': 7.0},
            {'username': 'brian_armstrong', 'weight': 6.0},
            {'username': 'SatoshiLite', 'weight': 5.0},
            {'username': 'APompliano', 'weight': 4.0},
            {'username': 'CryptoCobain', 'weight': 3.0},
            {'username': 'WClementeIII', 'weight': 3.0},
            {'username': 'RyanSAdams', 'weight': 2.5},
            {'username': 'laurashin', 'weight': 2.0}
        ]
    
    async def analyze_token_sentiment(self, token_symbol: str, hours_back: int = 24) -> SentimentData:
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            tweets = await self.fetch_tweets(token_symbol, start_time, end_time)
            influencer_mentions = await self.fetch_influencer_mentions(token_symbol, start_time, end_time)
            
            sentiment_scores = []
            mention_count = 0
            
            for tweet in tweets:
                text = tweet.get('text', '')
                cleaned_text = self.clean_tweet_text(text)
                
                if self.is_relevant_mention(cleaned_text, token_symbol):
                    sentiment = self.analyze_text_sentiment(cleaned_text)
                    sentiment_scores.append(sentiment)
                    mention_count += 1
            
            if not sentiment_scores:
                return self.get_neutral_sentiment(token_symbol)
            
            positive = np.mean([s['pos'] for s in sentiment_scores])
            negative = np.mean([s['neg'] for s in sentiment_scores])
            neutral = np.mean([s['neu'] for s in sentiment_scores])
            compound = np.mean([s['compound'] for s in sentiment_scores])
            
            trending_score = self.calculate_trending_score(mention_count, hours_back)
            
            return SentimentData(
                platform='twitter',
                token_symbol=token_symbol,
                mentions_count=mention_count,
                positive_sentiment=positive,
                negative_sentiment=negative,
                neutral_sentiment=neutral,
                compound_score=compound,
                volume_24h=mention_count,
                influencer_mentions=len(influencer_mentions),
                trending_score=trending_score,
                timestamp=int(time.time())
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {token_symbol}: {e}")
            return self.get_neutral_sentiment(token_symbol)
    
    async def fetch_tweets(self, token_symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        try:
            query = f"${token_symbol} OR #{token_symbol} OR {token_symbol} -is:retweet lang:en"
            
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                start_time=start_time,
                end_time=end_time,
                max_results=100,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations']
            ).flatten(limit=1000)
            
            tweet_data = []
            for tweet in tweets:
                tweet_data.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_id': tweet.author_id,
                    'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                    'like_count': tweet.public_metrics.get('like_count', 0)
                })
            
            return tweet_data
            
        except Exception as e:
            self.logger.error(f"Error fetching tweets: {e}")
            return []
    
    async def fetch_influencer_mentions(self, token_symbol: str, start_time: datetime, end_time: datetime) -> List[InfluencerMention]:
        mentions = []
        
        for influencer in self.crypto_influencers:
            try:
                user = self.client.get_user(username=influencer['username'])
                if not user.data:
                    continue
                
                user_tweets = self.client.get_users_tweets(
                    user.data.id,
                    start_time=start_time,
                    end_time=end_time,
                    max_results=50,
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if user_tweets.data:
                    for tweet in user_tweets.data:
                        text = tweet.text.lower()
                        if (token_symbol.lower() in text or 
                            f"${token_symbol.lower()}" in text or 
                            f"#{token_symbol.lower()}" in text):
                            
                            sentiment = self.analyze_text_sentiment(tweet.text)
                            
                            mention = InfluencerMention(
                                username=influencer['username'],
                                followers_count=user.data.public_metrics['followers_count'],
                                text=tweet.text,
                                sentiment_score=sentiment['compound'],
                                retweets=tweet.public_metrics.get('retweet_count', 0),
                                likes=tweet.public_metrics.get('like_count', 0),
                                timestamp=int(tweet.created_at.timestamp())
                            )
                            
                            mentions.append(mention)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error fetching tweets for {influencer['username']}: {e}")
                continue
        
        return mentions
    
    def clean_tweet_text(self, text: str) -> str:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s$]', '', text)
        text = text.strip()
        return text
    
    def is_relevant_mention(self, text: str, token_symbol: str) -> bool:
        text_lower = text.lower()
        symbol_lower = token_symbol.lower()
        
        if len(text_lower) < 10:
            return False
        
        negative_keywords = ['scam', 'rug', 'dump', 'fake', 'ponzi']
        if any(keyword in text_lower for keyword in negative_keywords):
            return True
        
        return (symbol_lower in text_lower or 
                f"${symbol_lower}" in text_lower or 
                f"#{symbol_lower}" in text_lower)
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        blob = TextBlob(text)
        blob_polarity = blob.sentiment.polarity
        blob_subjectivity = blob.sentiment.subjectivity
        
        combined_compound = (vader_scores['compound'] + blob_polarity) / 2
        
        return {
            'compound': combined_compound,
            'pos': vader_scores['pos'],
            'neu': vader_scores['neu'],
            'neg': vader_scores['neg'],
            'subjectivity': blob_subjectivity
        }
    
    def calculate_trending_score(self, mention_count: int, hours_back: int) -> float:
        mentions_per_hour = mention_count / hours_back
        
        if mentions_per_hour >= 100:
            return 1.0
        elif mentions_per_hour >= 50:
            return 0.8
        elif mentions_per_hour >= 20:
            return 0.6
        elif mentions_per_hour >= 10:
            return 0.4
        elif mentions_per_hour >= 5:
            return 0.2
        else:
            return 0.1
    
    def get_neutral_sentiment(self, token_symbol: str) -> SentimentData:
        return SentimentData(
            platform='twitter',
            token_symbol=token_symbol,
            mentions_count=0,
            positive_sentiment=0.33,
            negative_sentiment=0.33,
            neutral_sentiment=0.34,
            compound_score=0.0,
            volume_24h=0,
            influencer_mentions=0,
            trending_score=0.0,
            timestamp=int(time.time())
        )

class TelegramSentimentAnalyzer:
    def __init__(self):
        self.api_id = os.getenv('TELEGRAM_API_ID')
        self.api_hash = os.getenv('TELEGRAM_API_HASH')
        self.phone = os.getenv('TELEGRAM_PHONE')
        
        self.channels = [
            '@DefiPulse',
            '@CryptoNews',
            '@binancechannel',
            '@coinbase',
            '@uniswap',
            '@ethereum',
            '@defi_discussions'
        ]
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    async def analyze_token_sentiment(self, token_symbol: str, hours_back: int = 24) -> SentimentData:
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            messages = await self.fetch_messages(token_symbol, start_time, end_time)
            
            if not messages:
                return self.get_neutral_sentiment(token_symbol)
            
            sentiment_scores = []
            for message in messages:
                if self.is_relevant_message(message['text'], token_symbol):
                    sentiment = self.analyze_text_sentiment(message['text'])
                    sentiment_scores.append(sentiment)
            
            if not sentiment_scores:
                return self.get_neutral_sentiment(token_symbol)
            
            positive = np.mean([s['pos'] for s in sentiment_scores])
            negative = np.mean([s['neg'] for s in sentiment_scores])
            neutral = np.mean([s['neu'] for s in sentiment_scores])
            compound = np.mean([s['compound'] for s in sentiment_scores])
            
            return SentimentData(
                platform='telegram',
                token_symbol=token_symbol,
                mentions_count=len(messages),
                positive_sentiment=positive,
                negative_sentiment=negative,
                neutral_sentiment=neutral,
                compound_score=compound,
                volume_24h=len(messages),
                influencer_mentions=0,
                trending_score=min(len(messages) / 50, 1.0),
                timestamp=int(time.time())
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing Telegram sentiment: {e}")
            return self.get_neutral_sentiment(token_symbol)
    
    async def fetch_messages(self, token_symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        return []
    
    def is_relevant_message(self, text: str, token_symbol: str) -> bool:
        text_lower = text.lower()
        symbol_lower = token_symbol.lower()
        
        return (symbol_lower in text_lower or 
                f"${symbol_lower}" in text_lower or 
                f"#{symbol_lower}" in text_lower)
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        return self.vader_analyzer.polarity_scores(text)
    
    def get_neutral_sentiment(self, token_symbol: str) -> SentimentData:
        return SentimentData(
            platform='telegram',
            token_symbol=token_symbol,
            mentions_count=0,
            positive_sentiment=0.33,
            negative_sentiment=0.33,
            neutral_sentiment=0.34,
            compound_score=0.0,
            volume_24h=0,
            influencer_mentions=0,
            trending_score=0.0,
            timestamp=int(time.time())
        )

class RedditSentimentAnalyzer:
    def __init__(self):
        import praw
        
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='CryptoSentimentBot/1.0'
        )
        
        self.subreddits = [
            'CryptoCurrency',
            'ethereum',
            'defi',
            'UniSwap',
            'ethtrader',
            'cryptomarkets',
            'altcoins'
        ]
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    async def analyze_token_sentiment(self, token_symbol: str, hours_back: int = 24) -> SentimentData:
        try:
            posts = await self.fetch_posts(token_symbol, hours_back)
            comments = await self.fetch_comments(token_symbol, hours_back)
            
            all_texts = posts + comments
            
            if not all_texts:
                return self.get_neutral_sentiment(token_symbol)
            
            sentiment_scores = []
            for text in all_texts:
                if self.is_relevant_text(text, token_symbol):
                    sentiment = self.analyze_text_sentiment(text)
                    sentiment_scores.append(sentiment)
            
            if not sentiment_scores:
                return self.get_neutral_sentiment(token_symbol)
            
            positive = np.mean([s['pos'] for s in sentiment_scores])
            negative = np.mean([s['neg'] for s in sentiment_scores])
            neutral = np.mean([s['neu'] for s in sentiment_scores])
            compound = np.mean([s['compound'] for s in sentiment_scores])
            
            return SentimentData(
                platform='reddit',
                token_symbol=token_symbol,
                mentions_count=len(all_texts),
                positive_sentiment=positive,
                negative_sentiment=negative,
                neutral_sentiment=neutral,
                compound_score=compound,
                volume_24h=len(all_texts),
                influencer_mentions=0,
                trending_score=min(len(all_texts) / 100, 1.0),
                timestamp=int(time.time())
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing Reddit sentiment: {e}")
            return self.get_neutral_sentiment(token_symbol)
    
    async def fetch_posts(self, token_symbol: str, hours_back: int) -> List[str]:
        posts = []
        
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for post in subreddit.new(limit=50):
                    post_age = time.time() - post.created_utc
                    if post_age <= hours_back * 3600:
                        if token_symbol.lower() in post.title.lower() or \
                           token_symbol.lower() in post.selftext.lower():
                            posts.append(post.title + " " + post.selftext)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error fetching from r/{subreddit_name}: {e}")
                continue
        
        return posts
    
    async def fetch_comments(self, token_symbol: str, hours_back: int) -> List[str]:
        return []
    
    def is_relevant_text(self, text: str, token_symbol: str) -> bool:
        text_lower = text.lower()
        symbol_lower = token_symbol.lower()
        
        return (symbol_lower in text_lower or 
                f"${symbol_lower}" in text_lower or 
                f"#{symbol_lower}" in text_lower)
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        return self.vader_analyzer.polarity_scores(text)
    
    def get_neutral_sentiment(self, token_symbol: str) -> SentimentData:
        return SentimentData(
            platform='reddit',
            token_symbol=token_symbol,
            mentions_count=0,
            positive_sentiment=0.33,
            negative_sentiment=0.33,
            neutral_sentiment=0.34,
            compound_score=0.0,
            volume_24h=0,
            influencer_mentions=0,
            trending_score=0.0,
            timestamp=int(time.time())
        )

class RealSocialSentimentAnalyzer:
    def __init__(self):
        self.twitter_analyzer = TwitterSentimentAnalyzer()
        self.telegram_analyzer = TelegramSentimentAnalyzer()
        self.reddit_analyzer = RedditSentimentAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        self.platform_weights = {
            'twitter': 0.5,
            'telegram': 0.3,
            'reddit': 0.2
        }
    
    async def analyze_comprehensive_sentiment(self, token_symbol: str, hours_back: int = 24) -> Dict[str, SentimentData]:
        
        tasks = [
            self.twitter_analyzer.analyze_token_sentiment(token_symbol, hours_back),
            self.telegram_analyzer.analyze_token_sentiment(token_symbol, hours_back),
            self.reddit_analyzer.analyze_token_sentiment(token_symbol, hours_back)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        sentiment_data = {}
        
        for i, result in enumerate(results):
            platform = ['twitter', 'telegram', 'reddit'][i]
            
            if isinstance(result, Exception):
                self.logger.error(f"Error analyzing {platform}: {result}")
                sentiment_data[platform] = self.get_neutral_sentiment(token_symbol, platform)
            else:
                sentiment_data[platform] = result
        
        return sentiment_data
    
    def calculate_weighted_sentiment(self, sentiment_data: Dict[str, SentimentData]) -> SentimentData:
        
        total_weight = 0
        weighted_positive = 0
        weighted_negative = 0
        weighted_neutral = 0
        weighted_compound = 0
        total_mentions = 0
        total_influencer_mentions = 0
        weighted_trending = 0
        
        for platform, data in sentiment_data.items():
            weight = self.platform_weights.get(platform, 0.1)
            
            if data.mentions_count > 0:
                weighted_positive += data.positive_sentiment * weight
                weighted_negative += data.negative_sentiment * weight
                weighted_neutral += data.neutral_sentiment * weight
                weighted_compound += data.compound_score * weight
                weighted_trending += data.trending_score * weight
                total_weight += weight
            
            total_mentions += data.mentions_count
            total_influencer_mentions += data.influencer_mentions
        
        if total_weight == 0:
            return self.get_neutral_sentiment('UNKNOWN', 'combined')
        
        return SentimentData(
            platform='combined',
            token_symbol=list(sentiment_data.values())[0].token_symbol,
            mentions_count=total_mentions,
            positive_sentiment=weighted_positive / total_weight,
            negative_sentiment=weighted_negative / total_weight,
            neutral_sentiment=weighted_neutral / total_weight,
            compoun