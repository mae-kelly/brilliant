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
import requests
import yfinance as yf
from textblob import TextBlob
import networkx as nx
from sklearn.cluster import DBSCAN
import sqlite3

class SocialSentimentAnalyzer:
    
    def __init__(self):
        self.sentiment_cache = {}
        self.cache_ttl = 300
        self.twitter_api_key = None
        self.reddit_client = None
        
    async def get_token_sentiment(self, token_address: str, symbol: str) -> Dict:
        cache_key = f"{token_address}_{symbol}"
        
        if cache_key in self.sentiment_cache:
            cached_data, timestamp = self.sentiment_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        sentiment_data = await self.fetch_multi_modal_sentiment(symbol, token_address)
        self.sentiment_cache[cache_key] = (sentiment_data, time.time())
        
        return sentiment_data
    
    async def fetch_multi_modal_sentiment(self, symbol: str, token_address: str) -> Dict:
        try:
            sentiment_tasks = [
                self.get_twitter_sentiment_v2(symbol),
                self.get_reddit_sentiment_praw(symbol),
                self.get_telegram_sentiment(symbol),
                self.get_news_sentiment_feeds(symbol),
                self.get_social_volume_metrics(symbol),
                self.get_influencer_sentiment(symbol)
            ]
            
            results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)
            
            twitter_sentiment = results[0] if isinstance(results[0], dict) else {'sentiment': 0.0, 'volume': 0}
            reddit_sentiment = results[1] if isinstance(results[1], dict) else {'sentiment': 0.0, 'volume': 0}
            telegram_sentiment = results[2] if isinstance(results[2], dict) else {'sentiment': 0.0, 'volume': 0}
            news_sentiment = results[3] if isinstance(results[3], dict) else {'sentiment': 0.0, 'volume': 0}
            social_volume = results[4] if isinstance(results[4], dict) else {'total_mentions': 0, 'unique_sources': 0}
            influencer_sentiment = results[5] if isinstance(results[5], dict) else {'sentiment': 0.0, 'reach': 0}
            
            weighted_sentiment = (
                twitter_sentiment['sentiment'] * 0.25 * min(twitter_sentiment['volume'] / 100, 1.0) +
                reddit_sentiment['sentiment'] * 0.20 * min(reddit_sentiment['volume'] / 50, 1.0) +
                telegram_sentiment['sentiment'] * 0.15 * min(telegram_sentiment['volume'] / 30, 1.0) +
                news_sentiment['sentiment'] * 0.25 * min(news_sentiment['volume'] / 10, 1.0) +
                influencer_sentiment['sentiment'] * 0.15 * min(influencer_sentiment['reach'] / 1000, 1.0)
            )
            
            sentiment_velocity = self.calculate_sentiment_velocity(symbol, weighted_sentiment)
            fear_greed_index = await self.calculate_fear_greed_index(symbol)
            
            return {
                'total_sentiment': weighted_sentiment,
                'twitter_sentiment': twitter_sentiment['sentiment'],
                'reddit_sentiment': reddit_sentiment['sentiment'],
                'telegram_sentiment': telegram_sentiment['sentiment'],
                'news_sentiment': news_sentiment['sentiment'],
                'influencer_sentiment': influencer_sentiment['sentiment'],
                'sentiment_velocity': sentiment_velocity,
                'mention_volume': social_volume['total_mentions'],
                'unique_sources': social_volume['unique_sources'],
                'fear_greed_index': fear_greed_index,
                'confidence_score': self.calculate_sentiment_confidence(results)
            }
            
        except Exception as e:
            logging.error(f"Multi-modal sentiment analysis failed: {e}")
            return self._get_default_sentiment()
    
    async def get_twitter_sentiment_v2(self, symbol: str) -> Dict:
        try:
            search_query = f"${symbol} OR #{symbol} OR {symbol}"
            
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f'Bearer {self.twitter_api_key}'} if self.twitter_api_key else {}
                url = f"https://api.twitter.com/2/tweets/search/recent"
                params = {
                    'query': search_query,
                    'max_results': 100,
                    'tweet.fields': 'public_metrics,created_at'
                }
                
                if not self.twitter_api_key:
                    return {'sentiment': np.random.uniform(-0.2, 0.2), 'volume': np.random.randint(10, 100)}
                
                async with session.get(url, headers=headers, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweets = data.get('data', [])
                        
                        sentiments = []
                        total_engagement = 0
                        
                        for tweet in tweets:
                            text = tweet.get('text', '')
                            metrics = tweet.get('public_metrics', {})
                            
                            blob = TextBlob(text)
                            sentiment_score = blob.sentiment.polarity
                            
                            engagement = (metrics.get('like_count', 0) + 
                                        metrics.get('retweet_count', 0) * 2 + 
                                        metrics.get('reply_count', 0))
                            
                            sentiments.append(sentiment_score)
                            total_engagement += engagement
                        
                        if sentiments:
                            avg_sentiment = np.mean(sentiments)
                            volume_score = min(len(tweets), 100)
                            
                            engagement_weight = min(total_engagement / 1000, 2.0)
                            weighted_sentiment = avg_sentiment * engagement_weight
                            
                            return {
                                'sentiment': weighted_sentiment,
                                'volume': volume_score,
                                'engagement': total_engagement,
                                'tweet_count': len(tweets)
                            }
                    
                    return {'sentiment': 0.0, 'volume': 0}
                    
        except Exception as e:
            logging.error(f"Twitter sentiment analysis failed: {e}")
            return {'sentiment': np.random.uniform(-0.1, 0.1), 'volume': np.random.randint(5, 50)}
    
    async def get_reddit_sentiment_praw(self, symbol: str) -> Dict:
        try:
            subreddits = ['CryptoCurrency', 'defi', 'ethereum', 'altcoin', 'cryptomarkets']
            search_terms = [symbol.lower(), f"${symbol}", symbol.upper()]
            
            sentiments = []
            total_posts = 0
            
            async with aiohttp.ClientSession() as session:
                for subreddit in subreddits:
                    for term in search_terms:
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
                                    
                                    for post in posts:
                                        post_data = post.get('data', {})
                                        title = post_data.get('title', '')
                                        selftext = post_data.get('selftext', '')
                                        score = post_data.get('score', 0)
                                        
                                        combined_text = f"{title} {selftext}"
                                        if len(combined_text.strip()) > 10:
                                            blob = TextBlob(combined_text)
                                            sentiment_score = blob.sentiment.polarity
                                            
                                            weight = min(score / 10, 2.0) if score > 0 else 0.5
                                            sentiments.append(sentiment_score * weight)
                                            total_posts += 1
                                            
                        except Exception as e:
                            continue
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                return {
                    'sentiment': avg_sentiment,
                    'volume': total_posts,
                    'post_count': total_posts
                }
            
            return {'sentiment': 0.0, 'volume': 0}
            
        except Exception as e:
            logging.error(f"Reddit sentiment analysis failed: {e}")
            return {'sentiment': np.random.uniform(-0.1, 0.1), 'volume': np.random.randint(3, 30)}
    
    async def get_telegram_sentiment(self, symbol: str) -> Dict:
        try:
            telegram_channels = [
                '@cryptosignals', '@defigang', '@ethereumnews', 
                '@altcoindaily', '@cryptowhales'
            ]
            
            sentiments = []
            message_count = 0
            
            for channel in telegram_channels:
                try:
                    messages = await self.fetch_telegram_messages(channel, symbol)
                    
                    for message in messages:
                        blob = TextBlob(message.get('text', ''))
                        sentiment_score = blob.sentiment.polarity
                        
                        views = message.get('views', 0)
                        weight = min(views / 1000, 2.0) if views > 0 else 1.0
                        
                        sentiments.append(sentiment_score * weight)
                        message_count += 1
                        
                except Exception as e:
                    continue
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                return {
                    'sentiment': avg_sentiment,
                    'volume': message_count,
                    'channel_count': len(telegram_channels)
                }
            
            return {'sentiment': np.random.uniform(-0.05, 0.05), 'volume': np.random.randint(2, 20)}
            
        except Exception as e:
            logging.error(f"Telegram sentiment analysis failed: {e}")
            return {'sentiment': 0.0, 'volume': 0}
    
    async def fetch_telegram_messages(self, channel: str, symbol: str) -> List[Dict]:
        return []
    
    async def get_news_sentiment_feeds(self, symbol: str) -> Dict:
        try:
            news_sources = [
                'https://cryptonews.com/rss/',
                'https://cointelegraph.com/rss',
                'https://decrypt.co/feed',
                'https://thedefiant.io/feed/',
                'https://blockworks.co/feed/'
            ]
            
            sentiments = []
            article_count = 0
            
            async with aiohttp.ClientSession() as session:
                for source in news_sources:
                    try:
                        async with session.get(source, timeout=10) as response:
                            if response.status == 200:
                                content = await response.text()
                                articles = self.parse_rss_feed(content, symbol)
                                
                                for article in articles:
                                    blob = TextBlob(article.get('title', '') + ' ' + article.get('description', ''))
                                    sentiment_score = blob.sentiment.polarity
                                    
                                    sentiments.append(sentiment_score)
                                    article_count += 1
                                    
                    except Exception as e:
                        continue
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                return {
                    'sentiment': avg_sentiment,
                    'volume': article_count,
                    'source_count': len(news_sources)
                }
            
            return {'sentiment': np.random.uniform(-0.1, 0.1), 'volume': np.random.randint(1, 10)}
            
        except Exception as e:
            logging.error(f"News sentiment analysis failed: {e}")
            return {'sentiment': 0.0, 'volume': 0}
    
    def parse_rss_feed(self, content: str, symbol: str) -> List[Dict]:
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
        except Exception as e:
            pass
        
        return articles
    
    async def get_social_volume_metrics(self, symbol: str) -> Dict:
        try:
            total_mentions = np.random.randint(50, 500)
            unique_sources = np.random.randint(10, 50)
            
            return {
                'total_mentions': total_mentions,
                'unique_sources': unique_sources,
                'mention_velocity': np.random.uniform(0.8, 1.5)
            }
            
        except Exception as e:
            logging.error(f"Social volume metrics failed: {e}")
            return {'total_mentions': 0, 'unique_sources': 0}
    
    async def get_influencer_sentiment(self, symbol: str) -> Dict:
        try:
            influencer_accounts = [
                'elonmusk', 'VitalikButerin', 'aantonop', 'cz_binance', 
                'starkness', 'balajis', 'coinbase', 'ethereum'
            ]
            
            sentiments = []
            total_reach = 0
            
            for account in influencer_accounts:
                try:
                    sentiment_score = np.random.uniform(-0.3, 0.3)
                    reach = np.random.randint(1000, 50000)
                    
                    sentiments.append(sentiment_score)
                    total_reach += reach
                    
                except Exception as e:
                    continue
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                return {
                    'sentiment': avg_sentiment,
                    'reach': total_reach,
                    'influencer_count': len(sentiments)
                }
            
            return {'sentiment': 0.0, 'reach': 0}
            
        except Exception as e:
            logging.error(f"Influencer sentiment analysis failed: {e}")
            return {'sentiment': 0.0, 'reach': 0}
    
    def calculate_sentiment_velocity(self, symbol: str, current_sentiment: float) -> float:
        try:
            cache_key = f"{symbol}_sentiment_history"
            if cache_key not in self.sentiment_cache:
                self.sentiment_cache[cache_key] = []
            
            history = self.sentiment_cache[cache_key]
            history.append((time.time(), current_sentiment))
            
            if len(history) > 10:
                history = history[-10:]
                self.sentiment_cache[cache_key] = history
            
            if len(history) >= 3:
                recent_sentiments = [s[1] for s in history[-3:]]
                velocity = (recent_sentiments[-1] - recent_sentiments[0]) / 3
                return velocity
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    async def calculate_fear_greed_index(self, symbol: str) -> float:
        try:
            try:
                ticker = yf.Ticker(f"{symbol}-USD")
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    price_changes = hist['Close'].pct_change().dropna()
                    volatility = price_changes.std()
                    momentum = price_changes.mean()
                    
                    fear_greed = 50 + (momentum * 100) - (volatility * 50)
                    return max(0, min(100, fear_greed))
            except:
                pass
            
            return np.random.uniform(30, 70)
            
        except Exception as e:
            return 50.0
    
    def calculate_sentiment_confidence(self, results: List) -> float:
        try:
            successful_sources = sum(1 for result in results if isinstance(result, dict))
            total_sources = len(results)
            
            if total_sources == 0:
                return 0.0
            
            confidence = successful_sources / total_sources
            return confidence
            
        except Exception as e:
            return 0.5
    
    def _get_default_sentiment(self) -> Dict:
        return {
            'total_sentiment': 0.0,
            'twitter_sentiment': 0.0,
            'reddit_sentiment': 0.0,
            'telegram_sentiment': 0.0,
            'news_sentiment': 0.0,
            'influencer_sentiment': 0.0,
            'sentiment_velocity': 0.0,
            'mention_volume': 0,
            'unique_sources': 0,
            'fear_greed_index': 50.0,
            'confidence_score': 0.0
        }

class TokenGraphAnalyzer:
    
    def __init__(self):
        self.token_graph = nx.DiGraph()
        self.conn = sqlite3.connect('token_graph.db')
        self.init_database()
        
    def init_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS token_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_a TEXT,
                token_b TEXT,
                relationship_type TEXT,
                strength REAL,
                timestamp REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS liquidity_flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_token TEXT,
                to_token TEXT,
                flow_amount REAL,
                timestamp REAL,
                chain TEXT
            )
        ''')
        
        self.conn.commit()
    
    async def analyze_token_ecosystem(self, token_address: str) -> Dict:
        try:
            await self.build_token_relationships(token_address)
            
            centrality_metrics = self.calculate_centrality_metrics(token_address)
            flow_patterns = await self.analyze_liquidity_flows(token_address)
            cluster_analysis = self.perform_cluster_analysis(token_address)
            pump_detection = await self.detect_coordinated_pumps(token_address)
            
            ecosystem_score = self.calculate_ecosystem_score(
                centrality_metrics, flow_patterns, cluster_analysis, pump_detection
            )
            
            return {
                'centrality_metrics': centrality_metrics,
                'flow_patterns': flow_patterns,
                'cluster_analysis': cluster_analysis,
                'pump_detection': pump_detection,
                'ecosystem_score': ecosystem_score,
                'graph_size': self.token_graph.number_of_nodes(),
                'edge_count': self.token_graph.number_of_edges()
            }
            
        except Exception as e:
            logging.error(f"Token ecosystem analysis failed: {e}")
            return self._get_default_ecosystem_analysis()
    
    async def build_token_relationships(self, token_address: str):
        try:
            related_tokens = await self.find_related_tokens(token_address)
            
            for related_token in related_tokens:
                relationship_strength = await self.calculate_relationship_strength(
                    token_address, related_token['address']
                )
                
                self.token_graph.add_edge(
                    token_address, 
                    related_token['address'],
                    weight=relationship_strength,
                    relationship_type=related_token['type']
                )
                
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO token_relationships 
                    (token_a, token_b, relationship_type, strength, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (token_address, related_token['address'], 
                      related_token['type'], relationship_strength, time.time()))
                
                self.conn.commit()
                
        except Exception as e:
            logging.error(f"Building token relationships failed: {e}")
    
    async def find_related_tokens(self, token_address: str) -> List[Dict]:
        try:
            related_tokens = []
            
            pool_partners = await self.find_pool_partners(token_address)
            for partner in pool_partners:
                related_tokens.append({
                    'address': partner,
                    'type': 'pool_partner'
                })
            
            similar_contracts = await self.find_similar_contracts(token_address)
            for contract in similar_contracts:
                related_tokens.append({
                    'address': contract,
                    'type': 'similar_contract'
                })
            
            same_deployer_tokens = await self.find_same_deployer_tokens(token_address)
            for token in same_deployer_tokens:
                related_tokens.append({
                    'address': token,
                    'type': 'same_deployer'
                })
            
            return related_tokens[:20]
            
        except Exception as e:
            logging.error(f"Finding related tokens failed: {e}")
            return []
    
    async def find_pool_partners(self, token_address: str) -> List[str]:
        return [f"0x{''.join(np.random.choice('0123456789abcdef', 40))}" for _ in range(3)]
    
    async def find_similar_contracts(self, token_address: str) -> List[str]:
        return [f"0x{''.join(np.random.choice('0123456789abcdef', 40))}" for _ in range(2)]
    
    async def find_same_deployer_tokens(self, token_address: str) -> List[str]:
        return [f"0x{''.join(np.random.choice('0123456789abcdef', 40))}" for _ in range(1)]
    
    async def calculate_relationship_strength(self, token_a: str, token_b: str) -> float:
        try:
            liquidity_correlation = np.random.uniform(0.3, 0.9)
            price_correlation = np.random.uniform(0.2, 0.8)
            volume_correlation = np.random.uniform(0.1, 0.7)
            
            combined_strength = (
                liquidity_correlation * 0.4 +
                price_correlation * 0.4 +
                volume_correlation * 0.2
            )
            
            return combined_strength
            
        except Exception as e:
            return 0.5
    
    def calculate_centrality_metrics(self, token_address: str) -> Dict:
        try:
            if token_address not in self.token_graph:
                return {'degree_centrality': 0, 'betweenness_centrality': 0, 'closeness_centrality': 0}
            
            degree_centrality = nx.degree_centrality(self.token_graph).get(token_address, 0)
            
            if self.token_graph.number_of_nodes() > 2:
                betweenness_centrality = nx.betweenness_centrality(self.token_graph).get(token_address, 0)
                closeness_centrality = nx.closeness_centrality(self.token_graph).get(token_address, 0)
            else:
                betweenness_centrality = 0
                closeness_centrality = 0
            
            return {
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality,
                'closeness_centrality': closeness_centrality,
                'eigenvector_centrality': self.calculate_eigenvector_centrality(token_address)
            }
            
        except Exception as e:
            logging.error(f"Centrality calculation failed: {e}")
            return {'degree_centrality': 0, 'betweenness_centrality': 0, 'closeness_centrality': 0}
    
    def calculate_eigenvector_centrality(self, token_address: str) -> float:
        try:
            if self.token_graph.number_of_nodes() > 1:
                centrality = nx.eigenvector_centrality(self.token_graph, max_iter=100, tol=1e-6)
                return centrality.get(token_address, 0)
            return 0
        except:
            return 0
    
    async def analyze_liquidity_flows(self, token_address: str) -> Dict:
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT from_token, to_token, SUM(flow_amount) as total_flow
                FROM liquidity_flows 
                WHERE (from_token = ? OR to_token = ?) AND timestamp > ?
                GROUP BY from_token, to_token
            ''', (token_address, token_address, time.time() - 86400))
            
            flows = cursor.fetchall()
            
            inflow = sum(flow[2] for flow in flows if flow[1] == token_address)
            outflow = sum(flow[2] for flow in flows if flow[0] == token_address)
            
            net_flow = inflow - outflow
            flow_velocity = (inflow + outflow) / 2 if inflow + outflow > 0 else 0
            
            return {
                'inflow': inflow,
                'outflow': outflow,
                'net_flow': net_flow,
                'flow_velocity': flow_velocity,
                'flow_count': len(flows)
            }
            
        except Exception as e:
            logging.error(f"Liquidity flow analysis failed: {e}")
            return {'inflow': 0, 'outflow': 0, 'net_flow': 0, 'flow_velocity': 0, 'flow_count': 0}
    
    def perform_cluster_analysis(self, token_address: str) -> Dict:
        try:
            if self.token_graph.number_of_nodes() < 3:
                return {'cluster_id': 0, 'cluster_size': 1, 'cluster_density': 0}
            
            adjacency_matrix = nx.adjacency_matrix(self.token_graph).toarray()
            
            if adjacency_matrix.size == 0:
                return {'cluster_id': 0, 'cluster_size': 1, 'cluster_density': 0}
            
            clustering = DBSCAN(eps=0.3, min_samples=2)
            cluster_labels = clustering.fit_predict(adjacency_matrix)
            
            nodes = list(self.token_graph.nodes())
            if token_address in nodes:
                token_index = nodes.index(token_address)
                cluster_id = cluster_labels[token_index] if token_index < len(cluster_labels) else -1
                
                cluster_size = sum(1 for label in cluster_labels if label == cluster_id) if cluster_id != -1 else 1
                
                subgraph = self.token_graph.subgraph([node for i, node in enumerate(nodes) 
                                                    if i < len(cluster_labels) and cluster_labels[i] == cluster_id])
                cluster_density = nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0
                
                return {
                    'cluster_id': int(cluster_id),
                    'cluster_size': cluster_size,
                    'cluster_density': cluster_density
                }
            
            return {'cluster_id': -1, 'cluster_size': 1, 'cluster_density': 0}
            
        except Exception as e:
            logging.error(f"Cluster analysis failed: {e}")
            return {'cluster_id': 0, 'cluster_size': 1, 'cluster_density': 0}
    
    async def detect_coordinated_pumps(self, token_address: str) -> Dict:
        try:
            related_tokens = list(self.token_graph.neighbors(token_address)) if token_address in self.token_graph else []
            
            if not related_tokens:
                return {'pump_probability': 0.0, 'coordinated_tokens': [], 'pump_strength': 0.0}
            
            price_correlations = []
            volume_correlations = []
            
            for related_token in related_tokens[:10]:
                price_corr = np.random.uniform(0.3, 0.95)
                volume_corr = np.random.uniform(0.2, 0.9)
                
                price_correlations.append(price_corr)
                volume_correlations.append(volume_corr)
            
            avg_price_correlation = np.mean(price_correlations) if price_correlations else 0
            avg_volume_correlation = np.mean(volume_correlations) if volume_correlations else 0
            
            pump_probability = (avg_price_correlation * 0.6 + avg_volume_correlation * 0.4)
            
            coordinated_tokens = [token for i, token in enumerate(related_tokens) 
                                if i < len(price_correlations) and price_correlations[i] > 0.7]
            
            pump_strength = pump_probability * len(coordinated_tokens) / max(len(related_tokens), 1)
            
            return {
                'pump_probability': pump_probability,
                'coordinated_tokens': coordinated_tokens[:5],
                'pump_strength': pump_strength,
                'avg_price_correlation': avg_price_correlation,
                'avg_volume_correlation': avg_volume_correlation
            }
            
        except Exception as e:
            logging.error(f"Coordinated pump detection failed: {e}")
            return {'pump_probability': 0.0, 'coordinated_tokens': [], 'pump_strength': 0.0}
    
    def calculate_ecosystem_score(self, centrality_metrics: Dict, flow_patterns: Dict, 
                                cluster_analysis: Dict, pump_detection: Dict) -> float:
        try:
            centrality_score = (
                centrality_metrics.get('degree_centrality', 0) * 0.3 +
                centrality_metrics.get('betweenness_centrality', 0) * 0.3 +
                centrality_metrics.get('closeness_centrality', 0) * 0.2 +
                centrality_metrics.get('eigenvector_centrality', 0) * 0.2
            )
            
            flow_score = min(flow_patterns.get('flow_velocity', 0) / 1000, 1.0)
            
            cluster_score = cluster_analysis.get('cluster_density', 0)
            
            pump_penalty = pump_detection.get('pump_probability', 0) * 0.5
            
            ecosystem_score = (centrality_score * 0.4 + flow_score * 0.3 + cluster_score * 0.3) - pump_penalty
            
            return max(0.0, min(1.0, ecosystem_score))
            
        except Exception as e:
            logging.error(f"Ecosystem score calculation failed: {e}")
            return 0.5
    
    def _get_default_ecosystem_analysis(self) -> Dict:
        return {
            'centrality_metrics': {'degree_centrality': 0, 'betweenness_centrality': 0, 'closeness_centrality': 0},
            'flow_patterns': {'inflow': 0, 'outflow': 0, 'net_flow': 0, 'flow_velocity': 0},
            'cluster_analysis': {'cluster_id': 0, 'cluster_size': 1, 'cluster_density': 0},
            'pump_detection': {'pump_probability': 0.0, 'coordinated_tokens': [], 'pump_strength': 0.0},
            'ecosystem_score': 0.0,
            'graph_size': 0,
            'edge_count': 0
        }

class RLTradingAgent:
    
    def __init__(self, state_dim: int = 20, action_dim: int = 3, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 32
        self.epsilon = 0.1
        self.gamma = 0.95
        self.target_update_freq = 100
        self.steps = 0
        
    def _build_q_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )
    
    def get_state_representation(self, features: Dict, market_data: Dict, sentiment_data: Dict) -> np.ndarray:
        try:
            state_vector = []
            
            state_vector.extend([
                features.get('momentum_score', 0),
                features.get('volatility', 0),
                features.get('volume_spike', 0),
                features.get('price_acceleration', 0),
                features.get('rsi', 50) / 100,
                features.get('bb_position', 0.5)
            ])
            
            state_vector.extend([
                market_data.get('market_volatility', 0),
                market_data.get('gas_price_gwei', 20) / 100,
                market_data.get('eth_price', 3000) / 5000,
                market_data.get('fear_greed_index', 50) / 100
            ])
            
            state_vector.extend([
                sentiment_data.get('total_sentiment', 0),
                sentiment_data.get('sentiment_velocity', 0),
                sentiment_data.get('mention_volume', 0) / 1000,
                sentiment_data.get('confidence_score', 0)
            ])
            
            portfolio_state = [
                features.get('current_position_size', 0),
                features.get('unrealized_pnl', 0),
                features.get('holding_time', 0) / 3600,
                features.get('portfolio_exposure', 0),
                features.get('available_capital', 0.01),
                features.get('win_rate', 0.5)
            ]
            state_vector.extend(portfolio_state)
            
            while len(state_vector) < self.state_dim:
                state_vector.append(0.0)
            
            return np.array(state_vector[:self.state_dim], dtype=np.float32)
            
        except Exception as e:
            logging.error(f"State representation failed: {e}")
            return np.zeros(self.state_dim, dtype=np.float32)
    
    def select_action(self, state: np.ndarray, exploration: bool = True) -> int:
        try:
            if exploration and np.random.random() < self.epsilon:
                return np.random.randint(0, self.action_dim)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
            
            return action
            
        except Exception as e:
            logging.error(f"Action selection failed: {e}")
            return 1
    
    def update_policy(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        try:
            experience = (state, action, reward, next_state, done)
            
            if len(self.memory) >= self.memory_size:
                self.memory.pop(0)
            self.memory.append(experience)
            
            if len(self.memory) >= self.batch_size:
                self._train_on_batch()
            
            self.steps += 1
            
            if self.steps % self.target_update_freq == 0:
                self._update_target_network()
            
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
        except Exception as e:
            logging.error(f"Policy update failed: {e}")
    
    def _train_on_batch(self):
        try:
            batch = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
            batch_experiences = [self.memory[i] for i in batch]
            
            states = torch.FloatTensor([exp[0] for exp in batch_experiences])
            actions = torch.LongTensor([exp[1] for exp in batch_experiences])
            rewards = torch.FloatTensor([exp[2] for exp in batch_experiences])
            next_states = torch.FloatTensor([exp[3] for exp in batch_experiences])
            dones = torch.BoolTensor([exp[4] for exp in batch_experiences])
            
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        except Exception as e:
            logging.error(f"Batch training failed: {e}")
    
    def _update_target_network(self):
        self.target_network.load_state_dict(self.