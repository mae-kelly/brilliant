import pandas as pd
import numpy as np
from intelligence.signals.signal_detector import SignalDetector
from core.models.inference_model import MomentumEnsemble
from core.execution.risk_manager import RiskManager
import logging
import json
import aiohttp
import asyncio
import yaml
from web3 import Web3
import redis
import os
import time
from datetime import datetime, timedelta

async def fetch_pool_addresses(chain):
    try:
        with open('settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        
        query = """
        query($skip: Int!) {
          pools(first: 100, skip: $skip, where: {volumeUSD_gt: 500000, liquidity_gt: 100000}) {
            id
            token0 { symbol }
            token1 { symbol }
            volumeUSD
            liquidity
            createdAtTimestamp
          }
        }
        """
        
        pools = []
        skip = 0
        endpoint = settings['dex_endpoints'][chain][list(settings['dex_endpoints'][chain].keys())[0]]
        
        async with aiohttp.ClientSession() as session:
            while len(pools) < 50:
                variables = {'skip': skip}
                async with session.post(endpoint, json={'query': query, 'variables': variables}, timeout=15) as resp:
                    if resp.status != 200:
                        break
                    
                    data = await resp.json()
                    new_pools = data.get('data', {}).get('pools', [])
                    
                    if not new_pools:
                        break
                    
                    for pool in new_pools:
                        creation_time = int(pool.get('createdAtTimestamp', 0))
                        current_time = int(time.time())
                        
                        if current_time - creation_time > 7 * 24 * 3600:
                            pools.append(pool)
                    
                    skip += 100
                    
                    if len(pools) >= 50:
                        break
        
        return pools[:50]
        
    except Exception as e:
        logging.error(json.dumps({
            'event': 'fetch_pools_error',
            'chain': chain,
            'error': str(e)
        }))
        return []

async def fetch_historical_data(chain, pool_address, days_back=7):
    try:
        query = """
        query($pool: ID!, $timestamp_gte: Int!) {
          poolHourDatas(
            first: 1000, 
            where: {pool: $pool, periodStartUnix_gte: $timestamp_gte},
            orderBy: periodStartUnix,
            orderDirection: asc
          ) {
            periodStartUnix
            high
            low
            open
            close
            volumeUSD
            liquidity
            txCount
          }
        }
        """
        
        timestamp_gte = int(time.time()) - (days_back * 24 * 3600)
        variables = {
            'pool': pool_address.lower(), 
            'timestamp_gte': timestamp_gte
        }
        
        with open('settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        
        endpoint = settings['dex_endpoints'][chain][list(settings['dex_endpoints'][chain].keys())[0]]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json={'query': query, 'variables': variables}, timeout=20) as resp:
                if resp.status != 200:
                    return pd.DataFrame()
                
                data = await resp.json()
                hour_data = data.get('data', {}).get('poolHourDatas', [])
                
                if not hour_data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(hour_data)
                
                for col in ['high', 'low', 'open', 'close', 'volumeUSD', 'liquidity']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['periodStartUnix'] = pd.to_numeric(df['periodStartUnix'])
                df['timestamp'] = pd.to_datetime(df['periodStartUnix'], unit='s')
                
                df = df.sort_values('periodStartUnix').reset_index(drop=True)
                
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(24, min_periods=1).std()
                df['volume_ma'] = df['volumeUSD'].rolling(24, min_periods=1).mean()
                df['volume_spike'] = df['volumeUSD'] / df['volume_ma']
                
                df = df.dropna().reset_index(drop=True)
                
                return df
                
    except Exception as e:
        logging.error(json.dumps({
            'event': 'backtest_data_error',
            'chain': chain,
            'pool': pool_address,
            'error': str(e)
        }))
        return pd.DataFrame()

def simulate_trades(price_data, signal_detector, momentum_model, risk_manager, pool_info):
    try:
        if price_data.empty or len(price_data) < 50:
            return []
        
        trades = []
        position = None
        
        for i in range(30, len(price_data)):
            current_slice = price_data.iloc[:i+1]
            
            if len(current_slice) < 30:
                continue
            
            try:
                features = engineer_backtest_features(current_slice, pool_info)
                
                if features.empty:
                    continue
                
                momentum_score = momentum_model.predict(features.tail(1))
                
                current_price = current_slice['close'].iloc[-1]
                current_time = current_slice['timestamp'].iloc[-1]
                
                if position is None:
                    if (momentum_score > momentum_model.dynamic_threshold and 
                        signal_detector.is_breakout(features, pool_info)):
                        
                        position_size = risk_manager.calculate_position_size(features, 'backtest')
                        
                        position = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'entry_score': momentum_score,
                            'position_size': position_size,
                            'entry_index': i
                        }
                
                else:
                    holding_time = i - position['entry_index']
                    score_decay = (position['entry_score'] - momentum_score) / position['entry_score']
                    price_change = (current_price - position['entry_price']) / position['entry_price']
                    
                    exit_conditions = [
                        score_decay >= 0.005,
                        momentum_score < momentum_model.dynamic_threshold * 0.95,
                        price_change < -0.02,
                        holding_time > 24,
                        price_change > 0.15
                    ]
                    
                    if any(exit_conditions):
                        pnl = position['position_size'] * price_change
                        
                        trade = {
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'entry_score': position['entry_score'],
                            'exit_score': momentum_score,
                            'position_size': position['position_size'],
                            'holding_time': holding_time,
                            'pnl': pnl,
                            'return_pct': price_change,
                            'score_decay': score_decay,
                            'exit_reason': get_exit_reason(exit_conditions)
                        }
                        
                        trades.append(trade)
                        position = None
            
            except Exception as e:
                logging.error(f"Trade simulation error at index {i}: {e}")
                continue
        
        return trades
        
    except Exception as e:
        logging.error(json.dumps({
            'event': 'trade_simulation_error',
            'pool': pool_info.get('id', 'unknown'),
            'error': str(e)
        }))
        return []

def engineer_backtest_features(price_data, pool_info):
    try:
        features = pd.DataFrame()
        
        features['returns'] = price_data['returns']
        features['volatility'] = price_data['volatility']
        features['momentum'] = (price_data['close'].rolling(5, min_periods=1).mean() - 
                               price_data['close'].rolling(20, min_periods=1).mean()) / price_data['close']
        
        features['rsi'] = calculate_rsi(price_data['close'])
        features['bb_position'] = calculate_bollinger_position(price_data['close'])
        features['volume_ma'] = price_data['volume_ma']
        features['whale_activity'] = pd.Series([0.1] * len(price_data))
        features['price_acceleration'] = price_data['returns'].diff()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(20, min_periods=1).mean()
        features['momentum_strength'] = abs(features['momentum']) * features['volatility_ratio']
        features['swap_volume'] = price_data['volumeUSD']
        
        features = features.fillna(method='ffill').fillna(0)
        
        return features.dropna()
        
    except Exception as e:
        logging.error(f"Feature engineering error: {e}")
        return pd.DataFrame()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_bollinger_position(prices, window=20):
    ma = prices.rolling(window=window, min_periods=1).mean()
    std = prices.rolling(window=window, min_periods=1).std()
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    bb_position = (prices - lower) / (upper - lower).replace(0, 1)
    return bb_position.fillna(0.5)

def get_exit_reason(exit_conditions):
    reasons = ['score_decay', 'threshold_breach', 'stop_loss', 'max_holding', 'take_profit']
    for i, condition in enumerate(exit_conditions):
        if condition:
            return reasons[i]
    return 'unknown'

def calculate_performance_metrics(trades):
    try:
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_holding_time': 0,
                'profit_factor': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['pnl'] > 0])
        win_rate = wins / total_trades
        
        total_pnl = trades_df['pnl'].sum()
        avg_return = trades_df['return_pct'].mean()
        
        returns = trades_df['return_pct']
        sharpe_ratio = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
        
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        avg_holding_time = trades_df['holding_time'].mean()
        
        winning_trades = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losing_trades = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = winning_trades / (losing_trades + 1e-10)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_holding_time': avg_holding_time,
            'profit_factor': profit_factor,
            'best_trade': trades_df['pnl'].max(),
            'worst_trade': trades_df['pnl'].min(),
            'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0,
            'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - wins) > 0 else 0
        }
        
    except Exception as e:
        logging.error(f"Performance calculation error: {e}")
        return {'error': str(e)}

def run_backtest():
    try:
        chains = {
            'arbitrum': Web3(Web3.HTTPProvider(os.getenv('ARBITRUM_RPC_URL', 'https://arb1.arbitrum.io/rpc'))),
            'polygon': Web3(Web3.HTTPProvider(os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com/'))),
            'optimism': Web3(Web3.HTTPProvider(os.getenv('OPTIMISM_RPC_URL', 'https://mainnet.optimism.io')))
        }
        
        with open('settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        
        signal_detector = SignalDetector(chains, redis_client)
        momentum_model = MomentumEnsemble()
        risk_manager = RiskManager()
        
        all_trades = []
        
        for chain in ['arbitrum', 'polygon']:
            try:
                pools = asyncio.run(fetch_pool_addresses(chain))
                
                if not pools:
                    logging.warning(f"No pools found for {chain}")
                    continue
                
                logging.info(f"Testing {len(pools)} pools on {chain}")
                
                for i, pool in enumerate(pools[:10]):
                    try:
                        pool_address = pool['id']
                        price_data = asyncio.run(fetch_historical_data(chain, pool_address))
                        
                        if price_data.empty:
                            continue
                        
                        trades = simulate_trades(price_data, signal_detector, momentum_model, risk_manager, pool)
                        
                        for trade in trades:
                            trade['chain'] = chain
                            trade['pool'] = pool_address
                            trade['pool_info'] = pool
                        
                        all_trades.extend(trades)
                        
                        if (i + 1) % 5 == 0:
                            logging.info(f"Processed {i + 1}/{len(pools)} pools on {chain}")
                        
                    except Exception as e:
                        logging.error(f"Pool processing error {pool.get('id', 'unknown')}: {e}")
                        continue
                        
            except Exception as e:
                logging.error(f"Chain processing error {chain}: {e}")
                continue
        
        if not all_trades:
            logging.warning("No trades generated in backtest")
            return {
                'status': 'completed',
                'total_trades': 0,
                'message': 'No trading opportunities found in historical data'
            }
        
        performance = calculate_performance_metrics(all_trades)
        
        chain_performance = {}
        for chain in ['arbitrum', 'polygon']:
            chain_trades = [t for t in all_trades if t['chain'] == chain]
            if chain_trades:
                chain_performance[chain] = calculate_performance_metrics(chain_trades)
        
        backtest_result = {
            'status': 'completed',
            'overall_performance': performance,
            'chain_performance': chain_performance,
            'total_pools_tested': len(set(t['pool'] for t in all_trades)),
            'backtest_period': '7 days',
            'timestamp': int(time.time())
        }
        
        logging.info(json.dumps({
            'event': 'backtest_completed',
            'results': backtest_result
        }))
        
        return backtest_result
        
    except Exception as e:
        logging.error(json.dumps({
            'event': 'backtest_error',
            'error': str(e)
        }))
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': int(time.time())
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_backtest()
    print(json.dumps(result, indent=2))