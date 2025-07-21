import asyncio
import pandas as pd
from web3 import Web3
from signal_detector import SignalDetector
from inference_model import MomentumEnsemble
from trade_executor import TradeExecutor
from safety_checks import SafetyChecker
from risk_manager import RiskManager
from token_profiler import TokenProfiler
from anti_rug_analyzer import RugpullAnalyzer
from mempool_watcher import MempoolWatcher
from feedback_loop import FeedbackLoop
import logging
import logging.handlers
import json
import os
import redis
import yaml
from prometheus_client import Counter, Gauge, Histogram
import time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.handlers.RotatingFileHandler('trading.log', maxBytes=10*1024*1024, backupCount=5)]
)

trade_counter = Counter('trades_executed_total', 'Total trades executed', ['chain'])
momentum_histogram = Histogram('momentum_score', 'Momentum score distribution', ['chain', 'token'])
system_health = Gauge('system_health', 'System health status', ['component'])
pipeline_uptime = Gauge('pipeline_uptime_seconds', 'Pipeline uptime in seconds')
tokens_scanned = Counter('tokens_scanned_total', 'Total tokens scanned', ['chain'])
breakouts_detected = Counter('breakouts_detected_total', 'Total breakouts detected', ['chain'])

async def main_pipeline():
    start_time = time.time()
    
    try:
        system_health.labels(component='pipeline').set(1)
        
        with open('settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        
        chains = {
            'arbitrum': Web3(Web3.HTTPProvider(os.getenv('ARBITRUM_RPC_URL'), request_kwargs={'timeout': 10})),
            'polygon': Web3(Web3.HTTPProvider(os.getenv('POLYGON_RPC_URL'), request_kwargs={'timeout': 10})),
            'optimism': Web3(Web3.HTTPProvider(os.getenv('OPTIMISM_RPC_URL'), request_kwargs={'timeout': 10}))
        }
        
        backup_providers = {
            'arbitrum': Web3(Web3.HTTPProvider(os.getenv('ARBITRUM_BACKUP_RPC_URL'))),
            'polygon': Web3(Web3.HTTPProvider(os.getenv('POLYGON_BACKUP_RPC_URL'))),
            'optimism': Web3(Web3.HTTPProvider(os.getenv('OPTIMISM_BACKUP_RPC_URL')))
        }
        
        redis_client = redis.Redis(
            host=settings['redis']['host'], 
            port=settings['redis']['port'], 
            db=0,
            decode_responses=False
        )

        for chain, w3 in chains.items():
            try:
                if not w3.is_connected():
                    logging.warning(f"Primary {chain} node failed, switching to backup")
                    chains[chain] = backup_providers[chain]
                    if not chains[chain].is_connected():
                        system_health.labels(component=f'{chain}_connection').set(0)
                        raise ConnectionError(f"Failed to connect to {chain} node")
                    else:
                        system_health.labels(component=f'{chain}_connection').set(1)
                else:
                    system_health.labels(component=f'{chain}_connection').set(1)
            except Exception as e:
                logging.error(f"Chain {chain} connection error: {str(e)}")
                system_health.labels(component=f'{chain}_connection').set(0)

        signal_detector = SignalDetector(chains, redis_client)
        momentum_model = MomentumEnsemble()
        trade_executor = TradeExecutor(chains)
        safety_checker = SafetyChecker(chains)
        risk_manager = RiskManager()
        token_profiler = TokenProfiler(chains)
        rugpull_analyzer = RugpullAnalyzer(chains)
        mempool_watcher = MempoolWatcher(chains)
        feedback_loop = FeedbackLoop(momentum_model)

        logging.info("DeFi Momentum Trading Pipeline Started")
        
        scan_iteration = 0
        last_rebalance = time.time()
        
        while True:
            try:
                pipeline_uptime.set(time.time() - start_time)
                scan_iteration += 1
                
                market_regime = await signal_detector.detect_market_regime()
                
                if market_regime == 'extreme_volatility':
                    logging.warning(json.dumps({
                        'event': 'circuit_breaker_triggered', 
                        'reason': 'Extreme market volatility',
                        'iteration': scan_iteration
                    }))
                    system_health.labels(component='circuit_breaker').set(0)
                    await asyncio.sleep(300)
                    continue
                else:
                    system_health.labels(component='circuit_breaker').set(1)

                optimal_chains = signal_detector.select_optimal_chains()
                
                scan_tasks = []
                for chain in optimal_chains:
                    scan_tasks.append(process_chain(
                        chain, signal_detector, momentum_model, trade_executor,
                        safety_checker, risk_manager, token_profiler, 
                        rugpull_analyzer, mempool_watcher, feedback_loop,
                        scan_iteration
                    ))
                
                chain_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
                
                total_tokens_processed = 0
                total_trades_executed = 0
                
                for i, result in enumerate(chain_results):
                    if isinstance(result, dict):
                        total_tokens_processed += result.get('tokens_processed', 0)
                        total_trades_executed += result.get('trades_executed', 0)
                    elif isinstance(result, Exception):
                        chain_name = optimal_chains[i] if i < len(optimal_chains) else f"chain_{i}"
                        logging.error(json.dumps({
                            'event': 'chain_processing_error',
                            'chain': chain_name,
                            'error': str(result),
                            'iteration': scan_iteration
                        }))

                if time.time() - last_rebalance > 1800:
                    momentum_model.rebalance_thresholds()
                    await momentum_model.retrain_if_needed()
                    await feedback_loop.optimize_model()
                    await feedback_loop.adaptive_learning()
                    last_rebalance = time.time()

                logging.info(json.dumps({
                    'event': 'scan_iteration_completed',
                    'iteration': scan_iteration,
                    'tokens_processed': total_tokens_processed,
                    'trades_executed': total_trades_executed,
                    'market_regime': market_regime,
                    'uptime': time.time() - start_time
                }))

                await asyncio.sleep(30)
                
            except Exception as e:
                logging.error(json.dumps({
                    'event': 'pipeline_iteration_error', 
                    'error': str(e),
                    'iteration': scan_iteration
                }))
                system_health.labels(component='pipeline').set(0)
                await asyncio.sleep(60)
                
    except Exception as e:
        logging.error(json.dumps({
            'event': 'pipeline_fatal_error', 
            'error': str(e)
        }))
        system_health.labels(component='pipeline').set(0)
        raise

async def process_chain(chain, signal_detector, momentum_model, trade_executor,
                       safety_checker, risk_manager, token_profiler, 
                       rugpull_analyzer, mempool_watcher, feedback_loop, iteration):
    try:
        system_health.labels(component=f'{chain}_scan').set(1)
        
        tokens = await signal_detector.scan_tokens(chain)
        tokens_scanned.labels(chain=chain).inc(len(tokens))
        
        if not tokens:
            logging.info(json.dumps({
                'event': 'no_tokens_detected',
                'chain': chain,
                'iteration': iteration
            }))
            return {'tokens_processed': 0, 'trades_executed': 0}

        logging.info(json.dumps({
            'event': 'tokens_detected',
            'chain': chain,
            'count': len(tokens),
            'iteration': iteration
        }))

        trades_executed = 0
        tokens_processed = 0
        
        processing_tasks = []
        semaphore = asyncio.Semaphore(5)
        
        async def process_token_with_semaphore(token):
            async with semaphore:
                return await process_single_token(
                    chain, token, momentum_model, trade_executor,
                    safety_checker, risk_manager, token_profiler,
                    rugpull_analyzer, mempool_watcher, feedback_loop
                )
        
        for token in tokens[:100]:
            processing_tasks.append(process_token_with_semaphore(token))
        
        token_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        for result in token_results:
            if isinstance(result, dict):
                tokens_processed += 1
                if result.get('trade_executed', False):
                    trades_executed += 1
            elif isinstance(result, Exception):
                logging.error(json.dumps({
                    'event': 'token_processing_error',
                    'chain': chain,
                    'error': str(result)
                }))

        system_health.labels(component=f'{chain}_scan').set(1)
        
        return {
            'tokens_processed': tokens_processed,
            'trades_executed': trades_executed
        }
        
    except Exception as e:
        logging.error(json.dumps({
            'event': 'chain_processing_error',
            'chain': chain,
            'error': str(e)
        }))
        system_health.labels(component=f'{chain}_scan').set(0)
        return {'tokens_processed': 0, 'trades_executed': 0}

async def process_single_token(chain, token, momentum_model, trade_executor,
                              safety_checker, risk_manager, token_profiler,
                              rugpull_analyzer, mempool_watcher, feedback_loop):
    try:
        token_address = token['address']
        
        token_data = await token_profiler.profile_token(chain, token_address)
        if token_data.get('blacklisted', False):
            return {'trade_executed': False, 'reason': 'blacklisted'}

        safety_checks = await asyncio.gather(
            safety_checker.check_token(chain, token_address),
            rugpull_analyzer.analyze_token(chain, token_address),
            mempool_watcher.check_mempool(chain, token_address),
            return_exceptions=True
        )
        
        is_safe = all(check for check in safety_checks if isinstance(check, bool))
        
        if not is_safe:
            await token_profiler.blacklist_token(token_address)
            return {'trade_executed': False, 'reason': 'failed_safety_checks'}

        features_df = token['data']
        if features_df.empty or len(features_df) < 5:
            return {'trade_executed': False, 'reason': 'insufficient_data'}

        momentum_score = momentum_model.predict(features_df)
        momentum_histogram.labels(chain=chain, token=token_address).observe(momentum_score)
        
        velocity = token.get('velocity', 0)
        volume_spike = token.get('volume_spike', 1)
        
        trade_signals = {
            'momentum_above_threshold': momentum_score > momentum_model.dynamic_threshold,
            'velocity_sufficient': velocity >= 0.13,
            'volume_spike_detected': volume_spike >= 2.5,
            'breakout_confirmed': token.get('momentum_score', 0) > 0.7
        }
        
        signal_count = sum(trade_signals.values())
        
        if signal_count >= 3:
            breakouts_detected.labels(chain=chain).inc()
            
            position_size = risk_manager.calculate_position_size(features_df, chain)
            
            portfolio_check = risk_manager.check_portfolio_exposure(chain, position_size)
            balance_check = await trade_executor.check_wallet_balance(chain, position_size)
            gas_check = risk_manager.check_gas_budget(chain, 0.001)
            
            if portfolio_check and balance_check and gas_check:
                tx_hash = await trade_executor.execute_trade(
                    chain, token_address, momentum_score, position_size
                )
                
                if tx_hash:
                    trade_counter.labels(chain=chain).inc()
                    
                    await feedback_loop.log_trade(
                        chain, token_address, tx_hash, momentum_score, position_size, features_df
                    )
                    
                    logging.info(json.dumps({
                        'event': 'trade_executed_successfully',
                        'chain': chain,
                        'token': token_address,
                        'momentum_score': momentum_score,
                        'velocity': velocity,
                        'volume_spike': volume_spike,
                        'position_size': position_size,
                        'tx_hash': tx_hash.hex(),
                        'signals': trade_signals
                    }))
                    
                    return {'trade_executed': True, 'tx_hash': tx_hash.hex()}
                else:
                    return {'trade_executed': False, 'reason': 'execution_failed'}
            else:
                return {'trade_executed': False, 'reason': 'risk_limits_exceeded'}
        else:
            return {'trade_executed': False, 'reason': 'insufficient_signals', 'signal_count': signal_count}

    except Exception as e:
        logging.error(json.dumps({
            'event': 'token_processing_error',
            'chain': chain,
            'token': token.get('address', 'unknown'),
            'error': str(e)
        }))
        return {'trade_executed': False, 'reason': 'processing_error'}

async def emergency_shutdown():
    try:
        logging.critical("Emergency shutdown initiated")
        system_health.labels(component='pipeline').set(0)
        
    except Exception as e:
        logging.error(json.dumps({
            'event': 'emergency_shutdown_error',
            'error': str(e)
        }))

def setup_signal_handlers():
    import signal
    
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(emergency_shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    setup_signal_handlers()
    
    try:
        asyncio.run(main_pipeline())
    except KeyboardInterrupt:
        logging.info("Pipeline stopped by user")
    except Exception as e:
        logging.critical(json.dumps({
            'event': 'pipeline_crash',
            'error': str(e)
        }))
        raise