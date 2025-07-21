import asyncio
import pandas as pd
from web3 import Web3
from scanner_v3 import ScannerV3
from core.models.inference_model import MomentumEnsemble
from core.execution.trade_executor import TradeExecutor
from security.validators.safety_checks import SafetyChecker
from core.execution.risk_manager import RiskManager
from token_profiler import TokenProfiler
from security.rugpull.anti_rug_analyzer import RugpullAnalyzer
from security.mempool.mempool_watcher import MempoolWatcher
from feedback_loop import FeedbackLoop
from model_manager import ModelManager, TFLiteInferenceEngine
from core.features.vectorized_features import VectorizedFeatureEngine
from continuous_optimizer import ContinuousOptimizer
from advanced_ensemble import AdvancedEnsembleModel
from model_registry import ModelRegistry
from real_time_backtester import RealTimeBacktester
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
inference_latency = Histogram('inference_latency_seconds', 'ML inference latency', ['model_type'])

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

        model_manager = ModelManager()
        model_registry = ModelRegistry()
        
        momentum_model = MomentumEnsemble()
        tflite_path = model_manager.convert_pytorch_to_tflite(
            momentum_model.transformer,
            input_shape=(1, 11),
            model_name='momentum_transformer_v1',
            optimization_level='full'
        )
        
        tflite_engine = TFLiteInferenceEngine(tflite_path, num_threads=8)
        
        model_registry.register_model({
            'name': 'momentum_transformer_v1',
            'path': tflite_path,
            'performance_metrics': {'accuracy': 0.85, 'latency_ms': 12.5},
            'deployment_time': time.time()
        })

        scanner = ScannerV3(chains, redis_client)
        trade_executor = TradeExecutor(chains)
        safety_checker = SafetyChecker(chains)
        risk_manager = RiskManager()
        token_profiler = TokenProfiler(chains)
        rugpull_analyzer = RugpullAnalyzer(chains)
        mempool_watcher = MempoolWatcher(chains)
        feedback_loop = FeedbackLoop(momentum_model)
        feature_engine = VectorizedFeatureEngine()
        continuous_optimizer = ContinuousOptimizer(momentum_model, risk_manager)
        advanced_ensemble = AdvancedEnsembleModel()
        backtester = RealTimeBacktester()

        logging.info("DeFi Momentum Trading Pipeline Started with TFLite optimization")
        
        scan_iteration = 0
        last_rebalance = time.time()
        last_optimization = time.time()
        
        while True:
            try:
                pipeline_uptime.set(time.time() - start_time)
                scan_iteration += 1
                
                market_regime = await scanner.detect_market_regime()
                risk_manager.set_market_regime(market_regime)
                
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

                optimal_chains = scanner.select_optimal_chains()
                
                scan_tasks = []
                for chain in optimal_chains:
                    scan_tasks.append(process_chain_optimized(
                        chain, scanner, tflite_engine, trade_executor,
                        safety_checker, risk_manager, token_profiler, 
                        rugpull_analyzer, mempool_watcher, feedback_loop,
                        feature_engine, advanced_ensemble, backtester,
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

                if time.time() - last_optimization > 600:
                    await continuous_optimizer.optimize_parameters()
                    await advanced_ensemble.update_model_weights({})
                    last_optimization = time.time()

                logging.info(json.dumps({
                    'event': 'scan_iteration_completed',
                    'iteration': scan_iteration,
                    'tokens_processed': total_tokens_processed,
                    'trades_executed': total_trades_executed,
                    'market_regime': market_regime,
                    'uptime': time.time() - start_time
                }))

                await asyncio.sleep(15)
                
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

async def process_chain_optimized(chain, scanner, tflite_engine, trade_executor,
                                safety_checker, risk_manager, token_profiler, 
                                rugpull_analyzer, mempool_watcher, feedback_loop,
                                feature_engine, advanced_ensemble, backtester, iteration):
    try:
        system_health.labels(component=f'{chain}_scan').set(1)
        
        token_batches = await scanner.scan_tokens_ultra_fast(chain, target_count=2000)
        
        if not token_batches:
            logging.info(json.dumps({
                'event': 'no_tokens_detected',
                'chain': chain,
                'iteration': iteration
            }))
            return {'tokens_processed': 0, 'trades_executed': 0}

        total_tokens = sum(len(batch.addresses) for batch in token_batches)
        tokens_scanned.labels(chain=chain).inc(total_tokens)
        
        logging.info(json.dumps({
            'event': 'tokens_detected',
            'chain': chain,
            'count': total_tokens,
            'batches': len(token_batches),
            'iteration': iteration
        }))

        trades_executed = 0
        tokens_processed = 0
        
        for batch in token_batches:
            batch_start = time.time()
            
            vectorized_features = feature_engine.engineer_batch_features(batch.features)
            
            with inference_latency.labels(model_type='tflite').time():
                batch_predictions = tflite_engine.predict_batch(vectorized_features)
            
            ensemble_predictions = await advanced_ensemble.predict_with_multi_modal(
                chain, batch.addresses[0], pd.DataFrame(vectorized_features), 'batch'
            )
            
            for i, (address, prediction, ensemble_pred) in enumerate(zip(
                batch.addresses, batch_predictions, [ensemble_predictions] * len(batch.addresses)
            )):
                try:
                    token_data = {
                        'address': address,
                        'prediction': float(prediction),
                        'ensemble_prediction': ensemble_pred.get('ensemble_prediction', prediction),
                        'confidence': ensemble_pred.get('confidence', 0.5),
                        'metadata': batch.metadata[i]
                    }
                    
                    trade_result = await process_single_token_optimized(
                        chain, token_data, trade_executor, safety_checker, 
                        risk_manager, token_profiler, rugpull_analyzer, 
                        mempool_watcher, feedback_loop, backtester
                    )
                    
                    tokens_processed += 1
                    if trade_result.get('trade_executed', False):
                        trades_executed += 1
                        
                except Exception as e:
                    logging.error(json.dumps({
                        'event': 'token_processing_error',
                        'chain': chain,
                        'token': address,
                        'error': str(e)
                    }))
                    continue
            
            batch_time = time.time() - batch_start
            logging.info(f"Processed batch of {len(batch.addresses)} tokens in {batch_time:.2f}s")

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

async def process_single_token_optimized(chain, token_data, trade_executor,
                                       safety_checker, risk_manager, token_profiler,
                                       rugpull_analyzer, mempool_watcher, feedback_loop,
                                       backtester):
    try:
        token_address = token_data['address']
        prediction = token_data['prediction']
        confidence = token_data['confidence']
        
        if confidence < 0.7:
            return {'trade_executed': False, 'reason': 'low_confidence'}
        
        safety_checks = await asyncio.gather(
            safety_checker.check_token(chain, token_address),
            rugpull_analyzer.analyze_token(chain, token_address),
            mempool_watcher.estimate_mev_risk(chain, token_address, 0.001),
            return_exceptions=True
        )
        
        safety_passed = all(check for check in safety_checks if isinstance(check, bool))
        mev_risk = safety_checks[2] if isinstance(safety_checks[2], dict) else {'safe_to_trade': True}
        
        if not safety_passed or not mev_risk.get('safe_to_trade', False):
            await token_profiler.blacklist_token(token_address, 'failed_safety_or_mev_checks')
            return {'trade_executed': False, 'reason': 'failed_safety_checks'}

        features_df = pd.DataFrame([token_data['metadata']])
        position_size = risk_manager.calculate_position_size_regime_aware(features_df, chain)
        
        trade_signals = {
            'prediction_above_threshold': prediction > 0.75,
            'confidence_sufficient': confidence >= 0.7,
            'position_size_valid': position_size > 0.0001,
            'mev_safe': mev_risk.get('risk_score', 1.0) < 0.3
        }
        
        signal_count = sum(trade_signals.values())
        
        if signal_count >= 3:
            breakouts_detected.labels(chain=chain).inc()
            
            paper_trade_result = await backtester.validate_trade_strategy(
                chain, token_address, prediction, position_size
            )
            
            if paper_trade_result.get('expected_roi', 0) > 0.02:
                portfolio_check = risk_manager.check_portfolio_exposure(chain, position_size)
                balance_check = await trade_executor.check_wallet_balance(chain, position_size)
                
                if portfolio_check and balance_check:
                    tx_hash = await trade_executor.execute_trade(
                        chain, token_address, prediction, position_size
                    )
                    
                    if tx_hash:
                        trade_counter.labels(chain=chain).inc()
                        
                        await feedback_loop.log_trade(
                            chain, token_address, tx_hash, prediction, position_size, features_df
                        )
                        
                        logging.info(json.dumps({
                            'event': 'trade_executed_successfully',
                            'chain': chain,
                            'token': token_address,
                            'prediction': prediction,
                            'confidence': confidence,
                            'position_size': position_size,
                            'tx_hash': tx_hash.hex(),
                            'signals': trade_signals,
                            'expected_roi': paper_trade_result.get('expected_roi', 0)
                        }))
                        
                        return {'trade_executed': True, 'tx_hash': tx_hash.hex()}
                    else:
                        return {'trade_executed': False, 'reason': 'execution_failed'}
                else:
                    return {'trade_executed': False, 'reason': 'risk_limits_exceeded'}
            else:
                return {'trade_executed': False, 'reason': 'insufficient_expected_roi'}
        else:
            return {'trade_executed': False, 'reason': 'insufficient_signals', 'signal_count': signal_count}

    except Exception as e:
        logging.error(json.dumps({
            'event': 'token_processing_error',
            'chain': chain,
            'token': token_data.get('address', 'unknown'),
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