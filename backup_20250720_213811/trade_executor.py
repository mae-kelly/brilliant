from web3 import Web3
from eth_account import Account
import os
import aiohttp
import time
import logging
from abi import abi, UNISWAP_V3_POOL_ABI, ERC20_ABI
from prometheus_client import Counter, Gauge, Histogram
import json
import yaml
import asyncio
import numpy as np
import pandas as pd

from infrastructure.monitoring.error_handler import infrastructure.monitoring.error_handler, log_performance, CircuitBreaker, safe_execute
from infrastructure.monitoring.error_handler import infrastructure.monitoring.error_handler, NetworkError, ModelInferenceError

class TradeExecutor:
    def __init__(self, chains):
        self.chains = chains
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)
        self.router_addresses = {
            'arbitrum': "0xE592427A0AEce92De3Edee1F18E0157C05861564",
            'polygon': "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",
            'optimism': "0xE592427A0AEce92De3Edee1F18E0157C05861564"
        }
        self.weth_addresses = {
            'arbitrum': "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
            'polygon': "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
            'optimism': "0x4200000000000000000000000000000000000006"
        }
        self.account = Account.from_key(os.getenv('PRIVATE_KEY'))
        self.decay_threshold = self.settings['trading']['decay_threshold']
        self.trade_latency = Histogram('trade_latency_seconds', 'Trade execution latency', ['chain'])
        self.trade_success = Counter('trade_success_total', 'Successful trades', ['chain'])
        self.gas_price_gauge = Gauge('gas_price_gwei', 'Current gas price', ['chain'])
        self.cost_gauge = Gauge('gas_cost_eth', 'Cumulative gas cost', ['chain'])
        self.slippage_gauge = Gauge('trade_slippage', 'Trade slippage percentage', ['chain'])
        self.positions = {}

    async def check_wallet_balance(self, chain, position_size):
        try:
            w3 = self.chains[chain]
            balance = w3.eth.get_balance(self.account.address)
            min_balance = w3.to_wei(position_size + 0.01, 'ether')
            return balance >= min_balance
        except Exception as e:
            logging.error(json.dumps({
                'event': 'wallet_balance_check_error',
                'chain': chain,
                'error': str(e)
            }))
            return False

    async def check_token_approval(self, chain, token_address):
        try:
            w3 = self.chains[chain]
            token_contract = w3.eth.contract(address=token_address, abi=ERC20_ABI)
            allowance = token_contract.functions.allowance(
                self.account.address, 
                self.router_addresses[chain]
            ).call()
            
            required_allowance = w3.to_wei(1000, 'ether')
            
            if allowance < required_allowance:
                gas_price = await self.optimize_gas_price(chain)
                
                tx = token_contract.functions.approve(
                    self.router_addresses[chain], 
                    required_allowance
                ).build_transaction({
                    'from': self.account.address,
                    'nonce': w3.eth.get_transaction_count(self.account.address),
                    'gasPrice': gas_price,
                    'gas': 100000
                })
                
                signed_tx = w3.eth.account.sign_transaction(tx, self.account.key)
                tx_hash = await self.send_private_transaction(chain, signed_tx.rawTransaction)
                
                logging.info(json.dumps({
                    'event': 'token_approved',
                    'chain': chain,
                    'token': token_address,
                    'tx_hash': tx_hash.hex()
                }))
            return True
        except Exception as e:
            logging.error(json.dumps({
                'event': 'token_approval_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return False

    async def execute_trade(self, chain, token_address, momentum_score, position_size):
        try:
            w3 = self.chains[chain]
            router = w3.eth.contract(address=self.router_addresses[chain], abi=UNISWAP_V3_ROUTER_ABI)
            
            fee_tier = await self.select_fee_tier(chain, token_address)
            amount_in = w3.to_wei(position_size, 'ether')
            deadline = w3.eth.get_block('latest')['timestamp'] + 300
            gas_price = await self.optimize_gas_price(chain)
            
            self.gas_price_gauge.labels(chain=chain).set(gas_price / 1e9)

            if not await self.check_token_approval(chain, self.weth_addresses[chain]):
                return None

            expected_out = await self.calculate_expected_output(chain, token_address, amount_in, fee_tier)
            slippage = self.calculate_dynamic_slippage(momentum_score)
            min_amount_out = int(expected_out * (1 - slippage))
            
            self.slippage_gauge.labels(chain=chain).set(slippage * 100)

            tx_params = {
                'from': self.account.address,
                'nonce': w3.eth.get_transaction_count(self.account.address),
                'gasPrice': gas_price,
                'value': amount_in
            }
            
            swap_tx = router.functions.exactInputSingle({
                'tokenIn': self.weth_addresses[chain],
                'tokenOut': token_address,
                'fee': fee_tier,
                'recipient': self.account.address,
                'deadline': deadline,
                'amountIn': amount_in,
                'amountOutMinimum': min_amount_out,
                'sqrtPriceLimitX96': 0
            }).build_transaction(tx_params)
            
            tx_params['gas'] = w3.eth.estimate_gas(swap_tx)

            start_time = time.time()
            signed_tx = w3.eth.account.sign_transaction(swap_tx, self.account.key)
            tx_hash = await self.send_private_transaction(chain, signed_tx.rawTransaction)
            
            gas_cost = tx_params['gas'] * gas_price / 1e18
            self.cost_gauge.labels(chain=chain).inc(gas_cost)
            self.trade_latency.labels(chain=chain).observe(time.time() - start_time)
            self.trade_success.labels(chain=chain).inc()
            
            self.positions[tx_hash.hex()] = {
                'chain': chain,
                'token_address': token_address,
                'entry_score': momentum_score,
                'entry_time': time.time(),
                'position_size': position_size,
                'entry_price': await self.get_current_price(chain, token_address)
            }
            
            asyncio.create_task(self.monitor_position(chain, token_address, momentum_score, tx_hash))
            
            logging.info(json.dumps({
                'event': 'trade_executed',
                'chain': chain,
                'token': token_address,
                'tx_hash': tx_hash.hex(),
                'gas_cost': gas_cost,
                'slippage': slippage,
                'momentum_score': momentum_score
            }))

            return tx_hash
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'trade_execution_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return None

    def calculate_dynamic_slippage(self, momentum_score):
        base_slippage = 0.005
        urgency_multiplier = min(momentum_score * 2, 3.0)
        return min(base_slippage * urgency_multiplier, 0.03)

    async def calculate_expected_output(self, chain, token_address, amount_in, fee_tier):
        try:
            w3 = self.chains[chain]
            pool_address = await self.get_pool_address(chain, self.weth_addresses[chain], token_address, fee_tier)
            
            if not pool_address:
                return amount_in
            
            pool_contract = w3.eth.contract(address=pool_address, abi=UNISWAP_V3_POOL_ABI)
            slot0 = pool_contract.functions.slot0().call()
            sqrt_price = slot0[0]
            
            price = (sqrt_price / 2**96) ** 2
            expected_out = amount_in * price * (1 - fee_tier / 1000000)
            
            return int(expected_out)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'expected_output_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return amount_in

    async def get_pool_address(self, chain, token0, token1, fee):
        try:
            factory_addresses = {
                'arbitrum': "0x1F98431c8aD98523631AE4a59f267346ea31F984",
                'polygon': "0x1F98431c8aD98523631AE4a59f267346ea31F984",
                'optimism': "0x1F98431c8aD98523631AE4a59f267346ea31F984"
            }
            
            factory_abi = [{
                'constant': True,
                'inputs': [
                    {'name': 'tokenA', 'type': 'address'},
                    {'name': 'tokenB', 'type': 'address'},
                    {'name': 'fee', 'type': 'uint24'}
                ],
                'name': 'getPool',
                'outputs': [{'name': 'pool', 'type': 'address'}],
                'type': 'function'
            }]
            
            w3 = self.chains[chain]
            factory = w3.eth.contract(address=factory_addresses[chain], abi=factory_abi)
            pool_address = factory.functions.getPool(token0, token1, fee).call()
            
            return pool_address if pool_address != '0x0000000000000000000000000000000000000000' else None
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'pool_address_error',
                'chain': chain,
                'error': str(e)
            }))
            return None

    async def select_fee_tier(self, chain, token_address):
        try:
            fee_tiers = [500, 3000, 10000]
            best_liquidity = 0
            best_fee = 3000
            
            for fee in fee_tiers:
                pool_address = await self.get_pool_address(
                    chain, self.weth_addresses[chain], token_address, fee
                )
                
                if pool_address:
                    try:
                        contract = self.chains[chain].eth.contract(address=pool_address, abi=UNISWAP_V3_POOL_ABI)
                        liquidity = contract.functions.liquidity().call()
                        
                        if liquidity > best_liquidity:
                            best_liquidity = liquidity
                            best_fee = fee
                    except:
                        continue
            
            return best_fee
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'fee_tier_selection_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return 3000

    async def optimize_gas_price(self, chain):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f"Bearer {os.getenv('BLOCKNATIVE_API_KEY')}"}
                url = f"https://api.blocknative.com/gasprices"
                params = {'network': self.get_network_id(chain)}
                
                async with session.get(url, headers=headers, params=params, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        block_prices = data.get('blockPrices', [{}])[0]
                        estimated_prices = block_prices.get('estimatedPrices', [])
                        
                        fast_price = next((p['price'] for p in estimated_prices if p['confidence'] == 99), None)
                        
                        if fast_price:
                            return self.chains[chain].to_wei(fast_price, 'gwei')
                        
        except Exception as e:
            logging.error(json.dumps({
                'event': 'gas_optimization_error',
                'chain': chain,
                'error': str(e)
            }))
        
        return self.chains[chain].to_wei(20, 'gwei')

    def get_network_id(self, chain):
        return {'arbitrum': 42161, 'polygon': 137, 'optimism': 10}.get(chain, 1)

    async def send_private_transaction(self, chain, raw_tx):
        try:
            flashbots_endpoints = {
                'arbitrum': "https://rpc.flashbots.net/arbitrum",
                'polygon': "https://rpc.flashbots.net/polygon", 
                'optimism': "https://rpc.flashbots.net/optimism"
            }
            
            if chain in flashbots_endpoints:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'Authorization': f"Bearer {os.getenv('FLASHBOTS_AUTH_KEY')}",
                        'Content-Type': 'application/json'
                    }
                    
                    payload = {
                        'jsonrpc': '2.0',
                        'method': 'eth_sendRawTransaction',
                        'params': [raw_tx.hex()],
                        'id': 1
                    }
                    
                    async with session.post(flashbots_endpoints[chain], headers=headers, json=payload, timeout=10) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            if 'result' in result:
                                return bytes.fromhex(result['result'][2:])
            
            return self.chains[chain].eth.send_raw_transaction(raw_tx)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'private_transaction_error',
                'chain': chain,
                'error': str(e)
            }))
            return self.chains[chain].eth.send_raw_transaction(raw_tx)

    async def monitor_position(self, chain, token_address, entry_score, tx_hash):
        try:
            position_key = tx_hash.hex()
            if position_key not in self.positions:
                return
            
            position = self.positions[position_key]
            entry_price = position['entry_price']
            stop_loss_threshold = entry_score * (1 - self.decay_threshold)
            
            monitoring_interval = self.settings["trading"]["position_monitor_interval"]
            max_holding_time = self.settings["trading"]["max_holding_time"]
            
            while position_key in self.positions:
                current_time = time.time()
                holding_time = current_time - position['entry_time']
                
                if holding_time > max_holding_time:
                    await self.exit_position(chain, token_address, tx_hash, "max_holding_time")
                    break
                
                current_price = await self.get_current_price(chain, token_address)
                current_score = await self.get_current_momentum(chain, token_address)
                
                price_change = (current_price - entry_price) / entry_price
                score_decay = (entry_score - current_score) / entry_score
                
                exit_conditions = [
                    score_decay >= self.decay_threshold,
                    current_score < stop_loss_threshold,
                    price_change < -0.02,
                    self.detect_momentum_reversal(position, current_score, current_price)
                ]
                
                if any(exit_conditions):
                    exit_reason = self.get_exit_reason(exit_conditions)
                    await self.exit_position(chain, token_address, tx_hash, exit_reason)
                    break
                
                await asyncio.sleep(monitoring_interval)
                
        except Exception as e:
            logging.error(json.dumps({
                'event': 'position_monitoring_error',
                'chain': chain,
                'token': token_address,
                'tx_hash': tx_hash.hex(),
                'error': str(e)
            }))

    def detect_momentum_reversal(self, position, current_score, current_price):
        try:
            if 'price_history' not in position:
                position['price_history'] = []
                position['score_history'] = []
            
            position['price_history'].append(current_price)
            position['score_history'].append(current_score)
            
            if len(position['price_history']) > 10:
                position['price_history'] = position['price_history'][-10:]
                position['score_history'] = position['score_history'][-10:]
            
            if len(position['price_history']) >= 5:
                recent_trend = np.polyfit(range(5), position['price_history'][-5:], 1)[0]
                score_trend = np.polyfit(range(5), position['score_history'][-5:], 1)[0]
                
                return recent_trend < -0.001 and score_trend < -0.05
            
            return False
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'reversal_detection_error',
                'error': str(e)
            }))
            return False

    def get_exit_reason(self, exit_conditions):
        reasons = ['score_decay', 'stop_loss', 'price_decline', 'momentum_reversal']
        for i, condition in enumerate(exit_conditions):
            if condition:
                return reasons[i]
        return 'unknown'

    async def exit_position(self, chain, token_address, tx_hash, reason):
        try:
            position_key = tx_hash.hex()
            if position_key not in self.positions:
                return
            
            position = self.positions[position_key]
            
            w3 = self.chains[chain]
            router = w3.eth.contract(address=self.router_addresses[chain], abi=UNISWAP_V3_ROUTER_ABI)
            token_contract = w3.eth.contract(address=token_address, abi=ERC20_ABI)
            
            token_balance = token_contract.functions.balanceOf(self.account.address).call()
            
            if token_balance == 0:
                del self.positions[position_key]
                return
            
            fee_tier = await self.select_fee_tier(chain, token_address)
            deadline = w3.eth.get_block('latest')['timestamp'] + 300
            gas_price = await self.optimize_gas_price(chain)
            
            if not await self.check_token_approval(chain, token_address):
                return

            slippage = self.settings["trading"]["max_slippage"]
            min_eth_out = int(token_balance * 0.98)

            tx_params = {
                'from': self.account.address,
                'nonce': w3.eth.get_transaction_count(self.account.address),
                'gasPrice': gas_price
            }
            
            swap_tx = router.functions.exactInputSingle({
                'tokenIn': token_address,
                'tokenOut': self.weth_addresses[chain],
                'fee': fee_tier,
                'recipient': self.account.address,
                'deadline': deadline,
                'amountIn': token_balance,
                'amountOutMinimum': min_eth_out,
                'sqrtPriceLimitX96': 0
            }).build_transaction(tx_params)
            
            tx_params['gas'] = w3.eth.estimate_gas(swap_tx)

            signed_tx = w3.eth.account.sign_transaction(swap_tx, self.account.key)
            exit_tx_hash = await self.send_private_transaction(chain, signed_tx.rawTransaction)
            
            gas_cost = tx_params['gas'] * gas_price / 1e18
            self.cost_gauge.labels(chain=chain).inc(gas_cost)
            
            exit_price = await self.get_current_price(chain, token_address)
            holding_time = time.time() - position['entry_time']
            
            pnl = self.calculate_pnl(position['entry_price'], exit_price, position['position_size'])
            
            logging.info(json.dumps({
                'event': 'position_exited',
                'chain': chain,
                'token': token_address,
                'entry_tx': tx_hash.hex(),
                'exit_tx': exit_tx_hash.hex(),
                'reason': reason,
                'pnl': pnl,
                'holding_time': holding_time,
                'gas_cost': gas_cost
            }))
            
            del self.positions[position_key]
            
            return {
                'exit_tx': exit_tx_hash,
                'pnl': pnl,
                'holding_time': holding_time,
                'exit_price': exit_price,
                'reason': reason
            }
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'exit_position_error',
                'chain': chain,
                'token': token_address,
                'tx_hash': tx_hash.hex(),
                'error': str(e)
            }))

    def calculate_pnl(self, entry_price, exit_price, position_size):
        try:
            price_change = (exit_price - entry_price) / entry_price
            return position_size * price_change
        except:
            return 0.0

    async def get_current_price(self, chain, token_address):
        try:
            fee_tier = await self.select_fee_tier(chain, token_address)
            pool_address = await self.get_pool_address(
                chain, self.weth_addresses[chain], token_address, fee_tier
            )
            
            if not pool_address:
                return 100.0
            
            contract = self.chains[chain].eth.contract(address=pool_address, abi=UNISWAP_V3_POOL_ABI)
            slot0 = contract.functions.slot0().call()
            sqrt_price = slot0[0]
            
            price = (sqrt_price / 2**96) ** 2
            return price
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'current_price_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return 100.0

    async def get_current_momentum(self, chain, token_address):
        try:
            from backup_20250720_213811.test_signal_detector import backup_20250720_213811.signal_detector
            import redis
            
            redis_client = redis.Redis(host=self.settings['redis']['host'], port=self.settings['redis']['port'], db=0)
            signal_detector = SignalDetector(self.chains, redis_client)
            
            price_data = await signal_detector.get_price_movement(chain, token_address)
            if price_data.empty:
                return 0.5
            
            features = signal_detector.engineer_features(price_data, {'id': token_address, 'swaps': []})
            return signal_detector.calculate_momentum_score(features)
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'momentum_calculation_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return 0.5

    def get_position_summary(self):
        try:
            active_positions = len(self.positions)
            total_exposure = sum(pos['position_size'] for pos in self.positions.values())
            
            chains_active = set(pos['chain'] for pos in self.positions.values())
            
            return {
                'active_positions': active_positions,
                'total_exposure': total_exposure,
                'chains_active': list(chains_active),
                'positions': list(self.positions.keys())
            }
            
        except Exception as e:
            logging.error(json.dumps({
                'event': 'position_summary_error',
                'error': str(e)
            }))
            return {}# Additional API endpoints from api_server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
from prometheus_client import Summary, Counter, generate_latest
from fastapi.responses import Response
import json
import asyncio
import time
from typing import Dict, List, Optional
import yaml

app = FastAPI(title="DeFi Momentum Trading API", version="1.0.0")

class PredictionRequest(BaseModel):
    chain: str
    token_address: str
    features: Dict
    timestamp: Optional[int] = None

class PredictionResponse(BaseModel):
    momentum_score: float
    threshold: float
    uncertainty: float
    entropy: float
    recommendation: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    uptime: float
    model_loaded: bool
    predictions_served: int
    avg_response_time: float

class TradingSignal(BaseModel):
    chain: str
    token_address: str
    signal_strength: float
    confidence: float
    recommended_action: str
    position_size: float
    timestamp: int

predict_time = Summary('predict_request_processing_seconds', 'Time spent processing prediction requests')
prediction_counter = Counter('predictions_total', 'Total predictions made', ['chain', 'recommendation'])
error_counter = Counter('api_errors_total', 'Total API errors', ['error_type'])

start_time = time.time()
model_instance = None
settings = None

@app.on_event("startup")
async def startup_event():
    global model_instance, settings
    try:
        with open('settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        
        from core.models.inference_model import backup_20250720_213811.inference_model
        model_instance = MomentumEnsemble()
        
        logging.info("API server started successfully")
        
    except Exception as e:
        logging.error(f"Startup error: {e}")
        model_instance = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        uptime = time.time() - start_time
        
        predictions_served = sum([
            prediction_counter.labels(chain=chain, recommendation=rec)._value._value
            for chain in ['arbitrum', 'polygon', 'optimism']
            for rec in ['BUY', 'HOLD', 'SELL']
        ])
        
        avg_response_time = predict_time._sum._value / max(predict_time._count._value, 1)
        
        return HealthResponse(
            status="healthy" if model_instance else "degraded",
            uptime=uptime,
            model_loaded=model_instance is not None,
            predictions_served=int(predictions_served),
            avg_response_time=avg_response_time
        )
        
    except Exception as e:
        error_counter.labels(error_type="health_check").inc()
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
@predict_time.time()
async def predict(request: PredictionRequest):
    try:
        if not model_instance:
            error_counter.labels(error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_processing = time.time()
        
        features_df = pd.DataFrame([request.features])
        
        if features_df.empty or len(features_df.columns) < 5:
            error_counter.labels(error_type="insufficient_features").inc()
            raise HTTPException(status_code=400, detail="Insufficient features provided")
        
        momentum_score = model_instance.predict(features_df)
        
        uncertainty = 0.0
        entropy = 0.0
        
        if hasattr(model_instance, 'prediction_history') and model_instance.prediction_history:
            recent_predictions = model_instance.prediction_history[-10:]
            if recent_predictions:
                recent_uncertainties = [p.get('uncertainty', 0) for p in recent_predictions]
                recent_entropies = [p.get('entropy', 0) for p in recent_predictions]
                uncertainty = np.mean(recent_uncertainties)
                entropy = np.mean(recent_entropies)
        
        threshold = model_instance.dynamic_threshold
        
        recommendation = "HOLD"
        if momentum_score > threshold * 1.1:
            recommendation = "BUY"
        elif momentum_score < threshold * 0.8:
            recommendation = "SELL"
        
        processing_time = time.time() - start_processing
        
        prediction_counter.labels(chain=request.chain, recommendation=recommendation).inc()
        
        logging.info(json.dumps({
            'event': 'prediction_served',
            'chain': request.chain,
            'token': request.token_address,
            'momentum_score': momentum_score,
            'recommendation': recommendation,
            'processing_time': processing_time
        }))
        
        return PredictionResponse(
            momentum_score=momentum_score,
            threshold=threshold,
            uncertainty=uncertainty,
            entropy=entropy,
            recommendation=recommendation,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="prediction_error").inc()
        logging.error(json.dumps({
            'event': 'api_prediction_error',
            'chain': request.chain,
            'token': request.token_address,
            'error': str(e)
        }))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    try:
        if not model_instance:
            error_counter.labels(error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if len(requests) > 50:
            error_counter.labels(error_type="batch_too_large").inc()
            raise HTTPException(status_code=400, detail="Batch size too large (max 50)")
        
        results = []
        
        for request in requests:
            try:
                prediction_response = await predict(request)
                results.append({
                    'token_address': request.token_address,
                    'chain': request.chain,
                    'success': True,
                    'prediction': prediction_response.dict()
                })
            except Exception as e:
                results.append({
                    'token_address': request.token_address,
                    'chain': request.chain,
                    'success': False,
                    'error': str(e)
                })
        
        return {'results': results, 'total_processed': len(results)}
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="batch_prediction_error").inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/signals/{chain}")
async def get_trading_signals(chain: str, limit: int = 10):
    try:
        if chain not in ['arbitrum', 'polygon', 'optimism']:
            raise HTTPException(status_code=400, detail="Invalid chain")
        
        signals = []
        
        if model_instance and hasattr(model_instance, 'momentum_scores'):
            recent_scores = model_instance.momentum_scores[-limit:] if model_instance.momentum_scores else []
            
            for i, score in enumerate(recent_scores):
                if score > model_instance.dynamic_threshold:
                    signals.append(TradingSignal(
                        chain=chain,
                        token_address=f"0x{'0'*40}",  # Placeholder
                        signal_strength=score,
                        confidence=min(score / model_instance.dynamic_threshold, 1.0),
                        recommended_action="BUY" if score > model_instance.dynamic_threshold * 1.1 else "WATCH",
                        position_size=0.001 * (score / model_instance.dynamic_threshold),
                        timestamp=int(time.time()) - (len(recent_scores) - i) * 60
                    ))
        
        return {'signals': signals[-limit:], 'chain': chain, 'count': len(signals)}
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="signals_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get signals: {str(e)}")

@app.get("/model/status")
async def get_model_status():
    try:
        if not model_instance:
            return {'status': 'not_loaded', 'error': 'Model instance not available'}
        
        model_summary = model_instance.get_model_summary() if hasattr(model_instance, 'get_model_summary') else {}
        
        return {
            'status': 'loaded',
            'model_info': model_summary,
            'dynamic_threshold': getattr(model_instance, 'dynamic_threshold', 0.75),
            'total_predictions': len(getattr(model_instance, 'momentum_scores', [])),
            'last_retrain': 'unknown',
            'device': str(getattr(model_instance, 'device', 'unknown'))
        }
        
    except Exception as e:
        error_counter.labels(error_type="model_status_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@app.post("/model/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not hasattr(model_instance, 'retrain_if_needed'):
            raise HTTPException(status_code=501, detail="Retrain not supported")
        
        async def retrain_task():
            try:
                await model_instance.retrain_if_needed()
                logging.info("Manual retrain completed successfully")
            except Exception as e:
                logging.error(f"Manual retrain failed: {e}")
        
        background_tasks.add_task(retrain_task)
        
        return {'message': 'Retrain started', 'status': 'in_progress'}
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="retrain_trigger_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to trigger retrain: {str(e)}")

@app.get("/model/threshold")
async def get_threshold():
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            'current_threshold': model_instance.dynamic_threshold,
            'threshold_history': getattr(model_instance, 'threshold_history', []),
            'last_updated': int(time.time())
        }
        
    except Exception as e:
        error_counter.labels(error_type="threshold_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get threshold: {str(e)}")

@app.post("/model/threshold")
async def update_threshold(new_threshold: float):
    try:
        if not model_instance:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not 0.1 <= new_threshold <= 0.95:
            raise HTTPException(status_code=400, detail="Threshold must be between 0.1 and 0.95")
        
        old_threshold = model_instance.dynamic_threshold
        model_instance.dynamic_threshold = new_threshold
        
        logging.info(json.dumps({
            'event': 'threshold_updated_manually',
            'old_threshold': old_threshold,
            'new_threshold': new_threshold
        }))
        
        return {
            'message': 'Threshold updated',
            'old_threshold': old_threshold,
            'new_threshold': new_threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="threshold_update_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to update threshold: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    try:
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type="text/plain")
    except Exception as e:
        error_counter.labels(error_type="metrics_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/performance/{chain}")
async def get_performance_metrics(chain: str, days: int = 1):
    try:
        if chain not in ['arbitrum', 'polygon', 'optimism']:
            raise HTTPException(status_code=400, detail="Invalid chain")
        
        cutoff_time = time.time() - (days * 24 * 3600)
        
        performance_data = {
            'chain': chain,
            'period_days': days,
            'predictions_made': prediction_counter.labels(chain=chain, recommendation='BUY')._value._value + 
                             prediction_counter.labels(chain=chain, recommendation='SELL')._value._value + 
                             prediction_counter.labels(chain=chain, recommendation='HOLD')._value._value,
            'buy_signals': prediction_counter.labels(chain=chain, recommendation='BUY')._value._value,
            'sell_signals': prediction_counter.labels(chain=chain, recommendation='SELL')._value._value,
            'hold_signals': prediction_counter.labels(chain=chain, recommendation='HOLD')._value._value,
            'avg_processing_time': predict_time._sum._value / max(predict_time._count._value, 1),
            'error_rate': sum(error_counter._value._value for error_counter in error_counter._metrics.values()) / 
                         max(predict_time._count._value, 1)
        }
        
        return performance_data
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="performance_metrics_error").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@app.websocket("/ws/predictions")
async def websocket_predictions(websocket):
    await websocket.accept()
    try:
        while True:
            if model_instance and hasattr(model_instance, 'momentum_scores'):
                recent_score = model_instance.momentum_scores[-1] if model_instance.momentum_scores else 0
                
                data = {
                    'timestamp': int(time.time()),
                    'momentum_score': recent_score,
                    'threshold': model_instance.dynamic_threshold,
                    'recommendation': 'BUY' if recent_score > model_instance.dynamic_threshold else 'HOLD'
                }
                
                await websocket.send_json(data)
            
            await asyncio.sleep(5)
            
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)