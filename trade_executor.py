from web3 import Web3
from eth_account import Account
import os
import aiohttp
import time
import logging
from abi import UNISWAP_V3_ROUTER_ABI, UNISWAP_V3_POOL_ABI, ERC20_ABI
from prometheus_client import Counter, Gauge, Histogram
import json
import yaml
import asyncio
import numpy as np
import pandas as pd

from error_handler import retry_with_backoff, log_performance, CircuitBreaker, safe_execute
from error_handler import TradingSystemError, NetworkError, ModelInferenceError

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
            from signal_detector import SignalDetector
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
            return {}