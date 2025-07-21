from web3 import Web3
import aiohttp
import logging
from abi import UNISWAP_V3_POOL_ABI
from prometheus_client import Gauge
import json
import sqlite3
import yaml

from error_handler import retry_with_backoff, log_performance, CircuitBreaker, safe_execute
from error_handler import TradingSystemError, NetworkError, ModelInferenceError

class SafetyChecker:
    def __init__(self, chains):
        self.chains = chains
        with open('settings.yaml', 'r') as f:
            self.settings = yaml.safe_load(f)
        self.mempool_api = "https://api.eigenphi.io/mempool"
        self.chainalysis_api = "https://api.chainalysis.com/sanctions"
        self.honeypot_api = "https://api.honeypot.is/v2/IsHoneypot"
        self.honeypot_gauge = Gauge('honeypot_risk', 'Honeypot risk score', ['chain', 'token'])
        self.liquidity_gauge = Gauge('liquidity_check', 'Liquidity validation result', ['chain', 'token'])
        self.conn = sqlite3.connect('token_cache.db')

    async def check_token(self, chain, token_address):
        try:
            results = await asyncio.gather(
                self.check_honeypot(chain, token_address),
                self.check_mempool(chain, token_address),
                self.check_liquidity(chain, token_address),
                self.check_contract_integrity(chain, token_address),
                self.check_sanctions(token_address),
                return_exceptions=True
            )
            self.honeypot_gauge.labels(chain=chain, token=token_address).set(1 if results[0] else 0)
            self.liquidity_gauge.labels(chain=chain, token=token_address).set(1 if results[2] else 0)
            if all(isinstance(r, bool) and r for r in results):
                cursor = self.conn.cursor()
                cursor.execute("UPDATE tokens SET last_updated = ? WHERE address = ?", (int(time.time()), token_address))
                self.conn.commit()
                return True
            return False
        except Exception as e:
            logging.error(json.dumps({
                'event': 'safety_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return False

    async def check_honeypot(self, chain, token_address):
        try:
            w3 = self.chains[chain]
            code = w3.eth.get_code(token_address)
            if len(code) < 100:
                return False
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.honeypot_api}?address={token_address}&chainID={self.get_chain_id(chain)}") as resp:
                        data = await resp.json()
                        return data.get('isHoneypot', False) is False
            except:
                contract = w3.eth.contract(address=token_address, abi=UNISWAP_V3_POOL_ABI)
                transfer_func = contract.functions.transfer(w3.eth.default_account, 1).build_transaction({
                    'from': w3.eth.default_account,
                    'gas': 100000
                })
                try:
                    w3.eth.call(transfer_func)
                    return True
                except:
                    return False
        except Exception as e:
            logging.error(json.dumps({
                'event': 'honeypot_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return False

    async def check_mempool(self, chain, token_address):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.mempool_api}/?address={token_address}&chain={chain}&apiKey={os.getenv('BLOCKNATIVE_API_KEY')}") as resp:
                    data = await resp.json()
                    pending_txs = data.get('pending', [])
                    return len(pending_txs) < self.settings['safety']['max_pending_txs'] and not self.detect_frontrunning(pending_txs)
        except Exception as e:
            logging.error(json.dumps({
                'event': 'mempool_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return False

    def detect_frontrunning(self, pending_txs):
        gas_prices = [tx.get('gasPrice', 0) for tx in pending_txs]
        return len(gas_prices) > 0 and max(gas_prices) > np.percentile(gas_prices, 95)

    async def check_liquidity(self, chain, token_address):
        try:
            contract = self.chains[chain].eth.contract(address=token_address, abi=UNISWAP_V3_POOL_ABI)
            liquidity = contract.functions.liquidity().call()
            price_data = pd.Series(await self.fetch_historical_prices(chain, token_address)).pct_change().dropna()
            volatility = price_data.std() * np.sqrt(252)
            fee_tier = contract.functions.fee().call()
            dynamic_threshold = self.settings['safety']['min_liquidity'] * (1 + volatility) * (1 if fee_tier == 500 else 0.5 if fee_tier == 3000 else 0.25)
            return liquidity > dynamic_threshold
        except Exception as e:
            logging.error(json.dumps({
                'event': 'liquidity_check_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return False

    async def check_contract_integrity(self, chain, token_address):
        try:
            code = self.chains[chain].eth.get_code(token_address)
            malicious_patterns = [b'\x60\x60\x60\x40\x52', b'\x33\xff']
            return not any(pattern in code for pattern in malicious_patterns)
        except Exception as e:
            logging.error(json.dumps({
                'event': 'contract_integrity_error',
                'chain': chain,
                'token': token_address,
                'error': str(e)
            }))
            return False

    async def check_sanctions(self, token_address):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f"Bearer {os.getenv('CHAINALYSIS_API_KEY')}"}
                async with session.get(f"{self.chainalysis_api}/address/{token_address}") as resp:
                    data = await resp.json()
                    return not data.get('isSanctioned', False)
        except Exception as e:
            logging.error(json.dumps({
                'event': 'sanctions_check_error',
                'token': token_address,
                'error': str(e)
            }))
            return False

    async def fetch_historical_prices(self, chain, pool_address):
        from signal_detector import SignalDetector
        signal_detector = SignalDetector(self.chains, redis.Redis(host=self.settings['redis']['host'], port=self.settings['redis']['port'], db=0))
        return await signal_detector.fetch_historical_prices(chain, pool_address)

    def get_chain_id(self, chain):
        return {'arbitrum': 42161, 'polygon': 137, 'optimism': 10}.get(chain, 1)

    def __del__(self):
        self.conn.close()